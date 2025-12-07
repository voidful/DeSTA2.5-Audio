"""
OCAR-Qwen Model Architecture

Implements the Orthogonal Complementary Acoustic Residual Qwen architecture:
- HybridOCARAdapter: Global (Q-Former) + Local (Conv1d) branches
- GatedCrossAttention: Cross-attention with learnable gate initialized to 0
- CARQwen2DecoderLayer: Modified decoder layer with cross-attention injection
- OCARQwenForCausalLM: Main model with OCAR loss computation
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import OrderedDict

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from transformers.models.whisper import WhisperForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


class OCARQwenConfig(PretrainedConfig):
    """
    Configuration for OCAR-Qwen model.
    """
    model_type = "ocar_qwen"
    
    def __init__(
        self,
        llm_model_id: str = "Qwen/Qwen2-4B-Instruct",
        encoder_model_id: str = "openai/whisper-large-v3",
        # Adapter configs
        global_tokens: int = 32,
        local_stride: int = 4,
        qformer_layers: int = 2,
        # Cross-attention config
        cross_attn_num_heads: int = 16,
        cross_attn_gate_init: float = 0.0,
        # Loss weights
        w_llm: float = 1.0,
        w_ortho_text: float = 0.1,
        w_ortho_self: float = 0.1, 
        w_prosody_global: float = 0.25,
        w_prosody_local: float = 0.25,
        # Target Whisper layers for Q-Former
        target_layer_ids: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        
        self.global_tokens = global_tokens
        self.local_stride = local_stride
        self.qformer_layers = qformer_layers
        
        self.cross_attn_num_heads = cross_attn_num_heads
        self.cross_attn_gate_init = cross_attn_gate_init
        
        self.w_llm = w_llm
        self.w_ortho_text = w_ortho_text
        self.w_ortho_self = w_ortho_self
        self.w_prosody_global = w_prosody_global
        self.w_prosody_local = w_prosody_local
        
        # Default target layers for whisper-large-v3
        self.target_layer_ids = target_layer_ids or [7, 15, 23, 31]
        
        # Load sub-model configs
        cache_dir = os.getenv("HF_HOME")
        self.llm_config = AutoConfig.from_pretrained(llm_model_id, cache_dir=cache_dir)
        self.encoder_config = AutoConfig.from_pretrained(encoder_model_id, cache_dir=cache_dir)


class GatedCrossAttention(nn.Module):
    """
    Cross-attention module with learnable gate initialized to 0.
    Text (Query) attends to Local Audio (Key/Value).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        gate_init: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layer norm for audio input
        self.audio_layer_norm = nn.LayerNorm(hidden_size)
        
        # Learnable gate initialized to gate_init (default 0)
        self.gate = nn.Parameter(torch.tensor([gate_init]))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_kv: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention.
        
        Args:
            hidden_states: Text hidden states (B, T_text, D)
            audio_kv: Local audio features (B, T_audio, D)
            audio_attention_mask: Mask for valid audio positions (B, T_audio)
            
        Returns:
            Updated hidden states (B, T_text, D)
        """
        batch_size, text_len, _ = hidden_states.shape
        audio_len = audio_kv.shape[1]
        
        # Normalize audio
        audio_kv = self.audio_layer_norm(audio_kv)
        
        # Project Q, K, V
        Q = self.q_proj(hidden_states)  # (B, T_text, D)
        K = self.k_proj(audio_kv)       # (B, T_audio, D)
        V = self.v_proj(audio_kv)       # (B, T_audio, D)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply audio attention mask
        if audio_attention_mask is not None:
            # audio_attention_mask: (B, T_audio) -> (B, 1, 1, T_audio)
            mask = audio_attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, text_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Apply gate (tanh to bound between -1 and 1)
        gated_output = self.gate.tanh() * attn_output
        
        # Residual connection
        return hidden_states + gated_output


class CARQwen2DecoderLayer(nn.Module):
    """
    Modified Qwen2 decoder layer with cross-attention injection.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        num_heads: int,
        gate_init: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.cross_attn = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            gate_init=gate_init,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_kv: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with cross-attention injection.
        """
        # Run original layer (self-attention + FFN)
        outputs = self.original_layer(hidden_states, **kwargs)
        hidden_states = outputs[0]
        
        # Inject cross-attention to local audio features
        if audio_kv is not None:
            hidden_states = self.cross_attn(
                hidden_states, 
                audio_kv, 
                audio_attention_mask
            )
        
        return (hidden_states,) + outputs[1:]


class HybridOCARAdapter(nn.Module):
    """
    Hybrid adapter with Global (Q-Former) and Local (Conv1d) branches.
    
    Global Branch: Captures static style via Q-Former attending to Whisper layers
    Local Branch: Captures dynamic rhythm via Conv1d preserving temporal structure
    """
    
    def __init__(self, config: OCARQwenConfig):
        super().__init__()
        
        self.config = config
        
        whisper_dim = config.encoder_config.d_model  # 1280 for whisper-large-v3
        llm_dim = config.llm_config.hidden_size
        
        # === Global Branch (Q-Former) ===
        self.global_queries = nn.Parameter(
            torch.randn(1, config.global_tokens, whisper_dim) * 0.02
        )
        
        # Q-Former using BERT encoder with cross-attention
        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = config.qformer_layers
        qformer_config.num_attention_heads = 16
        qformer_config.hidden_size = whisper_dim
        qformer_config.intermediate_size = whisper_dim * 4
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True
        qformer_config._attn_implementation = "eager"
        
        self.qformer = BertEncoder(qformer_config)
        
        # Layer-wise attention weights
        self.layer_weights = nn.Parameter(
            torch.zeros(config.global_tokens, len(config.target_layer_ids))
        )
        
        # Global projection to LLM dim
        self.global_proj = nn.Sequential(
            nn.LayerNorm(whisper_dim),
            nn.Linear(whisper_dim, llm_dim),
        )
        
        # Global prosody head: predict F0/Energy mean/std (4 values)
        self.global_prosody_head = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, 4),  # F0_mean, F0_std, Energy_mean, Energy_std
        )
        
        # === Local Branch (Conv1d) ===
        self.local_conv = nn.Conv1d(
            in_channels=whisper_dim,
            out_channels=llm_dim,
            kernel_size=config.local_stride,
            stride=config.local_stride,
            padding=0,
        )
        self.local_layer_norm = nn.LayerNorm(llm_dim)
        
        # Local prosody head: predict F0, Energy per timestep
        self.local_prosody_head = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 4),
            nn.GELU(),
            nn.Linear(llm_dim // 4, 2),  # F0, Energy
        )
        
    def forward(
        self,
        encoder_hidden_states: List[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches.
        
        Args:
            encoder_hidden_states: List of hidden states from Whisper layers
            audio_attention_mask: Mask for valid audio positions (B, T_encoder)
            
        Returns:
            Dictionary with global_tokens, local_tokens, prosody predictions, and local_mask
        """
        batch_size = encoder_hidden_states[0].shape[0]
        
        # === Global Branch ===
        # Collect outputs from target layers
        layer_outputs = []
        for layer_idx in self.config.target_layer_ids:
            if layer_idx < len(encoder_hidden_states):
                hidden_state = encoder_hidden_states[layer_idx]
                
                # Expand queries for batch
                queries = self.global_queries.expand(batch_size, -1, -1)
                
                # Q-Former cross-attention
                qformer_output = self.qformer(
                    hidden_states=queries,
                    encoder_hidden_states=hidden_state,
                )
                layer_outputs.append(qformer_output.last_hidden_state)
        
        # Stack and weight layers
        if layer_outputs:
            layer_outputs = torch.stack(layer_outputs, dim=2)  # (B, tokens, layers, dim)
            norm_weights = F.softmax(self.layer_weights, dim=-1).unsqueeze(-1)  # (tokens, layers, 1)
            global_features = (layer_outputs * norm_weights).sum(dim=2)  # (B, tokens, dim)
        else:
            global_features = self.global_queries.expand(batch_size, -1, -1)
        
        # Project to LLM dimension
        global_tokens = self.global_proj(global_features)  # (B, global_tokens, llm_dim)
        
        # Global prosody prediction (pool then predict)
        global_pooled = global_tokens.mean(dim=1)  # (B, llm_dim)
        global_prosody_pred = self.global_prosody_head(global_pooled)  # (B, 4)
        
        # === Local Branch ===
        # Use the last encoder layer
        local_input = encoder_hidden_states[-1]  # (B, T_encoder, whisper_dim)
        
        # Conv1d expects (B, C, T)
        local_input = local_input.transpose(1, 2)  # (B, whisper_dim, T_encoder)
        local_features = self.local_conv(local_input)  # (B, llm_dim, T_local)
        local_features = local_features.transpose(1, 2)  # (B, T_local, llm_dim)
        
        local_tokens = self.local_layer_norm(local_features)  # (B, T_local, llm_dim)
        
        # Local prosody prediction
        local_prosody_pred = self.local_prosody_head(local_tokens)  # (B, T_local, 2)
        
        # Compute local attention mask
        local_mask = None
        if audio_attention_mask is not None:
            # Downsample mask by stride
            # audio_attention_mask: (B, T_encoder)
            mask_float = audio_attention_mask.float().unsqueeze(1)  # (B, 1, T_encoder)
            local_mask = F.avg_pool1d(
                mask_float, 
                kernel_size=self.config.local_stride, 
                stride=self.config.local_stride
            ).squeeze(1) > 0.5  # (B, T_local)
        
        return {
            "global_tokens": global_tokens,
            "local_tokens": local_tokens,
            "global_prosody_pred": global_prosody_pred,
            "local_prosody_pred": local_prosody_pred,
            "local_attention_mask": local_mask,
        }


class OCARQwenForCausalLM(PreTrainedModel):
    """
    OCAR-Qwen model for causal language modeling with audio conditioning.
    
    Implements:
    - Hybrid injection: Global tokens spliced, Local tokens via cross-attention
    - OCAR losses: LLM, Ortho_Text, Ortho_Self, Prosody
    """
    
    config_class = OCARQwenConfig
    
    def __init__(self, config: OCARQwenConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        self.config = config
        cache_dir = os.getenv("HF_HOME")
        
        # Load Whisper encoder (frozen)
        print(f"Loading Whisper encoder from {config.encoder_model_id}")
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            config.encoder_model_id,
            cache_dir=cache_dir,
        )
        
        # Load Qwen LLM
        print(f"Loading LLM from {config.llm_model_id}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        
        # Hybrid adapter
        self.adapter = HybridOCARAdapter(config)
        
        # Replace Qwen decoder layers with CAR variants
        self._inject_cross_attention()
        
        # Freeze Whisper
        self._freeze_whisper()
        
    def _freeze_whisper(self):
        """Freeze all Whisper parameters."""
        for param in self.whisper.parameters():
            param.requires_grad = False
            
    def _inject_cross_attention(self):
        """Replace Qwen decoder layers with CAR variants."""
        hidden_size = self.config.llm_config.hidden_size
        num_heads = self.config.cross_attn_num_heads
        gate_init = self.config.cross_attn_gate_init
        
        # Access the decoder layers
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
            layers = self.llm.model.layers
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'h'):
            layers = self.llm.transformer.h
        else:
            raise ValueError("Could not find decoder layers in LLM model")
        
        # Store CAR layers separately to track them
        self.car_layers = nn.ModuleList()
        
        for i, layer in enumerate(layers):
            car_layer = CARQwen2DecoderLayer(
                original_layer=layer,
                hidden_size=hidden_size,
                num_heads=num_heads,
                gate_init=gate_init,
            )
            self.car_layers.append(car_layer)
            
    def _forward_whisper_encoder(
        self,
        audio_values: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward through Whisper encoder and collect layer-wise hidden states.
        
        Args:
            audio_values: Mel spectrogram features (B, 128, 3000)
            
        Returns:
            List of hidden states from each encoder layer
        """
        encoder = self.whisper.model.encoder
        
        # Initial convolutions
        inputs_embeds = F.gelu(encoder.conv1(audio_values))
        inputs_embeds = F.gelu(encoder.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, T, D)
        
        # Add positional embeddings
        embed_pos = encoder.embed_positions.weight[:inputs_embeds.shape[1], :]
        hidden_states = inputs_embeds + embed_pos
        
        # Collect hidden states from all layers
        all_hidden_states = [hidden_states]
        
        for layer in encoder.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_output[0]
            all_hidden_states.append(hidden_states)
            
        return all_hidden_states
    
    def _compute_ortho_text_loss(
        self,
        audio_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between audio features and text embeddings.
        Minimizes cosine similarity to force disentanglement.
        
        Args:
            audio_features: Audio feature vectors (B, D)
            text_embeddings: Text embedding vectors (B, D)
            
        Returns:
            Orthogonality loss scalar
        """
        # Normalize
        audio_norm = F.normalize(audio_features, dim=-1)
        text_norm = F.normalize(text_embeddings, dim=-1)
        
        # Cosine similarity
        cos_sim = (audio_norm * text_norm).sum(dim=-1)
        
        # Minimize absolute cosine similarity
        return cos_sim.abs().mean()
    
    def _compute_ortho_self_loss(
        self,
        audio_queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self-orthogonality loss using Gram matrix regularization.
        Encourages audio queries to be orthogonal to each other.
        
        Args:
            audio_queries: Audio query vectors (B, N, D)
            
        Returns:
            Self-orthogonality loss scalar
        """
        batch_size, num_queries, dim = audio_queries.shape
        
        # Normalize queries
        queries_norm = F.normalize(audio_queries, dim=-1)
        
        # Compute Gram matrix: (B, N, N)
        gram = torch.bmm(queries_norm, queries_norm.transpose(1, 2))
        
        # Target: identity matrix (orthogonal)
        identity = torch.eye(num_queries, device=gram.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # MSE loss
        return F.mse_loss(gram, identity)
    
    def _compute_prosody_loss(
        self,
        global_prosody_pred: torch.Tensor,
        global_stats: torch.Tensor,
        local_prosody_pred: torch.Tensor,
        local_f0: torch.Tensor,
        local_energy: torch.Tensor,
        local_prosody_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prosody prediction losses.
        
        Args:
            global_prosody_pred: Predicted global stats (B, 4)
            global_stats: Target global stats (B, 4)
            local_prosody_pred: Predicted local contours (B, T, 2)
            local_f0: Target F0 contour (B, T)
            local_energy: Target Energy contour (B, T)
            local_prosody_mask: Valid positions mask (B, T)
            
        Returns:
            Tuple of (global_loss, local_loss)
        """
        # Global prosody loss (MSE)
        global_loss = F.mse_loss(global_prosody_pred, global_stats)
        
        # Local prosody loss (Masked MSE)
        # Align lengths
        pred_len = local_prosody_pred.shape[1]
        target_len = local_f0.shape[1]
        
        if pred_len != target_len:
            # Interpolate predictions or targets to match
            local_prosody_pred = F.interpolate(
                local_prosody_pred.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
        # Stack targets
        local_targets = torch.stack([local_f0, local_energy], dim=-1)  # (B, T, 2)
        
        # Masked MSE
        if local_prosody_mask is not None:
            mask = local_prosody_mask.unsqueeze(-1).expand_as(local_targets)
            diff = (local_prosody_pred - local_targets) ** 2
            masked_diff = diff * mask.float()
            local_loss = masked_diff.sum() / (mask.float().sum() + 1e-8)
        else:
            local_loss = F.mse_loss(local_prosody_pred, local_targets)
            
        return global_loss, local_loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        global_stats: Optional[torch.Tensor] = None,
        local_f0: Optional[torch.Tensor] = None,
        local_energy: Optional[torch.Tensor] = None,
        local_prosody_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with OCAR loss computation.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get text embeddings
        if hasattr(self.llm, 'model'):
            text_embeddings = self.llm.model.embed_tokens(input_ids)
        else:
            text_embeddings = self.llm.transformer.wte(input_ids)
        
        # Initialize adapter outputs
        adapter_output = None
        global_tokens = None
        local_tokens = None
        local_attention_mask = None
        
        # Process audio if provided
        if audio_values is not None:
            audio_values = audio_values.to(device)
            
            # Forward through Whisper encoder
            with torch.no_grad():
                encoder_hidden_states = self._forward_whisper_encoder(audio_values)
            
            # Forward through adapter
            adapter_output = self.adapter(
                encoder_hidden_states,
                audio_attention_mask
            )
            
            global_tokens = adapter_output["global_tokens"]
            local_tokens = adapter_output["local_tokens"]
            local_attention_mask = adapter_output["local_attention_mask"]
            
            # Splice global tokens (prepend to text embeddings)
            inputs_embeds = torch.cat([global_tokens, text_embeddings], dim=1)
            
            # Update attention mask for global tokens
            if attention_mask is not None:
                global_mask = torch.ones(
                    batch_size, global_tokens.shape[1], 
                    dtype=attention_mask.dtype, 
                    device=device
                )
                attention_mask = torch.cat([global_mask, attention_mask], dim=1)
                
            # Update labels for global tokens (ignore in loss)
            if labels is not None:
                global_labels = torch.full(
                    (batch_size, global_tokens.shape[1]), 
                    -100, 
                    dtype=labels.dtype, 
                    device=device
                )
                labels = torch.cat([global_labels, labels], dim=1)
        else:
            inputs_embeds = text_embeddings
            
        # Forward through CAR decoder layers
        hidden_states = inputs_embeds
        
        # Get position ids
        seq_length = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_length, device)
        
        # Forward through each CAR layer
        for car_layer in self.car_layers:
            layer_outputs = car_layer(
                hidden_states,
                audio_kv=local_tokens,
                audio_attention_mask=local_attention_mask,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]
            
        # Final layer norm
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'norm'):
            hidden_states = self.llm.model.norm(hidden_states)
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'ln_f'):
            hidden_states = self.llm.transformer.ln_f(hidden_states)
            
        # LM head
        logits = self.llm.lm_head(hidden_states)
        
        # Compute losses
        loss = None
        loss_dict = {}
        
        if labels is not None:
            # LLM loss (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_llm = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss_dict["loss_llm"] = loss_llm.item()
            
            loss = self.config.w_llm * loss_llm
            
            # OCAR losses (only when audio is provided)
            if adapter_output is not None:
                # Ortho-Text loss
                audio_pooled = global_tokens.mean(dim=1)
                text_pooled = text_embeddings.mean(dim=1)
                loss_ortho_text = self._compute_ortho_text_loss(audio_pooled, text_pooled)
                loss_dict["loss_ortho_text"] = loss_ortho_text.item()
                loss = loss + self.config.w_ortho_text * loss_ortho_text
                
                # Ortho-Self loss
                loss_ortho_self = self._compute_ortho_self_loss(global_tokens)
                loss_dict["loss_ortho_self"] = loss_ortho_self.item()
                loss = loss + self.config.w_ortho_self * loss_ortho_self
                
                # Prosody losses
                if global_stats is not None and local_f0 is not None:
                    global_stats = global_stats.to(device)
                    local_f0 = local_f0.to(device)
                    local_energy = local_energy.to(device)
                    if local_prosody_mask is not None:
                        local_prosody_mask = local_prosody_mask.to(device)
                    
                    loss_prosody_global, loss_prosody_local = self._compute_prosody_loss(
                        adapter_output["global_prosody_pred"],
                        global_stats,
                        adapter_output["local_prosody_pred"],
                        local_f0,
                        local_energy,
                        local_prosody_mask,
                    )
                    loss_dict["loss_prosody_global"] = loss_prosody_global.item()
                    loss_dict["loss_prosody_local"] = loss_prosody_local.item()
                    
                    loss = loss + self.config.w_prosody_global * loss_prosody_global
                    loss = loss + self.config.w_prosody_local * loss_prosody_local
                    
            loss_dict["loss"] = loss.item()
                    
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )
    
    def _create_causal_mask(
        self, 
        seq_length: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device),
            diagonal=1
        ).bool()
        return ~mask
    
    def get_trainable_parameters(self) -> List[str]:
        """Get list of trainable parameter names."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        return trainable
    
    def configure_trainable_parameters(self):
        """Configure which parameters are trainable."""
        # Freeze Whisper (already done in __init__)
        for param in self.whisper.parameters():
            param.requires_grad = False
            
        # Freeze base LLM parameters (LoRA will be applied separately)
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Make adapter trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        # Make cross-attention layers trainable
        for car_layer in self.car_layers:
            for param in car_layer.cross_attn.parameters():
                param.requires_grad = True
                
    def state_dict(self) -> OrderedDict:
        """Return only trainable parameters."""
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()
        return trainable_state_dict
    
    def setup_for_training(self):
        """Setup tokenizer and processor for training."""
        cache_dir = os.getenv("HF_HOME")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_id, 
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.encoder_model_id,
            cache_dir=cache_dir
        )
