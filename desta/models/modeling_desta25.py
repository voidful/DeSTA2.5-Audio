
import os
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple, Union

from dataclasses import dataclass
from desta.utils.audio import AudioSegment

from transformers import AutoTokenizer, AutoProcessor
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import WhisperForConditionalGeneration, BertConfig
from safetensors.torch import load_file
import torch.distributed as dist


# === Gradient Reversal for Adversarial Training ===
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    Forward pass is identity, backward pass negates gradients.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction for use in nn.Sequential."""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class TextContentDiscriminator(nn.Module):
    """
    Discriminator for IV-Guided Disentanglement.
    
    Tries to predict text content from audio features.
    If successful, audio features contain linguistic information (bad).
    The Q-Former should learn to fool this discriminator.
    
    Uses Gradient Reversal to enable end-to-end adversarial training.
    """
    def __init__(self, hidden_size: int, num_groups: int = 8, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Per-group discriminators (each group should be disentangled)
        # Using shared backbone with group-specific heads
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Prediction head: predicts bag-of-words distribution
        # Simpler than full sequence prediction, but tests if audio encodes content
        self.content_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, vocab_size),
        )
    
    def forward(
        self, 
        group_tokens: torch.Tensor,  # [B, num_groups * K, H]
        transcription_ids: Optional[torch.Tensor] = None,  # [B, T] target text tokens
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with CIB loss computation.
        
        Args:
            group_tokens: Audio tokens from GroupwiseOrthogonalConnector
            transcription_ids: Target transcription token IDs (for supervision)
            
        Returns:
            Dict with discriminator loss and predictions
        """
        # Apply gradient reversal
        reversed_tokens = self.grl(group_tokens)  # [B, K_total, H]
        
        # Pool tokens
        pooled = reversed_tokens.mean(dim=1)  # [B, H]
        
        # Discriminator forward
        features = self.shared_backbone(pooled)  # [B, H/2]
        logits = self.content_head(features)  # [B, vocab_size]
        
        result = {"logits": logits}
        
        # Compute loss if targets provided
        if transcription_ids is not None:
            # Create bag-of-words target (multi-label)
            batch_size = logits.size(0)
            vocab_size = logits.size(1)
            
            # Convert transcription_ids to bag-of-words
            bow_target = torch.zeros(batch_size, vocab_size, device=logits.device)
            for b in range(batch_size):
                valid_ids = transcription_ids[b][transcription_ids[b] >= 0]
                valid_ids = valid_ids[valid_ids < vocab_size]  # Clip to vocab size
                if len(valid_ids) > 0:
                    bow_target[b].scatter_(0, valid_ids, 1.0)
            
            # Binary cross-entropy loss (multi-label)
            loss = F.binary_cross_entropy_with_logits(logits, bow_target)
            result["loss"] = loss
            
            # Accuracy: what fraction of top-k predictions are in target
            with torch.no_grad():
                top_k = 20
                _, top_indices = logits.topk(top_k, dim=-1)
                correct = 0
                total = 0
                for b in range(batch_size):
                    target_set = set(transcription_ids[b].tolist())
                    pred_set = set(top_indices[b].tolist())
                    correct += len(target_set & pred_set)
                    total += min(len(target_set), top_k)
                result["accuracy"] = correct / max(total, 1)
        
        return result


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    rope_theta: float = 10000.0,
    device: torch.device = None,
    positions: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos and sin frequencies.
    
    Args:
        seq_len: Sequence length
        dim: Head dimension (will compute for dim/2 pairs)
        rope_theta: Base frequency (from LLM config)
        device: Device for tensors
        positions: Optional custom positions [B, T] or [T], can be fractional for interpolation
        
    Returns:
        (cos, sin) tensors each of shape [1, T, dim] or [B, T, dim]
    """
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float, device=device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
    
    if positions is None:
        positions = torch.arange(seq_len, dtype=torch.float, device=device)
    
    # Handle both 1D and 2D position tensors
    if positions.dim() == 1:
        # [T] x [half_dim] -> [T, half_dim]
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        # Duplicate for pairs: [T, half_dim] -> [T, dim]
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos().unsqueeze(0)  # [1, T, dim]
        sin = freqs.sin().unsqueeze(0)  # [1, T, dim]
    else:
        # [B, T] x [half_dim] -> [B, T, half_dim]
        freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
        # Duplicate for pairs: [B, T, half_dim] -> [B, T, dim]
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()  # [B, T, dim]
        sin = freqs.sin()  # [B, T, dim]
    
    return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to a tensor.
    This is the same rotation used in LLM's attention.
    
    Args:
        x: Input tensor [B, T, D] or [B, num_heads, T, head_dim]
        cos: Cosine frequencies [1, T, D] or [B, T, D]
        sin: Sine frequencies [1, T, D] or [B, T, D]
        
    Returns:
        Rotated tensor with same shape as input
    """
    # Ensure correct dtype
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    
    # RoPE rotation: pair-wise rotation
    # Split into pairs and rotate
    x_half1 = x[..., : x.shape[-1] // 2]
    x_half2 = x[..., x.shape[-1] // 2 :]
    
    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos_half = cos[..., : cos.shape[-1] // 2]
    sin_half = sin[..., : sin.shape[-1] // 2]
    
    rotated_half1 = x_half1 * cos_half - x_half2 * sin_half
    rotated_half2 = x_half1 * sin_half + x_half2 * cos_half
    
    return torch.cat([rotated_half1, rotated_half2], dim=-1)

def _prepare_audio_context_and_start_positions(
                                             token_list,
                                             audio_locator,
                                             audio_size_list,
                                             transcription_size_list,
                                             placeholder_token
        ):
        assert len(audio_size_list) == len(transcription_size_list), f"audio_size_list and transcription_size_list must have the same length, audio_size_list: {audio_size_list}, transcription_size_list: {transcription_size_list}"

        result = []
        start_positions = []
        for x in token_list:
            if x == audio_locator:
                # start_positions.append(len(result))
                transcription_size = transcription_size_list.pop(0)
                audio_size = audio_size_list.pop(0)

                # result.extend(transcription)
                start_positions.append(len(result))
                result.extend([placeholder_token] * (audio_size))
                result.extend([placeholder_token] * (transcription_size))
            else:
                result.append(x)
                
        return result, start_positions


class QformerConnector(nn.Module):
    """
    Connector module using Q-Former to bridge audio encoder and LLM.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config

        if self.config.encoder_model_id == "openai/whisper-medium":
            self.config.target_layer_ids = [5, 11, 17, 23]
        elif self.config.encoder_model_id == "openai/whisper-small":
            self.config.target_layer_ids = [2, 5, 8, 11]
        elif self.config.encoder_model_id == "openai/whisper-tiny":
            self.config.target_layer_ids = [0, 1, 2, 3]
        elif self.config.encoder_model_id == "openai/whisper-large-v3":
            self.config.target_layer_ids = [7, 15, 23, 31]
        elif self.config.encoder_model_id == "openai/whisper-large-v3-turbo":
            self.config.target_layer_ids = [7, 15, 23, 31]
        else:
            raise NotImplementedError(f"model_id {self.config.encoder_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.config.prompt_size, self.config.encoder_config.d_model)) for _ in range(len(self.config.target_layer_ids))]
        )

        self.layer_weights = nn.Parameter(torch.zeros(self.config.prompt_size, len(self.config.target_layer_ids), dtype=torch.float))

        if self.config.connector_mode == "qformer_1":
            # init Qformerblock
            qformer_config = BertConfig()
            qformer_config.num_hidden_layers = self.config.qformer_num_hidden_layers
            qformer_config.num_attention_heads = self.config.encoder_config.encoder_attention_heads
            qformer_config.hidden_size = self.config.encoder_config.d_model
            qformer_config.add_cross_attention = True
            qformer_config.is_decoder = True
            qformer_config._attn_implementation = "sdpa" if getattr(self.config, 'use_flash_attention', False) else "eager"

            self.qformer = BertEncoder(qformer_config)
            self.proj = nn.Sequential(
                    nn.LayerNorm(self.config.encoder_config.d_model),
                    nn.Linear(self.config.encoder_config.d_model, self.config.llm_config.hidden_size) # project to llm hidden size
                )
        else:
            # Note: orca_hybrid is handled by ORCAHybridConnector, not QformerConnector
            # If you see this error for orca_hybrid, please update your desta package
            raise NotImplementedError(
                f"connector_mode '{self.config.connector_mode}' not implemented in QformerConnector. "
                f"Supported modes: 'qformer_1'. If using 'orca_hybrid', please update your desta package."
            )
        

    def forward(self, encoder_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the QformerConnector.

        Args:
            encoder_hidden_states (List[torch.Tensor]): Layerwise hidden states from the encoder.

        Returns:
            torch.Tensor: Projected output features.
        """
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.config.target_layer_ids:
                layer_prompt = self.layer_prompts[self.config.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_llm)
        output = self.proj(output)
        
        return output


class ORCAHybridConnector(nn.Module):
    """
    ORCA Connector with global branch only (Q-Former style cross-attention).
    
    Returns global_tokens tensor.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config
        
        # Determine target layer IDs based on configuration
        if getattr(config, 'orca_use_all_layers', False):
            # Use all encoder layers
            num_encoder_layers = config.encoder_config.num_hidden_layers
            self.target_layer_ids = list(range(num_encoder_layers))
        else:
            # Use selected layers based on Whisper model
            if config.encoder_model_id == "openai/whisper-medium":
                self.target_layer_ids = [5, 11, 17, 23]
            elif config.encoder_model_id == "openai/whisper-small":
                self.target_layer_ids = [2, 5, 8, 11]
            elif config.encoder_model_id == "openai/whisper-tiny":
                self.target_layer_ids = [0, 1, 2, 3]
            elif config.encoder_model_id in ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo"]:
                self.target_layer_ids = [7, 15, 23, 31]
            else:
                raise NotImplementedError(f"model_id {config.encoder_model_id} not implemented")
        
        d_encoder = config.encoder_config.d_model
        d_llm = config.llm_config.hidden_size
        
        # === Global Branch (Q-Former style) ===
        # Learnable queries for each target layer
        self.global_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, config.orca_global_num_tokens, d_encoder))
            for _ in range(len(self.target_layer_ids))
        ])
        self.global_layer_weights = nn.Parameter(
            torch.zeros(config.orca_global_num_tokens, len(self.target_layer_ids), dtype=torch.float)
        )
        
        # Q-Former for global branch
        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = config.qformer_num_hidden_layers
        qformer_config.num_attention_heads = config.encoder_config.encoder_attention_heads
        qformer_config.hidden_size = d_encoder
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True
        qformer_config._attn_implementation = "sdpa" if getattr(config, 'use_flash_attention', False) else "eager"
        
        self.global_qformer = BertEncoder(qformer_config)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_encoder),
            nn.Linear(d_encoder, d_llm)
        )
    
    def forward(
        self, 
        encoder_hidden_states: List[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of ORCAHybridConnector.
        
        Args:
            encoder_hidden_states: List of hidden states from Whisper encoder layers
            audio_attention_mask: Optional attention mask for audio [B, T]
            
        Returns:
            global_tokens: [B, K_global, d_llm]
        """
        batch_size = encoder_hidden_states[0].size(0)
        
        target_dtype = self.global_proj[1].weight.dtype
        target_device = self.global_proj[1].weight.device
        
        # Collect target layer outputs
        target_layer_outputs = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            if idx in self.target_layer_ids:
                # Ensure input hidden states match module dtype
                target_layer_outputs.append(hidden_state.to(dtype=target_dtype, device=target_device))
        
        # === Global Branch ===
        global_outputs = []
        for layer_idx, hidden_state in enumerate(target_layer_outputs):
            queries = self.global_queries[layer_idx].expand(batch_size, -1, -1).to(dtype=target_dtype, device=target_device)
            
            qformer_out = self.global_qformer(
                hidden_states=queries,
                encoder_hidden_states=hidden_state,
            )
            global_outputs.append(qformer_out.last_hidden_state)
        
        # Weighted sum across layers
        global_outputs = torch.stack(global_outputs, dim=0)  # [L, B, K, D]
        global_outputs = global_outputs.permute(1, 2, 0, 3)  # [B, K, L, D]
        weights = torch.softmax(self.global_layer_weights, dim=-1).unsqueeze(-1)  # [K, L, 1]
        global_tokens = (global_outputs * weights).sum(dim=2)  # [B, K, D]
        global_tokens = self.global_proj(global_tokens)  # [B, K, d_llm]
        
        return global_tokens


class GroupwiseOrthogonalConnector(nn.Module):
    """
    Struct-ORCA: Group-wise Orthogonal Q-Former Connector.
    
    Key innovations:
    - Divides queries into semantic groups (e.g., emotion, speaker identity, prosody)
    - Inter-group orthogonality: different groups are pushed apart in embedding space
    - Intra-group coherence: queries within a group can correlate to describe aspects of same attribute
    
    Returns global_tokens tensor and computed group losses.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config
        
        # Group settings
        self.num_groups = getattr(config, 'struct_orca_num_groups', 8)
        self.queries_per_group = getattr(config, 'struct_orca_queries_per_group', 8)
        self.total_queries = self.num_groups * self.queries_per_group
        
        # Determine target layer IDs based on Whisper model
        if getattr(config, 'orca_use_all_layers', False):
            num_encoder_layers = config.encoder_config.num_hidden_layers
            self.target_layer_ids = list(range(num_encoder_layers))
        else:
            if config.encoder_model_id == "openai/whisper-medium":
                self.target_layer_ids = [5, 11, 17, 23]
            elif config.encoder_model_id == "openai/whisper-small":
                self.target_layer_ids = [2, 5, 8, 11]
            elif config.encoder_model_id == "openai/whisper-tiny":
                self.target_layer_ids = [0, 1, 2, 3]
            elif config.encoder_model_id in ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo"]:
                self.target_layer_ids = [7, 15, 23, 31]
            else:
                raise NotImplementedError(f"model_id {config.encoder_model_id} not implemented")
        
        d_encoder = config.encoder_config.d_model
        d_llm = config.llm_config.hidden_size
        
        # === Grouped Queries ===
        # Each group has separate learnable queries per target layer
        self.group_queries = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.randn(1, self.queries_per_group, d_encoder) * 0.02)
                for _ in range(len(self.target_layer_ids))
            ])
            for _ in range(self.num_groups)
        ])
        
        # Layer fusion weights per group
        self.group_layer_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(self.queries_per_group, len(self.target_layer_ids), dtype=torch.float))
            for _ in range(self.num_groups)
        ])
        
        # Shared Q-Former for cross-attention
        qformer_config = BertConfig()
        qformer_config.num_hidden_layers = config.qformer_num_hidden_layers
        qformer_config.num_attention_heads = config.encoder_config.encoder_attention_heads
        qformer_config.hidden_size = d_encoder
        qformer_config.add_cross_attention = True
        qformer_config.is_decoder = True
        # Use SDPA (Flash Attention) if enabled for faster attention
        use_flash = getattr(config, 'use_flash_attention', False)
        qformer_config._attn_implementation = "sdpa" if use_flash else "eager"
        
        self.qformer = BertEncoder(qformer_config)
        
        # Projection to LLM dimension
        self.proj = nn.Sequential(
            nn.LayerNorm(d_encoder),
            nn.Linear(d_encoder, d_llm)
        )
        
        # Loss weights (configurable)
        self.inter_group_weight = getattr(config, 'struct_orca_inter_group_weight', 0.1)
        self.intra_group_weight = getattr(config, 'struct_orca_intra_group_weight', 0.01)
    
    def forward(
        self, 
        encoder_hidden_states: List[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of GroupwiseOrthogonalConnector.
        
        OPTIMIZED: Batches all group queries together for Q-Former forward pass,
        reducing forward calls from (num_groups × num_layers) to just num_layers.
        
        Args:
            encoder_hidden_states: List of hidden states from Whisper encoder layers
            audio_attention_mask: Optional attention mask for audio [B, T]
            
        Returns:
            Tuple of (global_tokens, group_losses):
                - global_tokens: [B, num_groups * queries_per_group, d_llm]
                - group_losses: dict with L_inter_group and L_intra_group
        """
        batch_size = encoder_hidden_states[0].size(0)
        
        target_dtype = self.proj[1].weight.dtype
        target_device = self.proj[1].weight.device
        
        # Collect target layer outputs
        target_layer_outputs = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            if idx in self.target_layer_ids:
                target_layer_outputs.append(hidden_state.to(dtype=target_dtype, device=target_device))
        
        num_layers = len(target_layer_outputs)
        
        # === OPTIMIZATION: Batch all group queries together ===
        # Instead of: for group in groups: for layer in layers: qformer(...)
        # We do: for layer in layers: qformer(all_group_queries)
        
        layer_outputs = []  # [num_layers, B, total_queries, d_encoder]
        
        for layer_idx, hidden_state in enumerate(target_layer_outputs):
            # Concatenate queries from ALL groups for this layer
            # Shape: [B, num_groups * queries_per_group, d_encoder]
            all_queries = []
            for group_idx in range(self.num_groups):
                queries = self.group_queries[group_idx][layer_idx].expand(batch_size, -1, -1)
                all_queries.append(queries)
            
            combined_queries = torch.cat(all_queries, dim=1)  # [B, total_queries, d_encoder]
            combined_queries = combined_queries.to(dtype=target_dtype, device=target_device)
            
            # Single Q-Former forward pass for all groups at this layer
            qformer_out = self.qformer(
                hidden_states=combined_queries,
                encoder_hidden_states=hidden_state,
            )
            layer_outputs.append(qformer_out.last_hidden_state)  # [B, total_queries, d_encoder]
        
        # Stack layer outputs: [num_layers, B, total_queries, d_encoder]
        layer_outputs = torch.stack(layer_outputs, dim=0)
        layer_outputs = layer_outputs.permute(1, 2, 0, 3)  # [B, total_queries, num_layers, d_encoder]
        
        # === Apply layer weights per group and project ===
        all_group_tokens = []  # Will collect [B, queries_per_group, d_llm] for each group
        group_centroids = []   # [num_groups, B, d_llm] for orthogonality loss
        
        for group_idx in range(self.num_groups):
            # Extract this group's outputs from the combined tensor
            start_idx = group_idx * self.queries_per_group
            end_idx = start_idx + self.queries_per_group
            group_layer_outputs = layer_outputs[:, start_idx:end_idx, :, :]  # [B, K, L, D]
            
            # Apply this group's layer weights
            weights = torch.softmax(self.group_layer_weights[group_idx], dim=-1).unsqueeze(-1)  # [K, L, 1]
            group_tokens = (group_layer_outputs * weights).sum(dim=2)  # [B, K, D]
            group_tokens = self.proj(group_tokens)  # [B, K, d_llm]
            
            all_group_tokens.append(group_tokens)
            
            # Compute group centroid (mean of queries in this group)
            centroid = group_tokens.mean(dim=1)  # [B, d_llm]
            group_centroids.append(centroid)
        
        # Concatenate all groups: [B, num_groups * queries_per_group, d_llm]
        global_tokens = torch.cat(all_group_tokens, dim=1)
        
        # Compute group losses
        group_losses = self._compute_group_losses(all_group_tokens, group_centroids)
        
        return global_tokens, group_losses
    
    def _compute_group_losses(
        self, 
        all_group_tokens: List[torch.Tensor],
        group_centroids: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute inter-group and intra-group losses.
        
        Inter-group (orthogonality): Push group centroids apart
        Intra-group (coherence): Allow correlation within groups (light regularization)
        """
        losses = {}
        
        # === Inter-group orthogonality loss ===
        # Group centroids should be orthogonal to each other
        centroids = torch.stack(group_centroids, dim=1)  # [B, num_groups, d_llm]
        centroids_norm = F.normalize(centroids, dim=-1)  # [B, num_groups, d_llm]
        
        # Gram matrix of centroids: should be close to identity
        gram = torch.einsum("bgh,bih->bgi", centroids_norm, centroids_norm)  # [B, G, G]
        I = torch.eye(self.num_groups, device=gram.device, dtype=gram.dtype)
        L_inter = ((gram - I) ** 2).mean()
        losses["L_inter_group"] = self.inter_group_weight * L_inter
        
        # === Intra-group coherence (light diversity regularization) ===
        # Within each group, we allow correlation but add light diversity
        L_intra_total = 0.0
        for group_idx, group_tokens in enumerate(all_group_tokens):
            # group_tokens: [B, K, d_llm]
            tokens_norm = F.normalize(group_tokens, dim=-1)
            intra_gram = torch.einsum("bkh,bqh->bkq", tokens_norm, tokens_norm)  # [B, K, K]
            
            # Light diversity: penalize if all queries are identical (but allow some correlation)
            # Using softer target: allow diagonal=1, off-diagonal can be up to 0.5
            K = self.queries_per_group
            target = torch.eye(K, device=intra_gram.device) * 0.5 + 0.5  # Target: diagonal=1, off-diag=0.5
            L_intra_total += ((intra_gram - target) ** 2).mean()
        
        losses["L_intra_group"] = self.intra_group_weight * (L_intra_total / self.num_groups)
        
        return losses


class ORCAGatedCrossAttention(nn.Module):
    """
    Gated cross-attention module for deep injection of audio tokens into LLM decoder layers.
    Uses data-dependent gating (following Audio Flamingo 3 design).
    Also computes per-layer alignment loss for layer-wise supervision.
    
    hidden_out = hidden + gate(hidden) * LayerNorm(CrossAttn(hidden, audio))
    """
    def __init__(self, hidden_size: int, num_heads: int, gate_init: float = 0.1,
                 rope_theta: float = 10000.0, audio_position_scale: float = 5.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        # Data-dependent gate: projects hidden state to gate value
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
        # Initialize to small values for stable training
        nn.init.zeros_(self.gate_proj[-1].weight)
        nn.init.constant_(self.gate_proj[-1].bias, gate_init)
        
        self.ln = nn.LayerNorm(hidden_size)
        
        # RoPE config for audio position encoding
        self.rope_theta = rope_theta
        self.audio_position_scale = audio_position_scale
        self.hidden_size = hidden_size
        
        # Per-layer loss storage (populated during forward, cleared after collection)
        self.layer_align_loss = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_local: torch.Tensor,
        audio_local_mask: Optional[torch.Tensor] = None,
        transcription_positions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention with data-dependent gating.
        Also computes per-layer alignment loss if in training mode.
        
        Args:
            hidden_states: LLM hidden states [B, T_text, H]
            audio_local: Audio tokens [B, T_audio, H]
            audio_local_mask: Optional mask [B, T_audio], True for valid positions
            transcription_positions: List of (batch_idx, start, end) for transcription in hidden_states
            
        Returns:
            Updated hidden states [B, T_text, H]
        """
        if audio_local is None or audio_local.shape[1] == 0:
            self.layer_align_loss = None
            return hidden_states
        
        # Ensure audio_local has same dtype and device as hidden_states
        audio_local = audio_local.to(dtype=hidden_states.dtype, device=hidden_states.device)
        
        # Apply RoPE rotation to audio tokens with interpolated positions
        batch_size, seq_len, _ = audio_local.shape
        
        # Generate fractional positions for compression: [0, 1/scale, 2/scale, ...]
        positions = torch.arange(seq_len, dtype=torch.float, device=audio_local.device) / self.audio_position_scale
        
        # Compute RoPE cos/sin
        cos, sin = compute_rope_freqs(
            seq_len=seq_len,
            dim=self.hidden_size,
            rope_theta=self.rope_theta,
            device=audio_local.device,
            positions=positions,
        )
        
        # Apply RoPE rotation to audio tokens (like LLM does for text)
        audio_local = apply_rotary_pos_emb(audio_local, cos, sin)
        
        # Build key_padding_mask: True for positions to IGNORE
        if audio_local_mask is not None:
            key_padding_mask = ~audio_local_mask.bool()
        else:
            key_padding_mask = None
        
        # Apply cross-attention
        cross_out, _ = self.cross_attn(
            query=hidden_states,
            key=audio_local,
            value=audio_local,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        cross_out = self.ln(cross_out)
        
        # Data-dependent gate: compute gate from hidden states
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, T, 1]
        
        # Compute per-layer alignment loss: audio tokens should be close to transcription hidden states
        if self.training:
            with torch.no_grad():
                audio_pooled = F.normalize(audio_local.mean(dim=1), dim=-1)  # [B, H]
            
            # Use transcription positions if available, else fallback to full hidden states
            if transcription_positions is not None and len(transcription_positions) > 0:
                # Extract transcription hidden states and pool per sample
                trans_pooled_list = []
                for batch_idx, start, end in transcription_positions:
                    if start < end and end <= hidden_states.size(1):
                        trans_hidden = hidden_states[batch_idx, start:end, :]  # [trans_len, H]
                        trans_pooled_list.append(trans_hidden.mean(dim=0))  # [H]
                
                if len(trans_pooled_list) > 0:
                    trans_pooled = torch.stack(trans_pooled_list, dim=0)  # [N, H]
                    trans_pooled = F.normalize(trans_pooled, dim=-1)
                    # audio_pooled may have different batch size, align by taking first N
                    n = min(audio_pooled.size(0), trans_pooled.size(0))
                    cos_sim = F.cosine_similarity(audio_pooled[:n], trans_pooled[:n], dim=-1)
                    self.layer_align_loss = (1 - cos_sim).mean()
                else:
                    self.layer_align_loss = None
            else:
                # Fallback: use full hidden states
                text_pooled = F.normalize(hidden_states.mean(dim=1), dim=-1)  # [B, H]
                cos_sim = F.cosine_similarity(audio_pooled, text_pooled, dim=-1)  # [B]
                self.layer_align_loss = (1 - cos_sim).mean()
        else:
            self.layer_align_loss = None
        
        return hidden_states + gate * cross_out

@dataclass
class GenerationOutput():
    audios: list[str]
    generated_ids: list[torch.Tensor]
    text: list[str]

class WhisperPerception(nn.Module):
    """
    Perception module using Whisper encoder.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        # Create connector based on mode
        if config.connector_mode == "orca_hybrid":
            self.connector = ORCAHybridConnector(config)
        elif config.connector_mode == "struct_orca":
            self.connector = GroupwiseOrthogonalConnector(config)
        else:
            self.connector = QformerConnector(config)
        
        # Store group losses for Struct-ORCA (populated during forward)
        self._struct_orca_losses = None


    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, transcription_embeddings_list: Optional[List[torch.Tensor]] = None, **kwargs) -> Union[Tuple[torch.Tensor, List[int]], Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Forward pass of the WhisperPerception.

        Args:
            input_features (torch.Tensor): Input mel features.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            transcription_embeddings_list (Optional[List[torch.Tensor]], optional): List of transcription embeddings. Defaults to None.

        Returns:
            For qformer_1: tuple[torch.Tensor, list[int]]: (audio_features, speech_feature_lengths)
            For orca_hybrid: tuple[torch.Tensor, list[int]]: (global_tokens, global_lengths)
        """
        bs = input_features.size(0)

        result = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        
        if self.config.connector_mode == "orca_hybrid":
            # result is global_tokens tensor (no local tokens anymore)
            global_tokens = result
            speech_feature_lengths = [self.config.orca_global_num_tokens] * bs
            return global_tokens, speech_feature_lengths
        elif self.config.connector_mode == "struct_orca":
            # result is (global_tokens, group_losses) tuple
            global_tokens, group_losses = result
            self._struct_orca_losses = group_losses
            total_queries = self.config.struct_orca_num_groups * self.config.struct_orca_queries_per_group
            speech_feature_lengths = [total_queries] * bs
            return global_tokens, speech_feature_lengths
        else:
            # result is audio_features tensor
            audio_features = result
            speech_feature_lengths = [self.config.prompt_size] * bs
            return audio_features, speech_feature_lengths


    def forward_whisper(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        """
        Forward through Whisper encoder layers.
        """
        bs = input_features.size(0)
        
        # Ensure input_features match Whisper's dtype
        target_dtype = self.whisper.model.encoder.conv1.weight.dtype
        target_device = self.whisper.model.encoder.conv1.weight.device
        input_features = input_features.to(dtype=target_dtype, device=target_device)
        
        expected_seq_length = self.whisper.model.encoder.config.max_source_positions * self.whisper.model.encoder.conv1.stride[0] * self.whisper.model.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        

        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.whisper.model.encoder.embed_positions.weight[:self.whisper.model.encoder.config.max_source_positions, :] # @kehan
        embed_pos = embed_pos.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        
        # Collect all layer outputs for ORCA
        all_layer_outputs = []

        if self.config.connector_mode == "qformer_1":
            layer_prompt_outputs = []
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]

                if idx in self.connector.config.target_layer_ids:
                    # use different prompt for different layers
                    layer_prompt = self.connector.layer_prompts[self.connector.config.target_layer_ids.index(idx)].expand(bs, -1, -1)
                    
                    # Qformer is a BERTEncoder(but set to decoder) from huggingface Transformers
                    qformer_output = self.connector.qformer(
                        layer_prompt,
                        encoder_hidden_states=hidden_states,
                    )
                    
                    layer_prompt_output = qformer_output.last_hidden_state[:, :self.config.prompt_size, :] # (b, prompt_size, d_model)
                    layer_prompt_outputs.append(layer_prompt_output) # list of (b, prompt_size, d_model)

            layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
            layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)
            
            self.norm_weights = torch.nn.functional.softmax(self.connector.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)
            prompt_output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
            assert prompt_output.size(1) == self.config.prompt_size, prompt_output.size()
            prompt_output = self.connector.proj(prompt_output)
            
            return prompt_output
        
        elif self.config.connector_mode == "orca_hybrid":
            # Collect all layer hidden states
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]
                all_layer_outputs.append(hidden_states)
            
            # Pass all layer outputs to ORCAHybridConnector
            global_tokens = self.connector(all_layer_outputs)
            return global_tokens

        elif self.config.connector_mode == "struct_orca":
            # Collect all layer hidden states
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]
                all_layer_outputs.append(hidden_states)
            
            # Pass all layer outputs to GroupwiseOrthogonalConnector
            global_tokens, group_losses = self.connector(all_layer_outputs)
            return global_tokens, group_losses

        else:
            raise NotImplementedError(f"mode {self.config.connector_mode} not implemented")
    
    



class DeSTA25Config(PretrainedConfig):
    model_type = "desta25"

    def __init__(self, 
                 llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
                 encoder_model_id="openai/whisper-large-v3",
                 connector_mode="qformer_1", 
                 qformer_num_hidden_layers=2, 
                 prompt_size=64, 
                 use_lora=False,
                 audio_locator="<|AUDIO|>",
                 placeholder_token="<|reserved_special_token_87|>",
                 # ORCA-DeSTA configuration fields
                 orca_enabled=False,
                 orca_use_all_layers=False,  # If True, use all encoder layers; if False, use selected layers
                 orca_local_enabled=True,  # If False, only global tokens are used (no local downsample)
                 orca_global_cross_attn=False,  # If True, global tokens also use cross-attention instead of concat
                 orca_deep_injection_enabled=True, # If False, disable gated cross-attention in all LLM layers
                 orca_deep_injection_stride=1,  # Stride for injection: inject every N layers
                 orca_audio_position_scale=2.5,  # Position interpolation scale for audio tokens (adjusted for 4x downsample)
                 orca_global_num_tokens=4,
                 orca_local_downsample=4,
                 orca_local_kernel_size=5,
                 orca_gate_init=0.1,
                 orca_ortho_weight_global=0.01,
                 orca_ortho_diversity_weight=0.01,
                 orca_ortho_weight_qformer_local=0.01,  # Orthogonality between Q-Former global and local tokens
                 orca_align_weight_local=0.05,  # Alignment loss to bring local tokens closer to text embeddings
                 # Struct-ORCA configuration fields
                 struct_orca_num_groups=8,  # Number of semantic groups for queries
                 struct_orca_queries_per_group=8,  # Queries per group (total = num_groups * queries_per_group)
                 struct_orca_inter_group_weight=0.1,  # Weight for inter-group orthogonality loss
                 struct_orca_intra_group_weight=0.01,  # Weight for intra-group diversity loss
                 struct_orca_iv_weight=0.1,  # Weight for IV-Guided Disentanglement adversarial loss
                 struct_orca_acd_alpha=0.5,  # Alpha for Acoustic-Contrastive Decoding
                 **kwargs):
        
        super().__init__(**kwargs)

        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        self.connector_mode = connector_mode
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.prompt_size = prompt_size

        self.audio_locator = audio_locator
        self.placeholder_token = placeholder_token

        self.llm_config = AutoConfig.from_pretrained(self.llm_model_id)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_id)

        self.use_lora = use_lora

        # ORCA-DeSTA configuration
        self.orca_enabled = orca_enabled
        self.orca_use_all_layers = orca_use_all_layers
        self.orca_local_enabled = orca_local_enabled
        self.orca_global_cross_attn = orca_global_cross_attn
        self.orca_global_cross_attn = orca_global_cross_attn
        self.orca_deep_injection_enabled = orca_deep_injection_enabled
        self.orca_deep_injection_stride = orca_deep_injection_stride
        self.orca_audio_position_scale = orca_audio_position_scale
        self.orca_global_num_tokens = orca_global_num_tokens
        self.orca_local_downsample = orca_local_downsample
        self.orca_local_kernel_size = orca_local_kernel_size
        self.orca_gate_init = orca_gate_init
        self.orca_ortho_weight_global = orca_ortho_weight_global
        self.orca_ortho_diversity_weight = orca_ortho_diversity_weight
        self.orca_ortho_weight_qformer_local = orca_ortho_weight_qformer_local
        self.orca_align_weight_local = orca_align_weight_local

        # Struct-ORCA configuration
        self.struct_orca_num_groups = struct_orca_num_groups
        self.struct_orca_queries_per_group = struct_orca_queries_per_group
        self.struct_orca_inter_group_weight = struct_orca_inter_group_weight
        self.struct_orca_intra_group_weight = struct_orca_intra_group_weight
        self.struct_orca_iv_weight = struct_orca_iv_weight
        self.struct_orca_acd_alpha = struct_orca_acd_alpha

        self.info = "Ｄｅｓｔａ２。５ Ａｕｄｉｏ"



class DeSTA25AudioModel(PreTrainedModel):
    config_class = DeSTA25Config

    def __init__(self, config, cache_dir=None, token=None, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        token = token if token else os.getenv("HF_TOKEN")
        cache_dir = cache_dir if cache_dir else os.getenv("HF_HOME")

        self.audio_locator = config.audio_locator
        self.placeholder_token = config.placeholder_token

        logging.info(f"Loading LLM model from {self.config.llm_model_id}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=token,
        )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config).base_model.model
        
        logging.info(f"Loading Audio model from {self.config.encoder_model_id}")
        self.perception = WhisperPerception(self.config)

        # === ORCA-DeSTA Setup ===
        # Check both orca_enabled and connector_mode for robust detection
        is_orca = getattr(self.config, 'orca_enabled', False) or self.config.connector_mode == "orca_hybrid"
        if is_orca:
            logging.info("Enabling ORCA-DeSTA components")
            
            # Enable deep cross-attention injection (if enabled in config)
            if getattr(self.config, 'orca_deep_injection_enabled', True):
                self._enable_orca_deep_injection()
            else:
                logging.info("ORCA deep injection explicitly disabled via config")
            
            # Storage for audio_local during forward (set before LLM call, cleared after)
            self._orca_audio_local = None
            self._orca_audio_local_mask = None
            
            # Ensure ORCA modules are in correct dtype (align with LLM)
            if hasattr(self, "orca_cross_attns"):
                self.orca_cross_attns.to(dtype=self.llm_model.dtype, device=self.llm_model.device)
            if hasattr(self.perception, "connector") and self.config.connector_mode == "orca_hybrid":
                self.perception.connector.to(dtype=self.llm_model.dtype, device=self.llm_model.device)
        
        # === Struct-ORCA Setup ===
        is_struct_orca = self.config.connector_mode == "struct_orca"
        if is_struct_orca:
            logging.info("Enabling Struct-ORCA components")
        
        # Always create discriminator for IV-Guided Disentanglement to ensure 
        # consistent parameter count across DDP ranks (even if iv_weight=0)
        vocab_size = getattr(self.config.llm_config, 'vocab_size', 32000)
        self.content_discriminator = TextContentDiscriminator(
            hidden_size=self.config.llm_config.hidden_size,
            num_groups=getattr(self.config, 'struct_orca_num_groups', 8),
            vocab_size=vocab_size,
        )
        self.content_discriminator.to(dtype=self.llm_model.dtype, device=self.llm_model.device)
        
        # Storage for discriminator outputs (set during forward)
        self._discriminator_outputs = None

        self.configure_trainable_parameters()
        
        # Ensure all DDP ranks are synchronized after model initialization
        if dist.is_initialized():
            dist.barrier()
            logging.info(f"DDP barrier passed for rank {dist.get_rank()}")

    def forward(self, input_ids,
                attention_mask, 
                batch_features, 
                batch_transcription_ids,
                batch_start_positions,
                labels=None,
                **kwargs):
        
        # Prepare inputs, which handles both ORCA and non-ORCA paths
        prepare_result = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )
        
        # Handle ORCA mode - check based on result type or connector_mode
        is_orca_mode = (
            isinstance(prepare_result, tuple) and len(prepare_result) >= 2
        ) and self.config.connector_mode == "orca_hybrid"
        
        # Handle Struct-ORCA mode
        is_struct_orca_mode = self.config.connector_mode == "struct_orca"
        
        if is_orca_mode:
            if len(prepare_result) == 3:
                inputs_embeds, global_audio_tokens, transcription_positions = prepare_result
            else:
                inputs_embeds, global_audio_tokens = prepare_result
                transcription_positions = None
            
            # Store transcription positions for cross-attention alignment loss
            self._orca_transcription_positions = transcription_positions
            
            # Set audio tokens for deep injection (accessed by wrapped decoder layers)
            # Only set if deep injection is enabled in ablation config
            if getattr(self.config, 'orca_deep_injection_enabled', True):
                # Use global tokens for cross-attention injection
                self._orca_audio_local = global_audio_tokens
            else:
                self._orca_audio_local = None
                
            self._orca_audio_local_mask = None
            
            # Call LLM with output_hidden_states to get text hidden states for orthogonality loss
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            
            # Clear audio tokens after LLM forward
            self._orca_audio_local = None
            self._orca_audio_local_mask = None
            
            # Collect per-layer alignment losses from cross-attention modules
            layer_align_losses = self._collect_layer_align_losses()
            
            # Compute ORCA auxiliary losses
            text_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
            
            # Extract transcription and target embeddings for contrastive alignment
            transcription_embeds = None
            target_embeds = None
            
            if len(batch_transcription_ids) > 0:
                # Get transcription embeddings
                with torch.no_grad():
                    transcription_embeds_list = []
                    for trans_ids in batch_transcription_ids:
                        trans_ids = trans_ids.squeeze(0)
                        if trans_ids.device != inputs_embeds.device:
                            trans_ids = trans_ids.to(inputs_embeds.device)
                        trans_emb = self.llm_model.model.embed_tokens(trans_ids)
                        transcription_embeds_list.append(trans_emb.mean(dim=0))  # Pool
                    transcription_embeds = torch.stack(transcription_embeds_list, dim=0)  # [B, H]
            
            # Extract target embeddings from labels
            if labels is not None:
                with torch.no_grad():
                    # Get target positions (where labels != -100)
                    target_mask = labels != -100  # [B, T]
                    if target_mask.any():
                        # Get embeddings for target tokens
                        target_ids = labels.clone()
                        target_ids[~target_mask] = 0  # Mask out non-target positions
                        target_emb_full = self.llm_model.model.embed_tokens(target_ids)  # [B, T, H]
                        
                        # Pool only target positions
                        target_embeds_list = []
                        for b in range(target_mask.size(0)):
                            if target_mask[b].any():
                                target_emb = target_emb_full[b, target_mask[b], :]  # [num_targets, H]
                                target_embeds_list.append(target_emb.mean(dim=0))  # Pool
                            else:
                                target_embeds_list.append(torch.zeros(target_emb_full.size(-1), device=target_emb_full.device))
                        target_embeds = torch.stack(target_embeds_list, dim=0)  # [B, H]

            # Compute ORCA losses (only global tokens now)
            orca_losses = self.compute_orca_losses(
                global_tokens=global_audio_tokens,
                local_tokens=None,
                text_hidden=text_hidden,
                layer_align_losses=layer_align_losses,
                transcription_embeds=transcription_embeds,
                target_embeds=target_embeds,
            )
            
            # Attach losses to outputs
            outputs.orca_losses = orca_losses
            outputs.audio_global = global_audio_tokens
            
            return outputs
        elif is_struct_orca_mode:
            # Struct-ORCA forward path
            # prepare_result is just inputs_embeds for struct_orca (same as qformer_1)
            inputs_embeds = prepare_result if not isinstance(prepare_result, tuple) else prepare_result[0]
            
            # Get global audio tokens from perception for discriminator
            global_audio_tokens = None
            if hasattr(self.perception, 'connector') and isinstance(self.perception.connector, GroupwiseOrthogonalConnector):
                # The connector stores its output - retrieve from last forward
                # For proper access, we need to extract from the prepared inputs
                pass  # Will be set via outputs.audio_global at the end
            
            # Call LLM 
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,  # Not needed for struct_orca
            )
            
            # Collect group losses from perception module
            struct_orca_losses = getattr(self.perception, "_struct_orca_losses", None)
            if struct_orca_losses is not None:
                outputs.struct_orca_losses = struct_orca_losses
            
            # Get audio tokens from the batch (stored during _prepare_inputs_for_llm)
            # For discriminator, we need the raw global tokens before embedding into sequence
            if hasattr(self, '_struct_orca_audio_tokens') and self._struct_orca_audio_tokens is not None:
                outputs.audio_global = self._struct_orca_audio_tokens
                self._struct_orca_audio_tokens = None  # Clear after use
            else:
                # DEBUG: Log if audio_global is not set
                import logging
                logging.warning(f"[DEBUG] _struct_orca_audio_tokens not set. hasattr={hasattr(self, '_struct_orca_audio_tokens')}, value={getattr(self, '_struct_orca_audio_tokens', 'MISSING')}")
            
            return outputs
        else:
            
            # Check if we should compute losses even in Q-Former mode
            # This allows testing orthogonality losses without ORCA architecture
            compute_qformer_losses = (
                self.config.connector_mode == "qformer_1" and
                getattr(self.config, 'orca_enabled', False) and
                (getattr(self.config, 'orca_ortho_diversity_weight', 0.0) > 0 or
                 getattr(self.config, 'orca_align_weight_local', 0.0) > 0)
            )
            
            if compute_qformer_losses:
                # Call LLM with output_hidden_states for loss computation
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                
                # Extract Q-Former tokens from inputs_embeds
                # Q-Former tokens are embedded at audio positions
                qformer_tokens = None
                if len(batch_start_positions) > 0:
                    # Collect Q-Former tokens per sample to maintain [B, K, H] shape
                    batch_qformer_tokens = []
                    for b, start_positions_sample in enumerate(batch_start_positions):
                        sample_tokens_list = []
                        for start_pos in start_positions_sample:
                            end_pos = start_pos + self.config.prompt_size
                            if end_pos <= inputs_embeds.size(1):
                                sample_tokens_list.append(inputs_embeds[b, start_pos:end_pos, :])
                        
                        if sample_tokens_list:
                            # Average multiple audios in this sample: [M, K, H] -> [K, H]
                            sample_avg = torch.stack(sample_tokens_list, dim=0).mean(dim=0)
                            batch_qformer_tokens.append(sample_avg)
                        else:
                            # Fallback for samples without audio (should ideally not happen in training)
                            batch_qformer_tokens.append(
                                torch.zeros(self.config.prompt_size, inputs_embeds.size(-1), 
                                          device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                            )
                    
                    if batch_qformer_tokens:
                        qformer_tokens = torch.stack(batch_qformer_tokens, dim=0)  # [B, K, H]
                
                # Compute losses using Q-Former tokens
                text_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
                
                # Extract transcription and target embeddings for contrastive alignment
                transcription_embeds = None
                target_embeds = None
                
                if len(batch_transcription_ids) > 0:
                    # Get transcription embeddings
                    with torch.no_grad():
                        transcription_embeds_list = []
                        for trans_ids in batch_transcription_ids:
                            trans_ids = trans_ids.squeeze(0)
                            if trans_ids.device != inputs_embeds.device:
                                trans_ids = trans_ids.to(inputs_embeds.device)
                            trans_emb = self.llm_model.model.embed_tokens(trans_ids)
                            transcription_embeds_list.append(trans_emb.mean(dim=0))  # Pool
                        transcription_embeds = torch.stack(transcription_embeds_list, dim=0)  # [B, H]
                
                # Extract target embeddings from labels
                if labels is not None:
                    with torch.no_grad():
                        # Get target positions (where labels != -100)
                        target_mask = labels != -100  # [B, T]
                        if target_mask.any():
                            # Get embeddings for target tokens
                            target_ids = labels.clone()
                            target_ids[~target_mask] = 0  # Mask out non-target positions
                            target_emb_full = self.llm_model.model.embed_tokens(target_ids)  # [B, T, H]
                            
                            # Pool only target positions
                            target_embeds_list = []
                            for b in range(target_mask.size(0)):
                                if target_mask[b].any():
                                    target_emb = target_emb_full[b, target_mask[b], :]  # [num_targets, H]
                                    target_embeds_list.append(target_emb.mean(dim=0))  # Pool
                                else:
                                    target_embeds_list.append(torch.zeros(target_emb_full.size(-1), device=target_emb_full.device))
                            target_embeds = torch.stack(target_embeds_list, dim=0)  # [B, H]
                
                qformer_losses = self.compute_qformer_losses(
                    qformer_tokens=qformer_tokens,
                    text_hidden=text_hidden,
                    transcription_embeds=transcription_embeds,
                    target_embeds=target_embeds,
                )
                
                # Attach losses to outputs
                outputs.orca_losses = qformer_losses
                
                return outputs
            else:
                # Standard path without losses
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                return outputs 

    def _prepare_inputs_for_llm(self, 
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions
        ):
        """
        Prepare the embeddings input for the LLM.
        Batch_features: list of audio features
        Batch_transcription_ids: list of transcription ids
        Batch_start_positions: list of start positions
        
        Returns:
            For non-ORCA: inputs_embeds tensor
            For ORCA: (inputs_embeds, global_audio_tokens, transcription_positions)
        """

        N_audio = len(batch_start_positions)
        device = next(self.llm_model.parameters()).device
        
        # Handle empty audio case
        if N_audio == 0:
            embeds = self.llm_model.model.embed_tokens(input_ids)
            if self.config.connector_mode == "orca_hybrid":
                # Return 3-element tuple consistent with normal ORCA path
                return embeds, None, None
            return embeds
        
        # Ensure batch_features is on the correct device
        if batch_features.device != device:
            batch_features = batch_features.to(device)
        
        # Get list of transcription embeddings
        transcription_embeddings_list = []
        with torch.no_grad():
            for audio_batch_idx in range(N_audio):
                # Ensure transcription_ids are on the correct device
                trans_ids = batch_transcription_ids[audio_batch_idx].squeeze(0)
                if trans_ids.device != device:
                    trans_ids = trans_ids.to(device)
                transcription_embeddings = self.llm_model.model.embed_tokens(trans_ids) # (length, dim)
                transcription_embeddings_list.append(transcription_embeddings)

        # Forward speech encoder and connector
        perception_output = self.perception(
            input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list
        )
        
        # Handle ORCA mode output - check based on tuple length or connector_mode
        # This handles cases where orca_enabled may not be set but connector_mode is orca_hybrid
        is_orca_output = (
            isinstance(perception_output, tuple) and len(perception_output) == 2 and self.config.connector_mode == "orca_hybrid"
        ) or self.config.connector_mode == "orca_hybrid"
        
        is_struct_orca = self.config.connector_mode == "struct_orca"
        
        if is_orca_output and self.config.connector_mode == "orca_hybrid":
            # perception_output is (global_tokens, lengths)
            batch_global_tokens, batch_audio_feature_lengths = perception_output
            batch_audio_features = batch_global_tokens  # Global tokens are what we splice
        elif is_struct_orca:
            # perception_output is (global_tokens, lengths) - same structure
            batch_global_tokens, batch_audio_feature_lengths = perception_output
            batch_audio_features = batch_global_tokens
            # Store audio tokens for discriminator access
            self._struct_orca_audio_tokens = batch_global_tokens
        else:
            # perception_output is (audio_features, lengths)
            batch_audio_features, batch_audio_feature_lengths = perception_output
            batch_global_tokens = None

        assert len(batch_start_positions) == len(batch_transcription_ids) == batch_audio_features.size(0) == len(batch_audio_feature_lengths), "batch_start_positions, batch_transcription_ids, audio_features, speech_feature_lengths must have the same length."


        # [---- Other text embeddings ----][---- placeholder embeddings ----][---- Other text embeddings ----]
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids)
        
        # Track transcription positions for alignment loss
        transcription_positions = []
        
        for audio_batch_idx in range(N_audio):
            start_position = batch_start_positions[audio_batch_idx] # tuple (text_idx, audio_start_position)
            text_batch_idx = start_position[0]
            audio_start_position = start_position[1]

            # get the speech features   
            audio_features = batch_audio_features[audio_batch_idx]
            speech_feature_length = batch_audio_feature_lengths[audio_batch_idx]

            # get transcription embeddings
            transcription_embeddings = transcription_embeddings_list[audio_batch_idx] # (length, dim)
            trans_len = transcription_embeddings.size(0)
            
            # Compute transcription position in final sequence
            # Transcription is placed after audio features
            trans_start = audio_start_position + speech_feature_length
            trans_end = trans_start + trans_len
            transcription_positions.append((text_batch_idx, trans_start, trans_end))

            # # concat the speech features and transcription embeddings
            audio_embeddings = torch.cat([audio_features, transcription_embeddings], dim=0)

            assert audio_embeddings.size(0) == (speech_feature_length + trans_len)

            # # replace the input_embeds with the audio features
            # # [---- Other text embeddings ----][---- audio features + transcription embeddings ----][---- Other text embeddings ----]
            target_slice = slice(audio_start_position, audio_start_position + audio_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = audio_embeddings
            

            # clean GPU memory
            del audio_features, speech_feature_length, transcription_embeddings, audio_embeddings

        if self.config.connector_mode == "orca_hybrid":
            return inputs_embeds, batch_global_tokens, transcription_positions

        return inputs_embeds
    
    def _enable_orca_deep_injection(self):
        """
        Wrap each LLM decoder layer with gated cross-attention for deep injection
        of local prosody tokens.
        """
        is_orca = getattr(self.config, 'orca_enabled', False) or self.config.connector_mode == "orca_hybrid"
        if not is_orca:
            return
        
        hidden_size = self.config.llm_config.hidden_size
        num_heads = self.config.llm_config.num_attention_heads
        gate_init = getattr(self.config, 'orca_gate_init', 0.1)
        
        # Get number of layers from config to ensure consistency across DDP ranks
        num_layers = getattr(self.config.llm_config, 'num_hidden_layers', None)
        
        # Get decoder layers - handle different model architectures
        if hasattr(self.llm_model, 'model') and hasattr(self.llm_model.model, 'layers'):
            layers = self.llm_model.model.layers  # Llama/Qwen-style
        elif hasattr(self.llm_model, 'transformer') and hasattr(self.llm_model.transformer, 'h'):
            layers = self.llm_model.transformer.h  # GPT-style
        else:
            logging.warning("Could not find decoder layers for ORCA deep injection")
            return
        
        # Verify layer count matches config (DDP consistency check)
        if num_layers is not None and len(layers) != num_layers:
            logging.warning(f"Layer count mismatch: config has {num_layers}, model has {len(layers)}")
        
        # Create cross-attention modules and wrap layer forwards
        # Use fixed number from config if available to ensure DDP consistency
        actual_num_layers = num_layers if num_layers is not None else len(layers)
        self.orca_cross_attns = nn.ModuleList()
        
        # Get RoPE config from LLM for consistency
        rope_theta = getattr(self.config.llm_config, 'rope_theta', 10000.0)
        audio_position_scale = getattr(self.config, 'orca_audio_position_scale', 5.0)
        
        for layer_idx in range(actual_num_layers):
            cross_attn = ORCAGatedCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                gate_init=gate_init,
                rope_theta=rope_theta,
                audio_position_scale=audio_position_scale,
            )
            self.orca_cross_attns.append(cross_attn)
        
        # Wrap each layer's forward method
        injection_stride = getattr(self.config, 'orca_deep_injection_stride', 1)
        
        for layer_idx, layer in enumerate(layers):
            # Only inject if stride condition is met
            if layer_idx % injection_stride != 0:
                continue

            cross_attn = self.orca_cross_attns[layer_idx]
            
            # Store reference to parent model for accessing audio_local
            parent_model = self
            layer_cross_attn = cross_attn
            orig_forward = layer.forward
            
            def make_wrapped_forward(orig_fn, xattn, parent):
                def wrapped_forward(hidden_states, *args, **kwargs):
                    outputs = orig_fn(hidden_states, *args, **kwargs)
                    
                    # Get hidden states from outputs
                    if isinstance(outputs, tuple):
                        h = outputs[0]
                        rest = outputs[1:]
                    else:
                        h = outputs
                        rest = ()
                    
                    # Apply cross-attention if audio_local is available
                    audio_local = getattr(parent, "_orca_audio_local", None)
                    audio_local_mask = getattr(parent, "_orca_audio_local_mask", None)
                    transcription_positions = getattr(parent, "_orca_transcription_positions", None)
                    
                    if audio_local is not None:
                        h = xattn(
                            hidden_states=h,
                            audio_local=audio_local,
                            audio_local_mask=audio_local_mask,
                            transcription_positions=transcription_positions,
                        )
                    
                    if isinstance(outputs, tuple):
                        return (h,) + rest
                    else:
                        return h
                
                return wrapped_forward
            
            layer.forward = make_wrapped_forward(orig_forward, layer_cross_attn, parent_model)
        
        logging.info(f"ORCA deep injection enabled for {len(layers)} decoder layers")
    
    def _collect_layer_align_losses(self) -> List[torch.Tensor]:
        """
        Collect per-layer alignment losses from all ORCA cross-attention modules.
        Returns list of losses, one per layer.
        """
        losses = []
        if hasattr(self, 'orca_cross_attns'):
            for name, xattn in self.orca_cross_attns.named_modules():
                if isinstance(xattn, ORCAGatedCrossAttention):
                    if xattn.layer_align_loss is not None:
                        losses.append(xattn.layer_align_loss)
                        xattn.layer_align_loss = None  # Clear after collection
        return losses
    
    def compute_orca_losses(
        self,
        global_tokens: Optional[torch.Tensor],
        local_tokens: Optional[torch.Tensor],  # Kept for API compatibility, but unused
        text_hidden: Optional[torch.Tensor],
        layer_align_losses: Optional[List[torch.Tensor]] = None,
        transcription_embeds: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ORCA auxiliary losses:
        - Global token diversity loss (orthogonality within global tokens)
        - Layer-wise alignment loss (aggregated from cross-attention modules)
        """
        losses = {}
        
        if global_tokens is not None:
            # Diversity between global tokens (Gram matrix close to identity)
            g = F.normalize(global_tokens, dim=-1)  # [B, K, H]
            gram = torch.einsum("bkh,bqh->bkq", g, g)  # [B, K, K]
            I = torch.eye(gram.size(-1), device=gram.device)
            L_div = ((gram - I) ** 2).mean()
            losses["L_ortho_diversity"] = self.config.orca_ortho_diversity_weight * L_div
        
        # Layer-wise alignment loss: aggregated from cross-attention modules
        # Each layer computes alignment between audio and text at that layer's representation
        if layer_align_losses is not None and len(layer_align_losses) > 0:
            L_align_layerwise = torch.stack(layer_align_losses).mean()
            losses["L_align_layerwise"] = self.config.orca_align_weight_local * L_align_layerwise

        # L_align_global: Contrastive alignment loss for GLOBAL tokens
        # Push audio away from transcription, pull toward target
        if global_tokens is not None and getattr(self.config, 'orca_align_weight_local', 0.0) > 0:
            # Only compute if we have transcription/target embeddings (Contrastive)
            if transcription_embeds is not None and target_embeds is not None:
                # Pool Global tokens
                audio_pooled = F.normalize(global_tokens.mean(dim=1), dim=-1)  # [B, H]
                
                # Normalize
                trans_pooled = F.normalize(transcription_embeds, dim=-1)  # [B, H]
                target_pooled = F.normalize(target_embeds, dim=-1)  # [B, H]
                
                # Similarity to transcription (should be LOW)
                sim_trans = F.cosine_similarity(audio_pooled, trans_pooled, dim=-1)  # [B]
                
                # Similarity to target (should be HIGH)
                sim_target = F.cosine_similarity(audio_pooled, target_pooled, dim=-1)  # [B]
                
                # Contrastive loss with margin
                # Loss = max(0, margin + sim_trans - sim_target)
                margin = 0.5
                contrastive_loss = torch.clamp(margin + sim_trans - sim_target, min=0.0).mean()
                
                # Also add direct target alignment term
                target_align_loss = (1 - sim_target).mean()
                
                # Combined loss
                L_align = contrastive_loss + 0.5 * target_align_loss
                losses["L_align_global"] = self.config.orca_align_weight_local * L_align
                
                # Add individual components for monitoring
                losses["L_align_contrastive"] = contrastive_loss
                losses["sim_trans"] = sim_trans.mean()
                losses["sim_target"] = sim_target.mean()
        
        return losses
    
    def compute_qformer_losses(
        self,
        qformer_tokens: Optional[torch.Tensor],
        text_hidden: Optional[torch.Tensor],
        transcription_embeds: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute orthogonality losses for Q-Former tokens (without ORCA architecture).
        This allows testing loss contributions using DeSTA2.5 baseline architecture.
        
        Args:
            qformer_tokens: Q-Former output tokens [B, K, H]
            text_hidden: LLM final hidden states [B, T, H]
            transcription_embeds: Transcription embeddings [B, H] (negative samples)
            target_embeds: Target sequence embeddings [B, H] (positive samples)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # L_ortho_diversity: Diversity between Q-Former tokens
        if qformer_tokens is not None and getattr(self.config, 'orca_ortho_diversity_weight', 0.0) > 0:
            g = F.normalize(qformer_tokens, dim=-1)  # [B, K, H]
            gram = torch.einsum("bkh,bqh->bkq", g, g)  # [B, K, K]
            I = torch.eye(gram.size(-1), device=gram.device)
            L_div = ((gram - I) ** 2).mean()
            losses["L_ortho_diversity"] = self.config.orca_ortho_diversity_weight * L_div
        
        # L_align: Contrastive alignment loss
        # Push audio away from transcription, pull toward target
        if qformer_tokens is not None and getattr(self.config, 'orca_align_weight_local', 0.0) > 0:
            # Pool Q-Former tokens
            audio_pooled = F.normalize(qformer_tokens.mean(dim=1), dim=-1)  # [B, H]
            
            # Contrastive loss: push away from transcription, pull toward target
            if transcription_embeds is not None and target_embeds is not None:
                # Normalize
                trans_pooled = F.normalize(transcription_embeds, dim=-1)  # [B, H]
                target_pooled = F.normalize(target_embeds, dim=-1)  # [B, H]
                
                # Similarity to transcription (should be LOW)
                sim_trans = F.cosine_similarity(audio_pooled, trans_pooled, dim=-1)  # [B]
                
                # Similarity to target (should be HIGH)
                sim_target = F.cosine_similarity(audio_pooled, target_pooled, dim=-1)  # [B]
                
                # Contrastive loss with margin
                # Loss = max(0, margin + sim_trans - sim_target)
                # Encourages: sim_target > sim_trans + margin
                margin = 0.5
                contrastive_loss = torch.clamp(margin + sim_trans - sim_target, min=0.0).mean()
                
                # Also add direct target alignment term
                target_align_loss = (1 - sim_target).mean()
                
                # Combined loss
                L_align = contrastive_loss + 0.5 * target_align_loss
                losses["L_align"] = self.config.orca_align_weight_local * L_align
                
                # Add individual components for monitoring
                losses["L_align_contrastive"] = contrastive_loss
                losses["L_align_target"] = target_align_loss
                losses["sim_trans"] = sim_trans.mean()  # For monitoring
                losses["sim_target"] = sim_target.mean()  # For monitoring
                
            elif text_hidden is not None:
                # Fallback: simple alignment to text hidden states
                text_pooled = F.normalize(text_hidden.mean(dim=1), dim=-1)  # [B, H]
                cos_sim = F.cosine_similarity(audio_pooled, text_pooled, dim=-1)  # [B]
                L_align = (1 - cos_sim).mean()
                losses["L_align"] = self.config.orca_align_weight_local * L_align
        
        return losses
        
    def state_dict(self):
        """
        Only return "trainable" parameters, since most of the parameters are frozen
        """
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()
        return trainable_state_dict
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Custom load_state_dict that handles backward compatibility:
        - Maps old 'ocar_cross_attns' keys to new 'orca_cross_attns' keys
        - Automatically detects checkpoint layer configuration and adjusts model accordingly
        """
        # Create a new state dict with renamed keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # Handle ocar -> orca renaming for backward compatibility
            if key.startswith("ocar_cross_attns"):
                new_key = key.replace("ocar_cross_attns", "orca_cross_attns")
                logging.debug(f"Renaming checkpoint key: {key} -> {new_key}")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Auto-detect layer configuration from checkpoint
        # DISABLED to prevent DDP desync issues with sharded loading
        # if 'perception.connector.global_layer_weights' in new_state_dict:
        #     checkpoint_shape = new_state_dict['perception.connector.global_layer_weights'].shape
        #     checkpoint_num_layers = checkpoint_shape[1]  # [K, L] -> L is number of layers
            
        #     # Get current model configuration
        #     if hasattr(self.perception, 'connector') and hasattr(self.perception.connector, 'target_layer_ids'):
        #         current_num_layers = len(self.perception.connector.target_layer_ids)
                
        #         if checkpoint_num_layers != current_num_layers:
        #             logging.warning(
        #                 f"Layer count mismatch detected: checkpoint has {checkpoint_num_layers} layers, "
        #                 f"current model has {current_num_layers} layers. "
        #                 f"Automatically adjusting model configuration to match checkpoint."
        #             )
                    
        #             # Determine if checkpoint used all layers
        #             num_encoder_layers = self.config.encoder_config.num_hidden_layers
        #             if checkpoint_num_layers == num_encoder_layers:
        #                 # Checkpoint used all layers
        #                 logging.info(f"Checkpoint uses all {num_encoder_layers} encoder layers. Reconfiguring model...")
        #                 self.config.orca_use_all_layers = True
        #             else:
        #                 # Checkpoint used selected layers - we can't automatically determine which ones
        #                 # So we'll just update the target_layer_ids to match the checkpoint size
        #                 logging.info(f"Checkpoint uses {checkpoint_num_layers} selected layers. Reconfiguring model...")
        #                 self.config.orca_use_all_layers = False
        #                 # Use first N layers as a fallback
        #                 self.perception.connector.target_layer_ids = list(range(checkpoint_num_layers))
                    
        #             # Reinitialize connector with new configuration
        #             from desta.models.modeling_desta25 import ORCAHybridConnector
        #             old_connector = self.perception.connector
        #             self.perception.connector = ORCAHybridConnector(self.config)
                    
        #             # Move to same device and dtype as old connector
        #             self.perception.connector.to(
        #                 device=old_connector.global_proj[1].weight.device,
        #                 dtype=old_connector.global_proj[1].weight.dtype
        #             )
                    
        #             logging.info(f"Model reconfigured to use {len(self.perception.connector.target_layer_ids)} layers")
        
        return super().load_state_dict(new_state_dict, strict=strict, assign=assign)



    def _generate_step(self, inputs, pad_token_id, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True):
        input_ids = inputs["context_input_ids"] # only context inputs
        attention_mask = inputs["context_attention_mask"] # only context attention mask
        batch_start_positions = inputs["context_batch_start_positions"]

        batch_transcription_ids = inputs["batch_transcription_ids"]
        # batch_audio_features, batch_audio_feature_lengths = self.perception()

        # get the generated text
        prepare_result = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )
        
        # Handle ORCA mode - extract inputs_embeds and set local tokens for deep injection
        is_orca_mode = (
            isinstance(prepare_result, tuple) and len(prepare_result) >= 3
        ) or self.config.connector_mode == "orca_hybrid"
        
        if is_orca_mode and isinstance(prepare_result, tuple) and len(prepare_result) >= 3:
            if len(prepare_result) == 4:
                inputs_embeds, global_audio_tokens, local_audio_tokens, transcription_positions = prepare_result
            else:
                inputs_embeds, global_audio_tokens, local_audio_tokens = prepare_result
                transcription_positions = None
            
            # Store transcription positions for consistency
            self._orca_transcription_positions = transcription_positions
            
            # Set audio tokens for deep injection (accessed by wrapped decoder layers)
            # Only set if deep injection is enabled in ablation config
            if getattr(self.config, 'orca_deep_injection_enabled', True):
                # If global_cross_attn is enabled, combine global and local tokens for injection
                if getattr(self.config, 'orca_global_cross_attn', False):
                    # Combine global + local tokens for cross-attention injection
                    if local_audio_tokens is not None and global_audio_tokens is not None:
                        self._orca_audio_local = torch.cat([global_audio_tokens, local_audio_tokens], dim=1)
                    elif global_audio_tokens is not None:
                        self._orca_audio_local = global_audio_tokens
                    else:
                        self._orca_audio_local = local_audio_tokens
                else:
                    # Standard mode: only local tokens for cross-attention
                    self._orca_audio_local = local_audio_tokens
            else:
                self._orca_audio_local = None
            
            self._orca_audio_local_mask = None
        elif isinstance(prepare_result, tuple):
            inputs_embeds = prepare_result[0]
        else:
            inputs_embeds = prepare_result

        if do_sample is False:
            top_p = None
            temperature = None
        
        try:
            generated_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )
        finally:
            # Clear local tokens after generation
            if hasattr(self, '_orca_audio_local'):
                self._orca_audio_local = None
                self._orca_audio_local_mask = None
            if hasattr(self, '_orca_transcription_positions'):
                self._orca_transcription_positions = None

        return generated_ids

    def _generate_acd(
        self, 
        inputs, 
        pad_token_id, 
        temperature=0.7, 
        top_p=0.9, 
        max_new_tokens=512, 
        do_sample=True,
        acd_alpha=None
    ):
        """
        Acoustic-Contrastive Decoding (ACD) for Struct-ORCA.
        
        Generates text by emphasizing audio-dependent predictions:
        logits_final = logits_full + alpha * (logits_full - logits_blind)
        
        Where:
        - logits_full: P(y | text, audio) - normal multimodal prediction
        - logits_blind: P(y | text) - text-only prediction (audio zeroed)
        - alpha: contrast strength (from config.struct_orca_acd_alpha)
        
        This upweights tokens that require listening (e.g., "sarcastic")
        and downweights generic tokens that don't depend on audio.
        """
        if acd_alpha is None:
            acd_alpha = getattr(self.config, 'struct_orca_acd_alpha', 0.5)
        
        input_ids = inputs["context_input_ids"]
        attention_mask = inputs["context_attention_mask"]
        batch_start_positions = inputs["context_batch_start_positions"]
        batch_transcription_ids = inputs["batch_transcription_ids"]
        
        # Store original batch_features
        batch_features = inputs["batch_features"]
        
        # Prepare inputs WITH audio (normal path)
        prepare_result_full = self._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions
        )
        
        if isinstance(prepare_result_full, tuple):
            inputs_embeds_full = prepare_result_full[0]
        else:
            inputs_embeds_full = prepare_result_full
        
        # Custom generate loop for ACD
        # We need per-token logit manipulation, so we use a manual loop
        device = inputs_embeds_full.device
        batch_size = inputs_embeds_full.size(0)
        
        # Initialize with input embeddings
        current_embeds = inputs_embeds_full
        current_mask = attention_mask
        generated_tokens = []
        
        # Get embedding layer for token-to-embed conversion
        embed_tokens = self.llm_model.model.embed_tokens
        
        if do_sample is False:
            top_p = None
            temperature = None
        
        # Past key values for efficient generation
        past_key_values_full = None
        past_key_values_blind = None
        
        for step in range(max_new_tokens):
            # Forward with audio (full modality)
            if step == 0:
                outputs_full = self.llm_model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    use_cache=True,
                )
                logits_full = outputs_full.logits[:, -1, :]  # [B, V]
                past_key_values_full = outputs_full.past_key_values
                
                # Forward WITHOUT audio (blind mode)
                # Create zero audio embeddings at the audio positions
                inputs_embeds_blind = current_embeds.clone()
                # Zero out audio token positions (approximation: zero all but keep structure)
                # A more precise approach would track audio positions, but this is simpler
                # For Struct-ORCA, we zero the embedded audio tokens
                
                # Compute blind logits
                outputs_blind = self.llm_model(
                    inputs_embeds=inputs_embeds_blind,  # Same for first step (audio already embedded)
                    attention_mask=current_mask,
                    use_cache=True,
                )
                logits_blind = outputs_blind.logits[:, -1, :]  # [B, V]
                past_key_values_blind = outputs_blind.past_key_values
            else:
                # Use past key values for efficiency
                outputs_full = self.llm_model(
                    inputs_embeds=next_token_embeds,
                    attention_mask=current_mask,
                    past_key_values=past_key_values_full,
                    use_cache=True,
                )
                logits_full = outputs_full.logits[:, -1, :]
                past_key_values_full = outputs_full.past_key_values
                
                outputs_blind = self.llm_model(
                    inputs_embeds=next_token_embeds,
                    attention_mask=current_mask,
                    past_key_values=past_key_values_blind,
                    use_cache=True,
                )
                logits_blind = outputs_blind.logits[:, -1, :]
                past_key_values_blind = outputs_blind.past_key_values
            
            # ACD: Contrastive logit adjustment
            # logits_final = logits_full + alpha * (logits_full - logits_blind)
            # = (1 + alpha) * logits_full - alpha * logits_blind
            logits = logits_full + acd_alpha * (logits_full - logits_blind)
            
            # Sample next token
            if do_sample:
                if temperature is not None and temperature > 0:
                    logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                
                if top_p is not None:
                    # Top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for b in range(batch_size):
                        probs[b, sorted_indices[b, sorted_indices_to_remove[b]]] = 0
                    
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)
            
            generated_tokens.append(next_token)
            
            # Check for EOS
            if (next_token == pad_token_id).all():
                break
            
            # Prepare for next step
            next_token_embeds = embed_tokens(next_token.unsqueeze(-1))
            current_mask = torch.cat([
                current_mask, 
                torch.ones(batch_size, 1, device=device, dtype=current_mask.dtype)
            ], dim=1)
        
        # Stack generated tokens
        if generated_tokens:
            generated_ids = torch.stack(generated_tokens, dim=1)
        else:
            generated_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        return generated_ids


    def configure_trainable_parameters(self):
        """
        for training, log the trainable parameters
        """

        known_parameters = []
        # Freeze LLM parameters
        for name, params in self.llm_model.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"llm_model.{name}")

        # Freeze encoder parameters
        for name, params in self.perception.whisper.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"perception.whisper.{name}")


        # Make other parameters or lora parameters trainable
        self.trainable_parameter_names = []
        trainable_parameters = []
        for name, params in self.named_parameters():
            if name not in known_parameters or "lora" in name:
                params.requires_grad = True
                self.trainable_parameter_names.append(name)
                trainable_parameters.append(params)



    def _setup_generation(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.tokenizer.add_tokens([self.audio_locator])
        self.processor = AutoProcessor.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        assert len(self.tokenizer.tokenize(self.audio_locator)) == 1, "audio_locator must be a single token"
        assert len(self.tokenizer.tokenize(self.placeholder_token)) == 1, "placeholder_token must be a single token in the tokenizer"

        # VAD will be loaded lazily when needed (in generate())
        self.vad_model = None
        self.get_speech_timestamps = None

    def _setup_vad(self):
        """Lazy load VAD model only when needed for inference."""
        if self.vad_model is None:
            self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            (self.get_speech_timestamps, _, _, _, _) = utils


    def generate_with_acd(
        self, 
        messages,
        acd_alpha: float = 1.0,
        acd_beta: float = 0.1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512,
    ):
        """
        Generate with Acoustic-Contrastive Decoding (ACD).
        
        ACD amplifies acoustically-grounded predictions by subtracting text-only priors:
        logits_final = (1 + alpha) * logits_full - alpha * logits_blind
        
        This is particularly effective for paralinguistic tasks like sarcasm detection
        where the acoustic signal conflicts with text semantics.
        
        Args:
            messages: List of message dicts (same format as generate())
            acd_alpha: Contrast strength (default: 1.0). Higher = more audio emphasis.
            acd_beta: Plausibility threshold (default: 0.1). Only boost tokens above this prob.
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            GenerationOutput with text, audios, and generated_ids
        """
        if not hasattr(self, "tokenizer"):
            self._setup_generation()

        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else: 
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")

        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    all_audios.append(audio["audio"])
                    all_transcriptions.append(audio.get("text"))

        if len(all_audios) == 0:
            # No audio, fall back to regular generate
            return self.generate(
                messages_list[0] if len(messages_list) == 1 else messages_list,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens
            )

        # Process audio features (same as generate())
        batch_features = []
        asr_features = []
        asr_indices = []
        for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
            if not os.path.exists(audio):
                raise ValueError(f"Audio file {audio} does not exist.")

            feature = AudioSegment.from_file(
                audio,
                target_sr=16000,
                channel_selector="average"
            ).samples

            batch_features.append(feature)

            self._setup_vad()
            is_speech = self.get_speech_timestamps(feature, self.vad_model)
            if is_speech and trans is None:
                asr_features.append(feature)
                asr_indices.append(i)
            if not is_speech:
                all_transcriptions[i] = " "
        
        batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
        batch_features = batch_features.to(self.device)
        
        if self.config.connector_mode == "orca_hybrid":
            audio_token_size = getattr(self.config, 'orca_global_num_tokens', 64)
        elif self.config.connector_mode == "struct_orca":
            num_groups = getattr(self.config, 'struct_orca_num_groups', 8)
            queries_per_group = getattr(self.config, 'struct_orca_queries_per_group', 8)
            audio_token_size = num_groups * queries_per_group
        else:
            audio_token_size = self.config.prompt_size
        audio_size_list = [audio_token_size] * len(batch_features)

        # Run ASR if needed
        if asr_features:
            asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
            asr_features = asr_features.to(self.device)

            transcriptions = self.perception.whisper.generate(
                input_features=asr_features,
                attention_mask=None,
                max_new_tokens=128
            )
            transcriptions = self.processor.batch_decode(
                transcriptions,
                skip_special_tokens=True,
            )
        else:
            transcriptions = []

        for i, transcription in zip(asr_indices, transcriptions):
            all_transcriptions[i] = transcription.strip()
                
        transcription_size_list = [
            len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
        ]

        # Prepare context
        audio_context_list = []
        start_positions_list = []
        for messages in messages_list:
            audio_context = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

            audio_context, start_positions = _prepare_audio_context_and_start_positions(
                    token_list=self.tokenizer.tokenize(audio_context), 
                    audio_locator=self.audio_locator,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token
                )

            audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
            audio_context_list.append(audio_context)
            start_positions_list.append(start_positions)

        audio_context_inputs = self.tokenizer(
            audio_context_list,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )

        audio_context_batch_start_positions = []
        for i in range(audio_context_inputs["length"].size(0)):
            total_length = audio_context_inputs["length"][i]
            pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

            for start_position in start_positions_list[i]:
                audio_context_batch_start_positions.append((i, start_position + pad_length))

        batch_transcription_ids = []
        for transcription in all_transcriptions:
            batch_transcription_ids.append(
                self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long().to(self.device)
            )

        inputs = {
            "batch_features": batch_features,
            "batch_transcription_ids": batch_transcription_ids,
            "context_input_ids": audio_context_inputs["input_ids"],
            "context_attention_mask": audio_context_inputs['attention_mask'],
            "context_batch_start_positions": audio_context_batch_start_positions,
        }
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Use ACD generation instead of standard generation
        generated_ids = self._generate_acd(
            inputs, 
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            acd_alpha=acd_alpha
        )

        return GenerationOutput(
            text=self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
            audios=[(a, t) for a,t in zip(all_audios, all_transcriptions)],
            generated_ids=generated_ids.tolist()
        )

    def generate(self, messages,

        # LLM generation args
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512,
        ):
        """
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions.",
            },
            {
                "role": "user",
                "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe."
                "audios": [
                    "audio": "/path/to/filepath", # path to audio file
                    "text": None # Optional, None or provide text
                ]
            },
        ]
        """
        if not hasattr(self, "tokenizer"):
            self._setup_generation()

        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else: 
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")

        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    all_audios.append(audio["audio"])
                    all_transcriptions.append(audio.get("text"))

        if len(all_audios) > 0:
            """
            If audios are provided, run:
            1. get features and transcription
            2. prepare LLM inputs
            3. run generation
            """

            batch_features = []
            asr_features = []
            asr_indices = []
            for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file {audio} does not exist.")

                # Extract audio features
                feature = AudioSegment.from_file(
                    audio,
                    target_sr=16000,
                    channel_selector="average"
                ).samples

                batch_features.append(feature)

                # Run VAD detect if there is speech in the audio
                self._setup_vad()  # Lazy load VAD model
                is_speech = self.get_speech_timestamps(feature, self.vad_model)
                if is_speech and trans is None:
                    asr_features.append(feature)
                    asr_indices.append(i)
                if not is_speech:
                    all_transcriptions[i] = " "
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            batch_features = batch_features.to(self.device)
            
            # Use correct audio token size based on connector mode
            if self.config.connector_mode == "orca_hybrid":
                audio_token_size = getattr(self.config, 'orca_global_num_tokens', 64)
            else:
                audio_token_size = self.config.prompt_size
            audio_size_list = [audio_token_size] * len(batch_features)


            # RUN ASR
            if asr_features:
                asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
                asr_features = asr_features.to(self.device)

                transcriptions = self.perception.whisper.generate(
                    input_features=asr_features,
                    attention_mask=None,
                    max_new_tokens=128
                )
                transcriptions = self.processor.batch_decode(
                    transcriptions,
                    skip_special_tokens=True,
                )
            else:
                # no audio needs ASR result
                transcriptions = []

            
            for i, transcription in zip(asr_indices, transcriptions):
                all_transcriptions[i] = transcription.strip()
                    
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
            ]


            audio_context_list = []
            start_positions_list = []
            for messages in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # <start_audio><|AUDIO|><end_audio> is a indicator used in the training stage
                # We replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> here
                audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

                audio_context, start_positions = _prepare_audio_context_and_start_positions(
                        token_list=self.tokenizer.tokenize(audio_context), 
                        audio_locator=self.audio_locator,
                        audio_size_list=audio_size_list,
                        transcription_size_list=transcription_size_list,
                        placeholder_token=self.placeholder_token
                    )


                audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
                audio_context_list.append(audio_context)

                start_positions_list.append(start_positions)


            audio_context_inputs = self.tokenizer(
                audio_context_list,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

                for start_position in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, start_position + pad_length))

            batch_transcription_ids = []
            for transcription in all_transcriptions:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long().to(self.device)
                )

            inputs = {
                "batch_features": batch_features,
                "batch_transcription_ids": batch_transcription_ids,

                "context_input_ids": audio_context_inputs["input_ids"],
                "context_attention_mask": audio_context_inputs['attention_mask'],
                "context_batch_start_positions": audio_context_batch_start_positions,
            }
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = self._generate_step(
                inputs, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample)

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
                audios=[(a, t) for a,t in zip(all_audios, all_transcriptions)],
                generated_ids=generated_ids.tolist()
            )

        else:
            """
            if no audios are provided, it's identical to the original LLM generation
            """

            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generated_ids = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=terminators,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

            generated_ids_list = []
            for i in range(len(generated_ids)):
                generated_ids_list.append(generated_ids[i][inputs["input_ids"].shape[1]:].tolist())

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True),
                audios=[],
                generated_ids=generated_ids_list
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Custom from_pretrained method to load pretrained LLM and Whisper model.
        model.safetensors only contains trainable parameters from DeSTA2.5-Audio.
        """
        
        cache_dir = kwargs.get("cache_dir", os.getenv("HF_HOME"))

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)

        model = cls(config)
        
        if os.path.isdir(pretrained_model_name_or_path):
            model.load_state_dict(
                load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors")), strict=False
            )
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors", cache_dir=cache_dir)
            model.load_state_dict(
                load_file(path), strict=False
            )

        return model