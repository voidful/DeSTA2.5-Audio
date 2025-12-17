
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


def sinusoidal_position_embedding(
    positions: torch.Tensor, 
    dim: int, 
    rope_theta: float = 10000.0
) -> torch.Tensor:
    """
    Generate sinusoidal position embeddings for given positions.
    Uses the same frequency calculation as RoPE to ensure consistency with LLM.
    Supports fractional positions for interpolation/compression.
    
    Args:
        positions: Position indices [B, T] or [T], can be fractional for interpolation
        dim: Embedding dimension
        rope_theta: Base frequency for RoPE (from LLM config, e.g., 10000 for Qwen, 500000 for Llama-3.1)
        
    Returns:
        Position embeddings [B, T, dim] or [T, dim]
    """
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float, device=positions.device)
    # Use rope_theta from LLM config for consistency
    inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
    
    # Handle both 1D and 2D position tensors
    if positions.dim() == 1:
        positions = positions.unsqueeze(-1)  # [T, 1]
        sinusoid = positions * inv_freq  # [T, half_dim]
        pos_embed = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # [T, dim]
    else:
        positions = positions.unsqueeze(-1)  # [B, T, 1]
        sinusoid = positions * inv_freq  # [B, T, half_dim]
        pos_embed = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # [B, T, dim]
    
    return pos_embed

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
            qformer_config._attn_implementation = "eager"

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
    ORCA Hybrid Connector with dual-branch architecture:
    - Global branch: Q-Former style cross-attention for style tokens
    - Local branch: Conv1d downsampling for prosody tokens
    
    Returns (global_tokens, local_tokens) tuple.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config
        
        # Determine target layer IDs based on Whisper model
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
        qformer_config._attn_implementation = "eager"
        
        self.global_qformer = BertEncoder(qformer_config)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(d_encoder),
            nn.Linear(d_encoder, d_llm)
        )
        
        # === Local Branch (Conv1d downsampling) - only if enabled ===
        self.local_enabled = config.orca_local_enabled
        
        if self.local_enabled:
            kernel_size = config.orca_local_kernel_size
            stride = config.orca_local_downsample
            padding = kernel_size // 2
            
            # Learnable weights for layer fusion (same target layers as global)
            self.local_layer_weights = nn.Parameter(
                torch.zeros(len(self.target_layer_ids), dtype=torch.float)
            )
            
            self.local_proj_in = nn.Linear(d_encoder, d_llm)
            self.local_conv = nn.Conv1d(
                in_channels=d_llm,
                out_channels=d_llm,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.local_ln = nn.LayerNorm(d_llm)
    
    def forward(
        self, 
        encoder_hidden_states: List[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ORCAHybridConnector.
        
        Args:
            encoder_hidden_states: List of hidden states from Whisper encoder layers
            audio_attention_mask: Optional attention mask for audio [B, T]
            
        Returns:
            Tuple of (global_tokens, local_tokens):
                - global_tokens: [B, K_global, d_llm]
                - local_tokens: [B, T_local, d_llm]
        """
        batch_size = encoder_hidden_states[0].size(0)
        
        # Collect target layer outputs (used by both branches)
        target_layer_outputs = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            if idx in self.target_layer_ids:
                target_layer_outputs.append(hidden_state)
        
        # === Global Branch ===
        global_outputs = []
        for layer_idx, hidden_state in enumerate(target_layer_outputs):
            queries = self.global_queries[layer_idx].expand(batch_size, -1, -1)
            
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
        
        # === Local Branch (only if enabled) ===
        if self.local_enabled:
            # Learnable weighted sum of target layers (same layers as global)
            target_layers = torch.stack(target_layer_outputs, dim=0)  # [L, B, T, D]
            target_layers = target_layers.permute(1, 2, 0, 3)  # [B, T, L, D]
            local_weights = torch.softmax(self.local_layer_weights, dim=-1).unsqueeze(-1)  # [L, 1]
            fused_hidden = (target_layers * local_weights).sum(dim=2)  # [B, T, D]
            
            # Project to LLM dimension
            local_features = self.local_proj_in(fused_hidden)  # [B, T, d_llm]
            
            # Conv1d downsampling: [B, T, D] -> [B, D, T] -> conv -> [B, D, T'] -> [B, T', D]
            local_features = local_features.transpose(1, 2)  # [B, D, T]
            local_features = self.local_conv(local_features)  # [B, D, T']
            local_features = local_features.transpose(1, 2)  # [B, T', D]
            local_tokens = self.local_ln(local_features)  # [B, T_local, d_llm]
            
            # Apply interpolated position embedding (RoPE-style, with compression)
            # Use fractional positions to compress position range
            audio_position_scale = getattr(self.config, 'orca_audio_position_scale', 4.0)
            seq_len = local_tokens.size(1)
            d_llm = local_tokens.size(2)
            
            # Get rope_theta from LLM config for consistency with text position encoding
            rope_theta = getattr(self.config.llm_config, 'rope_theta', 10000.0)
            
            # Generate fractional positions: [0, 1/scale, 2/scale, ...]
            # This compresses position range - e.g., with scale=4, 1000 tokens span position 0-249.75
            positions = torch.arange(seq_len, dtype=torch.float, device=local_tokens.device) / audio_position_scale
            positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, T']
            
            # Add sinusoidal position embeddings using LLM's rope_theta
            pos_embed = sinusoidal_position_embedding(positions, d_llm, rope_theta=rope_theta)  # [B, T', d_llm]
            local_tokens = local_tokens + pos_embed.to(local_tokens.dtype)
        else:
            local_tokens = None
        
        return global_tokens, local_tokens

class ORCAGatedCrossAttention(nn.Module):
    """
    Gated cross-attention module for deep injection of audio tokens into LLM decoder layers.
    Uses data-dependent gating (following Audio Flamingo 3 design).
    Also computes per-layer alignment loss for layer-wise supervision.
    
    hidden_out = hidden + gate(hidden) * LayerNorm(CrossAttn(hidden, audio))
    """
    def __init__(self, hidden_size: int, num_heads: int, gate_init: float = 0.1):
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
        
        # Build key_padding_mask: True for positions to IGNORE
        if audio_local_mask is not None:
            key_padding_mask = ~audio_local_mask.bool()
        else:
            key_padding_mask = None
        
        # Cast cross_attn to same dtype as hidden_states
        cross_out, _ = self.cross_attn.to(dtype=hidden_states.dtype)(
            query=hidden_states,
            key=audio_local,
            value=audio_local,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        cross_out = self.ln.to(dtype=hidden_states.dtype)(cross_out)
        
        # Data-dependent gate: compute gate from hidden states
        gate = torch.sigmoid(self.gate_proj.to(dtype=hidden_states.dtype)(hidden_states))  # [B, T, 1]
        
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
        else:
            self.connector = QformerConnector(config)


    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, transcription_embeddings_list: Optional[List[torch.Tensor]] = None, **kwargs) -> Union[Tuple[torch.Tensor, List[int]], Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Forward pass of the WhisperPerception.

        Args:
            input_features (torch.Tensor): Input mel features.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            transcription_embeddings_list (Optional[List[torch.Tensor]], optional): List of transcription embeddings. Defaults to None.

        Returns:
            For qformer_1: tuple[torch.Tensor, list[int]]: (audio_features, speech_feature_lengths)
            For orca_hybrid: tuple[torch.Tensor, torch.Tensor, list[int]]: (global_tokens, local_tokens, global_lengths)
        """
        bs = input_features.size(0)

        result = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        
        if self.config.connector_mode == "orca_hybrid":
            # result is (global_tokens, local_tokens)
            global_tokens, local_tokens = result
            speech_feature_lengths = [self.config.orca_global_num_tokens] * bs
            return global_tokens, local_tokens, speech_feature_lengths
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
        
        expected_seq_length = self.whisper.model.encoder.config.max_source_positions * self.whisper.model.encoder.conv1.stride[0] * self.whisper.model.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        

        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.whisper.model.encoder.embed_positions.weight[:self.whisper.model.encoder.config.max_source_positions, :] # @kehan

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
            global_tokens, local_tokens = self.connector(all_layer_outputs)
            return global_tokens, local_tokens

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
                 orca_local_enabled=True,  # If False, only global tokens are used (no local downsample)
                 orca_global_cross_attn=False,  # If True, global tokens also use cross-attention instead of concat
                 orca_audio_position_scale=4.0,  # Position interpolation scale for audio tokens (higher = more compression)
                 orca_global_num_tokens=4,
                 orca_local_downsample=4,
                 orca_local_kernel_size=7,
                 orca_gate_init=0.1,
                 orca_ortho_weight_global=0.01,
                 orca_ortho_diversity_weight=0.01,
                 orca_ortho_weight_qformer_local=0.01,  # Orthogonality between Q-Former global and local tokens
                 orca_align_weight_local=0.05,  # Alignment loss to bring local tokens closer to text embeddings
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
        self.orca_local_enabled = orca_local_enabled
        self.orca_global_cross_attn = orca_global_cross_attn
        self.orca_audio_position_scale = orca_audio_position_scale
        self.orca_global_num_tokens = orca_global_num_tokens
        self.orca_local_downsample = orca_local_downsample
        self.orca_local_kernel_size = orca_local_kernel_size
        self.orca_gate_init = orca_gate_init
        self.orca_ortho_weight_global = orca_ortho_weight_global
        self.orca_ortho_diversity_weight = orca_ortho_diversity_weight
        self.orca_ortho_weight_qformer_local = orca_ortho_weight_qformer_local
        self.orca_align_weight_local = orca_align_weight_local

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
            
            # Enable deep cross-attention injection
            self._enable_orca_deep_injection()
            
            # Storage for audio_local during forward (set before LLM call, cleared after)
            self._orca_audio_local = None
            self._orca_audio_local_mask = None

        self.configure_trainable_parameters()

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
            isinstance(prepare_result, tuple) and len(prepare_result) >= 3
        ) or self.config.connector_mode == "orca_hybrid"
        
        if is_orca_mode and isinstance(prepare_result, tuple) and len(prepare_result) >= 3:
            if len(prepare_result) == 4:
                inputs_embeds, global_audio_tokens, local_audio_tokens, transcription_positions = prepare_result
            else:
                inputs_embeds, global_audio_tokens, local_audio_tokens = prepare_result
                transcription_positions = None
            
            # Store transcription positions for cross-attention alignment loss
            self._orca_transcription_positions = transcription_positions
            
            # Set audio tokens for deep injection (accessed by wrapped decoder layers)
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
            
            # Compute ORCA losses
            orca_losses = self.compute_orca_losses(
                global_tokens=global_audio_tokens,
                local_tokens=local_audio_tokens,
                text_hidden=text_hidden,
                layer_align_losses=layer_align_losses,
            )
            
            # Attach losses to outputs
            outputs.orca_losses = orca_losses
            outputs.audio_global = global_audio_tokens
            outputs.audio_local = local_audio_tokens
            
            return outputs
        else:
            # Standard non-ORCA path
            inputs_embeds = prepare_result
            
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
            For ORCA: (inputs_embeds, global_audio_tokens, local_audio_tokens)
        """

        N_audio = len(batch_start_positions)
        device = next(self.llm_model.parameters()).device
        
        # Handle empty audio case
        if N_audio == 0:
            embeds = self.llm_model.model.embed_tokens(input_ids)
            if self.config.connector_mode == "orca_hybrid":
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
            isinstance(perception_output, tuple) and len(perception_output) == 3
        ) or self.config.connector_mode == "orca_hybrid"
        
        if is_orca_output:
            # perception_output is (global_tokens, local_tokens, lengths)
            batch_global_tokens, batch_local_tokens, batch_audio_feature_lengths = perception_output
            batch_audio_features = batch_global_tokens  # Global tokens are what we splice
        else:
            # perception_output is (audio_features, lengths)
            batch_audio_features, batch_audio_feature_lengths = perception_output
            batch_global_tokens = None
            batch_local_tokens = None

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
            return inputs_embeds, batch_global_tokens, batch_local_tokens, transcription_positions

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
        
        for layer_idx in range(actual_num_layers):
            cross_attn = ORCAGatedCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                gate_init=gate_init,
            )
            self.orca_cross_attns.append(cross_attn)
        
        # Wrap each layer's forward method
        for layer_idx, layer in enumerate(layers):
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
        local_tokens: Optional[torch.Tensor],
        text_hidden: Optional[torch.Tensor],
        layer_align_losses: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ORCA auxiliary losses:
        - Global-text orthogonality loss
        - Global token diversity loss
        - Global-local orthogonality loss
        - Layer-wise alignment loss (aggregated from cross-attention modules)
        """
        losses = {}
        
        if global_tokens is not None and text_hidden is not None:
            # Global-text orthogonality: encourage global tokens to be orthogonal to text
            g = F.normalize(global_tokens, dim=-1)  # [B, K, H]
            t = F.normalize(text_hidden, dim=-1)    # [B, T, H]
            
            # Compute cosine similarity
            cos = torch.einsum("bkh,bth->bkt", g, t)  # [B, K, T]
            L_ortho = (cos ** 2).mean()
            losses["L_ortho_global"] = self.config.orca_ortho_weight_global * L_ortho
            
            # Diversity between global tokens (Gram matrix close to identity)
            gram = torch.einsum("bkh,bqh->bkq", g, g)  # [B, K, K]
            I = torch.eye(gram.size(-1), device=gram.device)
            L_div = ((gram - I) ** 2).mean()
            losses["L_ortho_diversity"] = self.config.orca_ortho_diversity_weight * L_div
        
        # Q-Former global vs local orthogonality loss
        if global_tokens is not None and local_tokens is not None:
            g = F.normalize(global_tokens, dim=-1)  # [B, Kg, H]
            l = F.normalize(local_tokens, dim=-1)   # [B, Tl, H]
            cos_gl = torch.einsum("bgh,blh->bgl", g, l)  # [B, Kg, Tl]
            L_ortho_ql = (cos_gl ** 2).mean()
            losses["L_ortho_qformer_local"] = self.config.orca_ortho_weight_qformer_local * L_ortho_ql
        
        # Layer-wise alignment loss: aggregated from cross-attention modules
        # Each layer computes alignment between audio and text at that layer's representation
        if layer_align_losses is not None and len(layer_align_losses) > 0:
            L_align_layerwise = torch.stack(layer_align_losses).mean()
            losses["L_align_layerwise"] = self.config.orca_align_weight_local * L_align_layerwise
        
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
        local_tokens = None
        if isinstance(prepare_result, tuple) and len(prepare_result) == 3:
            inputs_embeds, global_tokens, local_tokens = prepare_result
            # Set local tokens for deep injection during generation
            if local_tokens is not None:
                self._orca_audio_local = local_tokens
                self._orca_audio_local_mask = None
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