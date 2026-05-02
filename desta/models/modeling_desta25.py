
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass
from desta.utils.audio import AudioSegment

from transformers import AutoTokenizer, AutoProcessor
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import WhisperForConditionalGeneration, BertConfig
from safetensors.torch import load_file
import torch.distributed as dist


GROUPWISE_ORTHO_MODE = "groupwise_ortho"
GROUPWISE_ORTHO_ALIASES = {GROUPWISE_ORTHO_MODE, "orca_desta", "orca_r1"}
LEGACY_CONNECTOR_MODES = {"qformer_1", "orca_hybrid"}


def _canonical_connector_mode(mode: str) -> str:
    if mode in GROUPWISE_ORTHO_ALIASES:
        return GROUPWISE_ORTHO_MODE
    return mode


def _is_groupwise_ortho_mode(mode: str) -> bool:
    return _canonical_connector_mode(mode) == GROUPWISE_ORTHO_MODE


def _get_groupwise_ortho_audio_token_size(config: "DeSTA25Config") -> int:
    return config.orca_r1_num_groups * config.orca_r1_queries_per_group


def _get_audio_token_size(config: "DeSTA25Config") -> int:
    if _is_groupwise_ortho_mode(config.connector_mode):
        return _get_groupwise_ortho_audio_token_size(config)
    raise ValueError(f"connector_mode='{config.connector_mode}' is not supported in the main model path.")






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
            # Robust check for audio locator (handles tokenizer prefixes like Ġ<|AUDIO|>)
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


class GroupwiseOrthogonalConnector(nn.Module):
    """
    Groupwise-orthogonal Q-Former connector.

    This is the active connector in the main package path: grouped Q-Former
    tokens with inter-group and intra-group orthogonality losses.
    """
    def __init__(self, config: 'DeSTA25Config'):
        super().__init__()
        self.config = config
        
        # Group settings
        self.num_groups = getattr(config, 'orca_r1_num_groups', 8)
        self.queries_per_group = getattr(config, 'orca_r1_queries_per_group', 8)
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
        self.inter_group_weight = getattr(config, 'orca_r1_inter_group_weight', 0.1)
        self.intra_group_weight = getattr(config, 'orca_r1_intra_group_weight', 0.01)
        
    @staticmethod
    def _sanitize_token_tensor(tensor: torch.Tensor, clip_value: float = 5.0) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip_value, neginf=-clip_value)
        return tensor.clamp(min=-clip_value, max=clip_value)

    
    def forward(
        self, 
        encoder_hidden_states: List[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
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
                hidden_state = hidden_state.to(dtype=target_dtype, device=target_device)
                target_layer_outputs.append(self._sanitize_token_tensor(hidden_state))
        
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
            layer_outputs.append(self._sanitize_token_tensor(qformer_out.last_hidden_state))  # [B, total_queries, d_encoder]
        
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
            group_tokens = self._sanitize_token_tensor(group_tokens)
            
            all_group_tokens.append(group_tokens)
            
            # Compute group centroid (mean of queries in this group)
            centroid = group_tokens.mean(dim=1)  # [B, d_llm]
            group_centroids.append(centroid)
        
        # Concatenate all groups: [B, num_groups * queries_per_group, d_llm]
        global_tokens = torch.cat(all_group_tokens, dim=1)
        global_tokens = self._sanitize_token_tensor(global_tokens)
        

        
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
        centroids = torch.stack(group_centroids, dim=1).float()  # [B, num_groups, d_llm]
        centroids = self._sanitize_token_tensor(centroids)
        centroids_norm = F.normalize(centroids, dim=-1, eps=1e-6)  # [B, num_groups, d_llm]
        
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
            group_tokens = self._sanitize_token_tensor(group_tokens.float())
            tokens_norm = F.normalize(group_tokens, dim=-1, eps=1e-6)
            intra_gram = torch.einsum("bkh,bqh->bkq", tokens_norm, tokens_norm)  # [B, K, K]
            
            # Light diversity: penalize if all queries are identical (but allow some correlation)
            # Using softer target: allow diagonal=1, off-diagonal can be up to 0.5
            K = self.queries_per_group
            target = torch.eye(K, device=intra_gram.device) * 0.5 + 0.5  # Target: diagonal=1, off-diag=0.5
            L_intra_total += ((intra_gram - target) ** 2).mean()
        
        losses["L_intra_group"] = self.intra_group_weight * (L_intra_total / self.num_groups)
        
        return losses




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

        # The main package intentionally keeps only the groupwise-orthogonal
        # Q-Former connector. Baseline Q-Former and experimental variants live
        # in desta.models.legacy.modeling_desta25_experiments.
        if _is_groupwise_ortho_mode(config.connector_mode):
            self.connector = GroupwiseOrthogonalConnector(config)
        elif config.connector_mode in LEGACY_CONNECTOR_MODES:
            raise ValueError(
                f"connector_mode='{config.connector_mode}' is a legacy experiment. "
                "Use desta.models.legacy.modeling_desta25_experiments for old checkpoints, "
                "or retrain with connector.mode=groupwise_ortho."
            )
        else:
            raise NotImplementedError(
                f"connector_mode '{config.connector_mode}' is not supported. "
                "Use 'groupwise_ortho'."
            )
        
        # Store groupwise-orthogonal connector losses (populated during forward)
        self._groupwise_ortho_losses = None
        self._use_safe_whisper_encoder_layer = False
        self._warned_safe_whisper_encoder_layer = False
        

    def _forward_encoder_layer(self, encoder_layer, hidden_states, attention_mask=None):
        """
        Run a Whisper encoder layer with a fallback explicit attention layout.

        Some Transformers/PyTorch combinations route Whisper through SDPA with
        query shaped as [B, T, H, D] while key/value are [B, H, T, D], which
        raises a 1500-vs-head-count shape error. The fallback keeps Q/K/V
        consistently [B, H, T, D] before calling SDPA.
        """
        if not getattr(self, "_use_safe_whisper_encoder_layer", False):
            try:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=None,
                    output_attentions=False,
                )
                if isinstance(layer_outputs, torch.Tensor):
                    return layer_outputs, None
                return layer_outputs
            except RuntimeError as exc:
                if "must match the size of tensor" not in str(exc):
                    raise
                self._use_safe_whisper_encoder_layer = True
                if not getattr(self, "_warned_safe_whisper_encoder_layer", False):
                    logging.warning(
                        "Falling back to safe Whisper encoder attention after SDPA shape mismatch: %s",
                        exc,
                    )
                    self._warned_safe_whisper_encoder_layer = True

        residual = hidden_states
        hidden_states = encoder_layer.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self._forward_whisper_self_attention(
            encoder_layer.self_attn,
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=encoder_layer.dropout, training=encoder_layer.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = encoder_layer.final_layer_norm(hidden_states)
        hidden_states = encoder_layer.activation_fn(encoder_layer.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=encoder_layer.activation_dropout,
            training=encoder_layer.training,
        )
        hidden_states = encoder_layer.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=encoder_layer.dropout, training=encoder_layer.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


    @staticmethod
    def _forward_whisper_self_attention(attn_module, hidden_states, attention_mask=None):
        bsz, tgt_len, _ = hidden_states.size()
        query_states = attn_module.q_proj(hidden_states) * attn_module.scaling
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        query_states = query_states.view(
            bsz,
            tgt_len,
            attn_module.num_heads,
            attn_module.head_dim,
        ).transpose(1, 2).contiguous()
        key_states = key_states.view(
            bsz,
            tgt_len,
            attn_module.num_heads,
            attn_module.head_dim,
        ).transpose(1, 2).contiguous()
        value_states = value_states.view(
            bsz,
            tgt_len,
            attn_module.num_heads,
            attn_module.head_dim,
        ).transpose(1, 2).contiguous()

        dropout_p = attn_module.dropout if attn_module.training else 0.0
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=1.0,
            is_causal=False,
        )

        embed_dim = getattr(attn_module, "embed_dim", attn_module.out_proj.in_features)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim).contiguous()
        attn_output = attn_module.out_proj(attn_output)

        return attn_output, None



    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, transcription_embeddings_list: Optional[List[torch.Tensor]] = None, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        """
        Forward pass of the WhisperPerception.

        Args:
            input_features (torch.Tensor): Input mel features.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            transcription_embeddings_list (Optional[List[torch.Tensor]], optional): List of transcription embeddings. Defaults to None.

        Returns:
            tuple[torch.Tensor, list[int]]: (global_tokens, global_lengths)
        """
        bs = input_features.size(0)

        result = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        
        if _is_groupwise_ortho_mode(self.config.connector_mode):
            global_tokens, group_losses = result
            self._groupwise_ortho_losses = group_losses
            return global_tokens, [_get_groupwise_ortho_audio_token_size(self.config)] * bs
        raise NotImplementedError(f"mode {self.config.connector_mode} not implemented")


    def forward_whisper(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        """
        Forward through Whisper encoder layers.
        """
        # Ensure input_features match Whisper's dtype
        target_dtype = self.whisper.model.encoder.conv1.weight.dtype
        target_device = self.whisper.model.encoder.conv1.weight.device
        input_features = input_features.to(dtype=target_dtype, device=target_device)
        input_features = torch.nan_to_num(input_features, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-80.0, max=80.0)
        
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
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=5.0, neginf=-5.0).clamp(min=-5.0, max=5.0)
        
        # Collect all layer outputs for the groupwise connector.
        all_layer_outputs = []

        if _is_groupwise_ortho_mode(self.config.connector_mode):
            # Collect all layer hidden states
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                layer_outputs = self._forward_encoder_layer(encoder_layer, hidden_states, attention_mask=None)
                hidden_states = layer_outputs[0]
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=5.0, neginf=-5.0).clamp(min=-5.0, max=5.0)
                all_layer_outputs.append(hidden_states)
            
            return self.connector(all_layer_outputs)

        raise NotImplementedError(f"mode {self.config.connector_mode} not implemented")
    
    

class DeSTA25Config(PretrainedConfig):
    model_type = "desta25"

    def __init__(self, 
                 llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
                 encoder_model_id="openai/whisper-large-v3",
                 connector_mode=GROUPWISE_ORTHO_MODE,
                 qformer_num_hidden_layers=2, 
                 prompt_size=64, 
                 use_lora=False,
                 audio_locator="<|AUDIO|>",
                 placeholder_token="<|reserved_special_token_87|>",
                 
                 # Groupwise-orthogonal connector configuration
                 orca_r1_num_groups=8,
                 orca_r1_queries_per_group=8,
                 orca_r1_inter_group_weight=0.1,
                 orca_r1_intra_group_weight=0.01,
                 
                 # Optimization flags
                 use_flash_attention=True,
                 
                 **kwargs):
        for legacy_key in (
            "variational_grouping_enabled",
            "variational_kl_weight",
            "modality_dpo_enabled",
            "modality_dpo_beta",
            "asr_dropout_prob",
            "s1_kl_annealing_enabled",
            "s1_kl_annealing_warmup_steps",
            "s1_kl_annealing_cycle_steps",
            "s1_free_bits",
            "s1_inference_alpha",
        ):
            kwargs.pop(legacy_key, None)
        
        super().__init__(**kwargs)

        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        self.connector_mode = _canonical_connector_mode(connector_mode)
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.prompt_size = prompt_size

        self.audio_locator = audio_locator
        self.placeholder_token = placeholder_token

        self.llm_config = AutoConfig.from_pretrained(self.llm_model_id)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_id)

        self.use_lora = use_lora
        self.use_flash_attention = use_flash_attention

        # Groupwise-orthogonal configuration. Attribute names keep orca_r1_*
        # for checkpoint compatibility with earlier experiment runs.
        self.orca_r1_num_groups = orca_r1_num_groups
        self.orca_r1_queries_per_group = orca_r1_queries_per_group
        self.orca_r1_inter_group_weight = orca_r1_inter_group_weight
        self.orca_r1_intra_group_weight = orca_r1_intra_group_weight

        self.info = "DeSTA2.5 with groupwise-orthogonal Q-Former connector"



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

        if _is_groupwise_ortho_mode(self.config.connector_mode):
            logging.info("Enabling groupwise-orthogonal Q-Former connector")

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
        
        # Prepare inputs for the groupwise connector path.
        prepare_result = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions,
        )
        
        is_groupwise_ortho_mode = _is_groupwise_ortho_mode(self.config.connector_mode)
        
        if is_groupwise_ortho_mode:
            inputs_embeds = prepare_result
            
            # Call LLM 
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
            )
            lm_loss = outputs.loss
            total_loss = lm_loss
            outputs.lm_loss = lm_loss
            
            # Collect groupwise-orthogonal connector losses.
            groupwise_losses = getattr(self.perception, "_groupwise_ortho_losses", None)
            
            # Initialize loss log dict (detached values for trainer logging)
            groupwise_loss_log = {}
            groupwise_total = 0.0
            
            # Add only actual loss terms.
            if outputs.loss is not None and groupwise_losses is not None:
                for name, loss in groupwise_losses.items():
                    if loss is not None and isinstance(loss, torch.Tensor):
                        loss_value = loss.detach().float().item()
                        groupwise_loss_log[name] = loss_value
                        if name.startswith("L_"):
                            if torch.isfinite(loss.detach()).all():
                                total_loss = total_loss + loss
                                groupwise_total += loss_value
                            else:
                                logging.warning("Skipping non-finite groupwise loss %s=%s", name, loss_value)
            
            outputs.loss = total_loss
            try:
                outputs["loss"] = total_loss
            except TypeError:
                pass
            # Attach losses to outputs
            outputs.groupwise_ortho_losses = groupwise_loss_log
            outputs.groupwise_ortho_total_loss = groupwise_total
            self._groupwise_ortho_loss_log = groupwise_loss_log
            self._groupwise_ortho_loss_total = groupwise_total
            # Backward-compatible names for older trainer utilities.
            self._orca_desta_loss_log = groupwise_loss_log
            self._orca_desta_loss_total = groupwise_total
            self._orca_r1_loss_log = groupwise_loss_log
            self._orca_r1_loss_total = groupwise_total
            
            return outputs

        raise NotImplementedError(f"mode {self.config.connector_mode} not implemented")
 

    def _prepare_inputs_for_llm(self, 
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions,
        ):
        """
        Prepare the embeddings input for the LLM.
        Batch_features: list of audio features
        Batch_transcription_ids: list of transcription ids
        Batch_start_positions: list of start positions
        
        Returns:
            inputs_embeds tensor
        """

        N_audio = len(batch_start_positions)
        device = next(self.llm_model.parameters()).device
        
        # Handle empty audio case
        if N_audio == 0:
            self._audio_token_spans = []
            embeds = self.llm_model.model.embed_tokens(input_ids)
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

        # Forward speech encoder and connector.
        perception_output = self.perception(
            input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list,
        )
        
        batch_audio_features, batch_audio_feature_lengths = perception_output
        if _is_groupwise_ortho_mode(self.config.connector_mode):
            self._groupwise_ortho_audio_tokens = batch_audio_features
            self._orca_desta_audio_tokens = batch_audio_features

        assert len(batch_start_positions) == len(batch_transcription_ids) == batch_audio_features.size(0) == len(batch_audio_feature_lengths), "batch_start_positions, batch_transcription_ids, audio_features, speech_feature_lengths must have the same length."


        # [---- Other text embeddings ----][---- placeholder embeddings ----][---- Other text embeddings ----]
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids)
        
        # Track transcription positions for alignment loss
        transcription_positions = []
        audio_token_spans = []
        
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
            audio_token_spans.append((text_batch_idx, audio_start_position, audio_start_position + speech_feature_length))

            # # concat the speech features and transcription embeddings
            audio_embeddings = torch.cat([audio_features, transcription_embeddings], dim=0)
            audio_embeddings = torch.nan_to_num(audio_embeddings, nan=0.0, posinf=5.0, neginf=-5.0).clamp(min=-5.0, max=5.0)

            assert audio_embeddings.size(0) == (speech_feature_length + trans_len)

            # # replace the input_embeds with the audio features
            # # [---- Other text embeddings ----][---- audio features + transcription embeddings ----][---- Other text embeddings ----]
            
            target_slice = slice(audio_start_position, audio_start_position + audio_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = audio_embeddings
            

            # clean GPU memory
            del audio_features, speech_feature_length, transcription_embeddings, audio_embeddings

        self._audio_token_spans = audio_token_spans

        return inputs_embeds

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



    def _generate_step(self, inputs, pad_token_id, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True, repetition_penalty=None, **kwargs):
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
        
        inputs_embeds = prepare_result

        if do_sample is False:
            top_p = None
            temperature = None
        
        try:
            gen_kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs
            )
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            generated_ids = self.llm_model.generate(**gen_kwargs)
        finally:
            self._audio_token_spans = []

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

        # Clear Whisper's default max_length to avoid warning when max_new_tokens is passed
        if hasattr(self.perception.whisper, 'generation_config'):
            self.perception.whisper.generation_config.max_length = None

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
        repetition_penalty=None,
        **kwargs
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
                if not is_speech and trans is None:
                    all_transcriptions[i] = " "
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            batch_features = batch_features.to(self.device)
            
            audio_token_size = _get_audio_token_size(self.config)
            audio_size_list = [audio_token_size] * len(batch_features)


            # RUN ASR
            if asr_features:
                asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
                whisper_encoder = self.perception.whisper.model.encoder
                asr_features = asr_features.to(
                    device=whisper_encoder.conv1.weight.device,
                    dtype=whisper_encoder.conv1.weight.dtype,
                )

                transcriptions = self.perception.whisper.generate(
                    input_features=asr_features,
                    attention_mask=None,
                    max_new_tokens=128,
                    max_length=None
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
            
            # Use encode() to ensure size matches exactly what will be injected
            transcription_size_list = []
            for text in all_transcriptions:
                 encoded_len = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").size(1)
                 transcription_size_list.append(encoded_len)


            audio_context_list = []
            start_positions_list = []
            for messages in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # Restore explicit wrapping for robustness (matches old working version)
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
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs)

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

        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)

        model = cls(config)
        
        if os.path.isdir(pretrained_model_name_or_path):
            path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors", cache_dir=cache_dir)

        state_dict = load_file(path)
        load_result = model.load_state_dict(state_dict, strict=False)
        model._desta_load_missing_keys = list(load_result.missing_keys)
        model._desta_load_unexpected_keys = list(load_result.unexpected_keys)
        del state_dict

        return model
