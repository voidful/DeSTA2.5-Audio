import torch
from types import SimpleNamespace

import desta.models.modeling_desta25 as modeling
from desta.models.modeling_desta25 import (
    DeSTA25Config,
    GroupwiseOrthogonalConnector,
    _get_audio_token_size,
)


def patch_auto_config(monkeypatch):
    def fake_from_pretrained(model_id, *args, **kwargs):
        if "whisper" in model_id:
            return SimpleNamespace(
                d_model=384,
                encoder_attention_heads=6,
                num_hidden_layers=4,
            )
        return SimpleNamespace(hidden_size=128)

    monkeypatch.setattr(modeling.AutoConfig, "from_pretrained", fake_from_pretrained)


def make_config(monkeypatch, **kwargs):
    patch_auto_config(monkeypatch)
    defaults = dict(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="groupwise_ortho",
        qformer_num_hidden_layers=2,
        prompt_size=4,
        use_lora=False,
    )
    defaults.update(kwargs)
    return DeSTA25Config(**defaults)


def test_groupwise_config_alias_and_fields(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="orca_r1",
        orca_r1_num_groups=4,
        orca_r1_queries_per_group=2,
        orca_r1_inter_group_weight=0.2,
        orca_r1_intra_group_weight=0.03,
        modality_dpo_enabled=True,
        s1_inference_alpha=0.5,
    )

    assert config.connector_mode == "groupwise_ortho"
    assert config.orca_r1_num_groups == 4
    assert config.orca_r1_queries_per_group == 2
    assert config.orca_r1_inter_group_weight == 0.2
    assert config.orca_r1_intra_group_weight == 0.03
    assert _get_audio_token_size(config) == 8
    assert not hasattr(config, "modality_dpo_enabled")
    assert not hasattr(config, "s1_inference_alpha")


def test_groupwise_orthogonal_connector_forward(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="groupwise_ortho",
        orca_r1_num_groups=4,
        orca_r1_queries_per_group=2,
    )
    connector = GroupwiseOrthogonalConnector(config)

    batch_size = 2
    seq_len = 100
    d_model = config.encoder_config.d_model
    encoder_hidden_states = [
        torch.randn(batch_size, seq_len, d_model)
        for _ in range(config.encoder_config.num_hidden_layers)
    ]

    global_tokens, losses = connector(encoder_hidden_states)

    assert global_tokens.shape == (
        batch_size,
        config.orca_r1_num_groups * config.orca_r1_queries_per_group,
        config.llm_config.hidden_size,
    )
    assert "L_inter_group" in losses
    assert "L_intra_group" in losses


def test_groupwise_connector_has_no_extra_latent_modules(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="groupwise_ortho",
        orca_r1_num_groups=2,
        orca_r1_queries_per_group=2,
    )
    connector = GroupwiseOrthogonalConnector(config)

    batch_size = 2
    seq_len = 64
    d_model = config.encoder_config.d_model
    encoder_hidden_states = [
        torch.randn(batch_size, seq_len, d_model)
        for _ in range(config.encoder_config.num_hidden_layers)
    ]

    tokens, losses = connector(encoder_hidden_states)

    assert tokens.shape[1] == config.orca_r1_num_groups * config.orca_r1_queries_per_group
    assert "L_kl" not in losses
    assert not hasattr(connector, "mu_proj")
    assert not hasattr(connector, "logvar_proj")


def test_qformer_mode_is_legacy(monkeypatch):
    config = make_config(monkeypatch, connector_mode="qformer_1")

    assert config.connector_mode == "qformer_1"
    try:
        _get_audio_token_size(config)
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("qformer_1 should be treated as a legacy connector mode")
