import torch
from types import SimpleNamespace

import desta.models.modeling_desta25 as modeling
from desta.models.modeling_desta25 import (
    DeSTA25AudioModel,
    DeSTA25Config,
    GroupwiseOrthogonalConnector,
    QformerConnector,
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
        connector_mode="qformer_1",
        qformer_num_hidden_layers=2,
        prompt_size=4,
        use_lora=False,
    )
    defaults.update(kwargs)
    return DeSTA25Config(**defaults)


def test_qformer_connector_forward(monkeypatch):
    config = make_config(monkeypatch)
    connector = QformerConnector(config)

    batch_size = 2
    seq_len = 100
    d_model = config.encoder_config.d_model
    encoder_hidden_states = [
        torch.randn(batch_size, seq_len, d_model)
        for _ in range(config.encoder_config.num_hidden_layers)
    ]

    output = connector(encoder_hidden_states)

    assert output.shape == (
        batch_size,
        config.prompt_size,
        config.llm_config.hidden_size,
    )


def test_orca_desta_config_alias_and_fields(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="orca_r1",
        orca_r1_num_groups=4,
        orca_r1_queries_per_group=2,
        orca_r1_inter_group_weight=0.2,
        orca_r1_intra_group_weight=0.03,
        variational_grouping_enabled=True,
        modality_dpo_enabled=True,
        asr_dropout_prob=0.2,
    )

    assert config.connector_mode == "orca_desta"
    assert config.orca_r1_num_groups == 4
    assert config.orca_r1_queries_per_group == 2
    assert config.orca_r1_inter_group_weight == 0.2
    assert config.orca_r1_intra_group_weight == 0.03
    assert config.variational_grouping_enabled is True
    assert config.modality_dpo_enabled is True
    assert config.asr_dropout_prob == 0.2


def test_groupwise_orthogonal_connector_forward(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="orca_desta",
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


def test_stochastic_perturbation_adds_kl_loss(monkeypatch):
    config = make_config(
        monkeypatch,
        connector_mode="orca_desta",
        orca_r1_num_groups=2,
        orca_r1_queries_per_group=2,
        variational_grouping_enabled=True,
        variational_kl_weight=0.01,
        s1_kl_annealing_enabled=True,
        s1_kl_annealing_warmup_steps=100,
        s1_free_bits=0.1,
        s1_inference_alpha=0.0,
    )
    connector = GroupwiseOrthogonalConnector(config)
    connector.train()

    batch_size = 2
    seq_len = 64
    d_model = config.encoder_config.d_model
    encoder_hidden_states = [
        torch.randn(batch_size, seq_len, d_model)
        for _ in range(config.encoder_config.num_hidden_layers)
    ]

    z, losses = connector(encoder_hidden_states, global_step=50)

    assert z.shape[1] == config.orca_r1_num_groups * config.orca_r1_queries_per_group
    assert hasattr(connector, "mu_proj")
    assert hasattr(connector, "logvar_proj")
    assert "L_kl" in losses
    assert "kl_weight_effective" in losses


def test_target_log_probs_use_causal_shift():
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, 2, 3, -100]])
    logits[0, 0, 2] = 5.0
    logits[0, 1, 3] = 5.0

    log_probs = DeSTA25AudioModel._get_target_log_probs(None, logits, labels)
    expected = (
        logits[:, :-1, :]
        .log_softmax(-1)
        .gather(2, labels[:, 1:].clamp_min(0).unsqueeze(2))
        .squeeze(2)
    )
    mask = labels[:, 1:] != -100
    expected = (expected * mask).sum(-1) / mask.sum(-1)

    assert torch.allclose(log_probs, expected)
