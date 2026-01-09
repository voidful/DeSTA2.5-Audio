import pytest
import torch
from desta.models.modeling_desta25 import DeSTA25Config, QformerConnector, WhisperPerception

@pytest.fixture
def config():
    return DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny", # Use tiny for faster tests
        connector_mode="qformer_1",
        qformer_num_hidden_layers=2,
        prompt_size=4,
        use_lora=False
    )

def test_qformer_connector_initialization(config):
    connector = QformerConnector(config)
    assert isinstance(connector, torch.nn.Module)
    assert len(connector.layer_prompts) == 4 # tiny has 4 layers

def test_qformer_connector_forward(config):
    connector = QformerConnector(config)
    # Mock encoder hidden states: list of tensors (batch, seq, dim)
    # tiny d_model = 384
    batch_size = 2
    seq_len = 100
    d_model = 384
    
    # Create mock hidden states for all layers (tiny has 4 layers)
    encoder_hidden_states = [torch.randn(batch_size, seq_len, d_model) for _ in range(4)]
    
    output = connector(encoder_hidden_states)
    
    # Output shape should be (batch, prompt_size, llm_hidden_size)
    # Llama 3.1 8B hidden size is 4096
    assert output.shape == (batch_size, config.prompt_size, config.llm_config.hidden_size)

def test_whisper_perception_initialization(config):
    perception = WhisperPerception(config)
    assert isinstance(perception, torch.nn.Module)
    assert perception.whisper is not None

# Note: Testing WhisperPerception forward requires actual audio features or careful mocking of Whisper internals, 
# which might be too heavy for a unit test. We'll stick to initialization for now.


# === ORCA-DeSTA Tests ===

@pytest.fixture
def orca_config():
    """Config with ORCA enabled for testing."""
    return DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="orca_hybrid",
        qformer_num_hidden_layers=2,
        prompt_size=4,
        use_lora=False,
        orca_enabled=True,
        orca_global_num_tokens=4,
        orca_local_downsample=4,
        orca_local_kernel_size=7,
        orca_gate_init=0.1,
    )


def test_orca_config_fields():
    """Test that ORCA config fields are properly initialized."""
    config = DeSTA25Config(
        orca_enabled=True,
        orca_global_num_tokens=8,
        orca_gate_init=0.2
    )
    assert config.orca_enabled is True
    assert config.orca_global_num_tokens == 8
    assert config.orca_gate_init == 0.2
    # Check defaults
    assert config.orca_local_downsample == 4
    assert config.orca_ortho_weight_global == 0.01


def test_orca_hybrid_connector_initialization(orca_config):
    """Test ORCAHybridConnector initialization."""
    from desta.models.modeling_desta25 import ORCAHybridConnector
    connector = ORCAHybridConnector(orca_config)
    
    assert isinstance(connector, torch.nn.Module)
    assert len(connector.global_queries) == 4  # tiny has 4 target layers
    assert connector.global_qformer is not None


def test_orca_hybrid_connector_forward(orca_config):
    """Test ORCAHybridConnector forward pass."""
    from desta.models.modeling_desta25 import ORCAHybridConnector
    connector = ORCAHybridConnector(orca_config)
    
    batch_size = 2
    seq_len = 100
    d_encoder = 384  # whisper-tiny d_model
    
    # Create mock hidden states for all layers (tiny has 4 layers)
    encoder_hidden_states = [torch.randn(batch_size, seq_len, d_encoder) for _ in range(4)]
    
    global_tokens = connector(encoder_hidden_states)
    
    # Check global tokens shape: [B, K_global, d_llm]
    assert global_tokens.shape == (batch_size, orca_config.orca_global_num_tokens, orca_config.llm_config.hidden_size)


def test_orca_gated_cross_attention():
    """Test ORCAGatedCrossAttention forward pass."""
    from desta.models.modeling_desta25 import ORCAGatedCrossAttention
    
    hidden_size = 128
    num_heads = 4
    batch_size = 2
    seq_len = 10
    audio_len = 5
    
    cross_attn = ORCAGatedCrossAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        gate_init=0.1
    )
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    audio_local = torch.randn(batch_size, audio_len, hidden_size)
    
    output = cross_attn(hidden_states, audio_local)
    
    # Output should have same shape as input
    assert output.shape == hidden_states.shape
    
    # Test with no audio (should return input unchanged)
    output_no_audio = cross_attn(hidden_states, None)
    assert torch.allclose(output_no_audio, hidden_states)


def test_orca_backward_compatibility():
    """Test that ORCA-disabled config works like original."""
    config = DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="qformer_1",
        orca_enabled=False
    )
    
    # Should still work with QformerConnector
    connector = QformerConnector(config)
    assert isinstance(connector, torch.nn.Module)
    assert config.orca_enabled is False


def test_orca_mixed_precision(orca_config):
    """Test that ORCAHybridConnector handles float32 inputs when module is in bfloat16."""
    from desta.models.modeling_desta25 import ORCAHybridConnector
    connector = ORCAHybridConnector(orca_config)
    
    # Move connector to bfloat16
    connector = connector.to(dtype=torch.bfloat16)
    
    batch_size = 2
    seq_len = 20
    d_encoder = 384
    
    # Inputs remain in float32 (standard behavior from dataloader)
    encoder_hidden_states = [torch.randn(batch_size, seq_len, d_encoder, dtype=torch.float32) for _ in range(4)]
    
    # This should not raise RuntimeError: expected mat1 and mat2 to have the same dtype
    try:
        global_tokens = connector(encoder_hidden_states)
    except RuntimeError as e:
        pytest.fail(f"ORCAHybridConnector failed with mixed precision: {e}")
    
    assert global_tokens.dtype == torch.bfloat16


# === Struct-ORCA Tests ===

@pytest.fixture
def struct_orca_config():
    """Config with Struct-ORCA enabled for testing."""
    return DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="struct_orca",
        qformer_num_hidden_layers=2,
        prompt_size=64,
        use_lora=False,
        struct_orca_num_groups=8,
        struct_orca_queries_per_group=8,
        struct_orca_inter_group_weight=0.1,
        struct_orca_intra_group_weight=0.01,
        struct_orca_iv_weight=0.1,
        struct_orca_acd_alpha=0.5,
    )


def test_struct_orca_config_fields():
    """Test that Struct-ORCA config fields are properly initialized."""
    config = DeSTA25Config(
        connector_mode="struct_orca",
        struct_orca_num_groups=8,
        struct_orca_queries_per_group=8,
        struct_orca_inter_group_weight=0.1,
    )
    assert config.struct_orca_num_groups == 8
    assert config.struct_orca_queries_per_group == 8
    assert config.struct_orca_inter_group_weight == 0.1
    # Check defaults
    assert config.struct_orca_intra_group_weight == 0.01
    assert config.struct_orca_iv_weight == 0.1


def test_groupwise_orthogonal_connector_initialization(struct_orca_config):
    """Test GroupwiseOrthogonalConnector initialization."""
    from desta.models.modeling_desta25 import GroupwiseOrthogonalConnector
    connector = GroupwiseOrthogonalConnector(struct_orca_config)
    
    assert isinstance(connector, torch.nn.Module)
    assert connector.num_groups == 8
    assert connector.queries_per_group == 8
    assert connector.total_queries == 64
    assert len(connector.group_queries) == 8  # 8 groups


def test_groupwise_orthogonal_connector_forward(struct_orca_config):
    """Test GroupwiseOrthogonalConnector forward pass."""
    from desta.models.modeling_desta25 import GroupwiseOrthogonalConnector
    connector = GroupwiseOrthogonalConnector(struct_orca_config)
    
    batch_size = 2
    seq_len = 100
    d_encoder = 384  # whisper-tiny d_model
    
    # Create mock hidden states for all layers (tiny has 4 target layers)
    encoder_hidden_states = [torch.randn(batch_size, seq_len, d_encoder) for _ in range(4)]
    
    global_tokens, group_losses = connector(encoder_hidden_states)
    
    # Check output shapes
    total_tokens = struct_orca_config.struct_orca_num_groups * struct_orca_config.struct_orca_queries_per_group
    assert global_tokens.shape == (batch_size, total_tokens, struct_orca_config.llm_config.hidden_size)
    
    # Check group losses are computed
    assert "L_inter_group" in group_losses
    assert "L_intra_group" in group_losses


def test_text_content_discriminator():
    """Test TextContentDiscriminator for IV-GD."""
    from desta.models.modeling_desta25 import TextContentDiscriminator
    
    hidden_size = 128
    num_groups = 8
    vocab_size = 1000
    batch_size = 4
    
    discriminator = TextContentDiscriminator(
        hidden_size=hidden_size,
        num_groups=num_groups,
        vocab_size=vocab_size
    )
    
    # Input: group tokens
    group_tokens = torch.randn(batch_size, num_groups * 8, hidden_size)
    transcription_ids = torch.randint(0, vocab_size, (batch_size, 20))
    
    output = discriminator(group_tokens, transcription_ids)
    
    assert "loss" in output
    assert output["loss"].requires_grad


def test_gradient_reversal_layer():
    """Test GradientReversalLayer for adversarial training."""
    from desta.models.modeling_desta25 import GradientReversalLayer
    
    grl = GradientReversalLayer(lambda_=1.0)
    
    x = torch.randn(4, 10, requires_grad=True)
    y = grl(x)
    
    # Forward: identity
    assert torch.allclose(y, x)
    
    # Backward: negated gradients
    loss = y.sum()
    loss.backward()
    
    # Gradients should be negated
    assert torch.allclose(x.grad, -torch.ones_like(x))

