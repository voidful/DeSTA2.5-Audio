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


# === OCAR-DeSTA Tests ===

@pytest.fixture
def ocar_config():
    """Config with OCAR enabled for testing."""
    return DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="ocar_hybrid",
        qformer_num_hidden_layers=2,
        prompt_size=4,
        use_lora=False,
        ocar_enabled=True,
        ocar_global_num_tokens=4,
        ocar_local_downsample=4,
        ocar_local_kernel_size=7,
        ocar_gate_init=0.1,
    )


def test_ocar_config_fields():
    """Test that OCAR config fields are properly initialized."""
    config = DeSTA25Config(
        ocar_enabled=True,
        ocar_global_num_tokens=8,
        ocar_gate_init=0.2
    )
    assert config.ocar_enabled is True
    assert config.ocar_global_num_tokens == 8
    assert config.ocar_gate_init == 0.2
    # Check defaults
    assert config.ocar_local_downsample == 4
    assert config.ocar_ortho_weight_global == 0.01


def test_ocar_hybrid_connector_initialization(ocar_config):
    """Test OCARHybridConnector initialization."""
    from desta.models.modeling_desta25 import OCARHybridConnector
    connector = OCARHybridConnector(ocar_config)
    
    assert isinstance(connector, torch.nn.Module)
    assert len(connector.global_queries) == 4  # tiny has 4 target layers
    assert connector.global_qformer is not None
    assert connector.local_conv is not None


def test_ocar_hybrid_connector_forward(ocar_config):
    """Test OCARHybridConnector forward pass."""
    from desta.models.modeling_desta25 import OCARHybridConnector
    connector = OCARHybridConnector(ocar_config)
    
    batch_size = 2
    seq_len = 100
    d_encoder = 384  # whisper-tiny d_model
    
    # Create mock hidden states for all layers (tiny has 4 layers)
    encoder_hidden_states = [torch.randn(batch_size, seq_len, d_encoder) for _ in range(4)]
    
    global_tokens, local_tokens = connector(encoder_hidden_states)
    
    # Check global tokens shape: [B, K_global, d_llm]
    assert global_tokens.shape == (batch_size, ocar_config.ocar_global_num_tokens, ocar_config.llm_config.hidden_size)
    
    # Check local tokens shape: [B, T_local, d_llm]
    # T_local = ceil((seq_len + padding) / stride)
    assert local_tokens.shape[0] == batch_size
    assert local_tokens.shape[2] == ocar_config.llm_config.hidden_size
    assert local_tokens.shape[1] > 0  # Some temporal reduction happened


def test_ocar_gated_cross_attention():
    """Test OCARGatedCrossAttention forward pass."""
    from desta.models.modeling_desta25 import OCARGatedCrossAttention
    
    hidden_size = 128
    num_heads = 4
    batch_size = 2
    seq_len = 10
    audio_len = 5
    
    cross_attn = OCARGatedCrossAttention(
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


def test_ocar_backward_compatibility():
    """Test that OCAR-disabled config works like original."""
    config = DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-tiny",
        connector_mode="qformer_1",
        ocar_enabled=False
    )
    
    # Should still work with QformerConnector
    connector = QformerConnector(config)
    assert isinstance(connector, torch.nn.Module)
    assert config.ocar_enabled is False
