import pytest
import torch
from omegaconf import OmegaConf
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
