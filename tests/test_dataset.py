import pytest
import torch
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, AutoFeatureExtractor
from desta.trainer.data.simple_dataset import BaseCollateFn

@pytest.fixture
def tokenizer():
    # Use a small tokenizer for testing
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok.padding_side = "left"
    return tok

@pytest.fixture
def processor():
    # Use a small processor
    return AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

@pytest.fixture
def data_cfg():
    return DictConfig({"max_seq_length": 128})

def test_collate_fn_initialization(data_cfg, tokenizer, processor):
    """Test that BaseCollateFn can be initialized properly."""
    collate_fn = BaseCollateFn(data_cfg, tokenizer, processor)
    
    assert collate_fn.tokenizer == tokenizer
    assert collate_fn.processor == processor
    assert collate_fn.max_seq_length == 128

# Note: Full collate_fn testing with actual audio files would require:
# 1. Mock audio files or test fixtures
# 2. Fixing numpy compatibility issues in AudioSegment
# For now, we test initialization which validates the basic structure.
