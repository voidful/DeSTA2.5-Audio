
import torch
from transformers import Qwen2Config, Qwen2Model

def test_float_position_ids():
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
    )
    model = Qwen2Model(config)
    
    # Create inputs
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    inputs_embeds = model.embed_tokens(input_ids)
    
    # Standard Int position IDs
    pos_ids_int = torch.arange(seq_len).unsqueeze(0).long()
    
    # Float position IDs (simulating 4x slowdown: 0, 0.25, 0.5...)
    pos_ids_float = (torch.arange(seq_len).unsqueeze(0) / 4.0).float()
    
    print(f"Testing Float Position IDs: {pos_ids_float}")
    
    try:
        # Pass inputs_embeds to bypass token embedding lookup (which would require int ids if used there)
        # But Qwen2 doesn't use position_ids for absolute embedding, only RoPE.
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids_float
        )
        print("Success! Qwen2 accepted float position_ids.")
        print("Output shape:", outputs.last_hidden_state.shape)
        return True
    except Exception as e:
        print(f"Failed with error: {e}")
        return False

if __name__ == "__main__":
    test_float_position_ids()
