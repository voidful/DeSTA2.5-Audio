#!/usr/bin/env python3
"""
Test script to verify that the layer selection alignment fix works correctly.
This script tests:
1. Loading a checkpoint with 32 layers (all layers)
2. Automatic detection and reconfiguration
3. Successful model initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desta.models.modeling_desta25 import DeSTA25AudioModel
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_checkpoint_loading():
    """Test loading a checkpoint with automatic layer detection."""
    
    print("=" * 80)
    print("Testing Checkpoint Loading with Automatic Layer Detection")
    print("=" * 80)
    
    # This checkpoint was trained with all 32 layers
    DESTA_MODEL_ID = "voidful/QAQ_0.6b_abl_global32"
    
    print(f"\n1. Loading model from: {DESTA_MODEL_ID}")
    print("   Expected: Automatic detection of 32-layer configuration\n")
    
    try:
        desta_model = DeSTA25AudioModel.from_pretrained(DESTA_MODEL_ID)
        
        print("\n✓ Model loaded successfully!")
        
        # Verify configuration
        num_layers = len(desta_model.perception.connector.target_layer_ids)
        use_all_layers = desta_model.config.orca_use_all_layers
        
        print(f"\n2. Model Configuration:")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Use all layers: {use_all_layers}")
        print(f"   - Target layer IDs: {desta_model.perception.connector.target_layer_ids[:5]}..." 
              if num_layers > 5 else f"   - Target layer IDs: {desta_model.perception.connector.target_layer_ids}")
        
        # Verify shapes
        global_weights_shape = desta_model.perception.connector.global_layer_weights.shape
        local_weights_shape = desta_model.perception.connector.local_layer_weights.shape
        
        print(f"\n3. Parameter Shapes:")
        print(f"   - global_layer_weights: {global_weights_shape}")
        print(f"   - local_layer_weights: {local_weights_shape}")
        
        # Check if shapes match expected
        expected_num_layers = 32  # For whisper-large-v3
        if global_weights_shape[1] == expected_num_layers and local_weights_shape[0] == expected_num_layers:
            print("\n✓ All checks passed! Model is correctly configured.")
            return True
        else:
            print(f"\n✗ Shape mismatch! Expected {expected_num_layers} layers")
            return False
            
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)
