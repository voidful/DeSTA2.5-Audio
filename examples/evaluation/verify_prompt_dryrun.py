
import sys
import os
import unittest.mock
from unittest.mock import MagicMock
import json

LOG_FILE = "debug_trace.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log("SCRIPT STARTING")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    log(f"Added path: {current_dir}")

    log("Importing sakura_eval...")
    try:
        import sakura_eval
        log("Import success")
    except ImportError as e:
        log(f"Import failed: {e}")
        raise

    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.text = "Mocked Response"
    mock_model.generate.return_value = mock_output
    
    dummy_item = {
        "instruction": "What animal is making this sound?",
        "id": "test_id_123"
    }
    
    log("Patching write_wav...")
    with unittest.mock.patch('sakura_eval.write_wav_from_dataset_item') as mock_write:
        log("Calling run_desta_on_item...")
        parser = sakura_eval.run_desta_on_item(mock_model, dummy_item, hop_prefix="", wav_path="dummy.wav")
        
        call_args = mock_model.generate.call_args
        if call_args:
             kwargs = call_args.kwargs
             messages = kwargs.get('messages')
             log("--- GENERATED MESSAGES ---")
             log(json.dumps(messages, indent=2))
        else:
             log("Error: model.generate was not called.")

    log("SCRIPT FINISHED")

except Exception as e:
    log(f"CRITICAL EXCEPTION: {e}")
    import traceback
    with open(LOG_FILE, "a") as f:
        traceback.print_exc(file=f)
