#!/bin/bash
# upload_to_hf.sh - Upload trained model to HuggingFace Hub
# Usage: ./upload_to_hf.sh <checkpoint_dir> <model_name>

set -e

CHECKPOINT_DIR=$1
MODEL_NAME=$2
HF_USERNAME="voidful"

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <checkpoint_dir> <model_name>"
    exit 1
fi

# Find the best/latest checkpoint
if [ -d "$CHECKPOINT_DIR/checkpoint-best" ]; then
    CKPT_PATH="$CHECKPOINT_DIR/checkpoint-best"
elif [ -d "$CHECKPOINT_DIR/checkpoint-latest" ]; then
    CKPT_PATH="$CHECKPOINT_DIR/checkpoint-latest"
else
    # Find the highest numbered checkpoint
    CKPT_PATH=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -n 1)
fi

if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "=========================================="
echo "Uploading model to HuggingFace Hub"
echo "Checkpoint: $CKPT_PATH"
echo "Repo: ${HF_USERNAME}/${MODEL_NAME}"
echo "=========================================="

# Upload using huggingface_hub
python -c "
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = '${HF_USERNAME}/${MODEL_NAME}'

# Create repo if not exists
try:
    api.create_repo(repo_id, exist_ok=True, private=True)
except Exception as e:
    print(f'Repo creation note: {e}')

# Upload the checkpoint folder
api.upload_folder(
    folder_path='${CKPT_PATH}',
    repo_id=repo_id,
    repo_type='model',
    commit_message='Upload trained Struct-ORCA model'
)
print(f'Successfully uploaded to https://huggingface.co/{repo_id}')
"

echo "Upload complete!"
