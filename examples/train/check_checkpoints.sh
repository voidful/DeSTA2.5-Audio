#!/bin/bash

# Script to check checkpoint structure in existing experiment directories
# Run this on the cluster to diagnose checkpoint detection issues

output_root="/work/voidful2nlp/desta/outputs/desta25_qwen3_0.6b_orca"
name="qwen3-0.6b-instruct-orca"

echo "=== Checking checkpoint directories ==="
echo "Output root: $output_root"
echo "Name pattern: *_${name}"
echo ""

# Find all matching directories
echo "All matching experiment directories:"
ls -td ${output_root}/*_${name} 2>/dev/null
echo ""

# Find the latest directory
latest_dir=$(ls -td ${output_root}/*_${name} 2>/dev/null | head -n 1)
echo "Latest directory: $latest_dir"
echo ""

if [ -d "$latest_dir" ]; then
    echo "Contents of latest directory:"
    ls -la "$latest_dir" | head -30
    echo ""
    
    # Check for checkpoint-latest
    if [ -d "$latest_dir/checkpoint-latest" ]; then
        echo "✓ Found checkpoint-latest directory"
    else
        echo "✗ No checkpoint-latest directory found"
    fi
    
    # Check for any checkpoint directories
    echo ""
    echo "All checkpoint directories in latest:"
    ls -d "$latest_dir"/checkpoint-* 2>/dev/null || echo "No checkpoint-* directories found"
    
    # Check for numbered checkpoints
    echo ""
    echo "Numbered checkpoint directories:"
    ls -d "$latest_dir"/checkpoint-[0-9]* 2>/dev/null || echo "No numbered checkpoint directories found"
else
    echo "Latest directory does not exist!"
fi
