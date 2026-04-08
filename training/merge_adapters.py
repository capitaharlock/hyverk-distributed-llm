#!/usr/bin/env python3
"""
Merge layer-sharded LoRA adapters into one complete adapter.

Called after a training round completes:
  python3 merge_adapters.py \
    --adapter-dir /tmp/round_adapters/ \
    --output /data/hyverk-v0.1-r1.safetensors

Each input file contains LoRA weights for 2 layers (e.g., layers 0-2, 2-4, etc.).
The merged output contains all layer adapters in one file.

Optionally averages multiple adapters for the same layers (FedAvg).
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Merge layer-sharded LoRA adapters")
    parser.add_argument("--adapter-dir", required=True, help="Directory with per-shard adapter files")
    parser.add_argument("--output", required=True, help="Output merged adapter path")
    parser.add_argument("--round-info", default="", help="JSON with round metadata")
    args = parser.parse_args()

    try:
        from safetensors.torch import load_file, save_file
        import torch
    except ImportError as e:
        print(json.dumps({"error": f"Missing dependency: {e}"}))
        sys.exit(1)

    # Collect all adapter files
    adapter_files = sorted([
        os.path.join(args.adapter_dir, f)
        for f in os.listdir(args.adapter_dir)
        if f.endswith('.safetensors')
    ])

    if not adapter_files:
        print(json.dumps({"error": "No adapter files found"}))
        sys.exit(1)

    print(f"Merging {len(adapter_files)} adapter files...", file=sys.stderr)

    # Load and merge all tensors
    merged = {}
    tensor_counts = {}  # For FedAvg: track how many adapters contribute to each key

    for path in adapter_files:
        try:
            tensors = load_file(path, device="cpu")
            for key, tensor in tensors.items():
                if key in merged:
                    # FedAvg: accumulate for averaging
                    merged[key] = merged[key] + tensor
                    tensor_counts[key] = tensor_counts.get(key, 1) + 1
                else:
                    merged[key] = tensor.clone()
                    tensor_counts[key] = 1
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)

    # Average any keys that had multiple contributions
    for key in merged:
        if tensor_counts.get(key, 1) > 1:
            merged[key] = merged[key] / tensor_counts[key]

    if not merged:
        print(json.dumps({"error": "No tensors to merge"}))
        sys.exit(1)

    # Save merged adapter
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Ensure contiguous tensors for safetensors
    save_dict = {k: v.contiguous() for k, v in merged.items()}
    save_file(save_dict, args.output)

    size = os.path.getsize(args.output)
    print(f"Merged {len(merged)} tensors from {len(adapter_files)} files", file=sys.stderr)

    result = {
        "merged_tensors": len(merged),
        "adapter_files": len(adapter_files),
        "output_path": args.output,
        "output_size": size,
        "keys": list(merged.keys())[:10],  # first 10 for preview
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
