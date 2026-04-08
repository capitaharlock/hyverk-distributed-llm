#!/usr/bin/env python3
"""
Node-side inference worker. Called by Rust ws_worker as subprocess.

Modes:
  download: Download assigned layers from coordinator
  forward:  Run forward pass on hidden states
  embed:    Embed token IDs (first node only)
  generate: Run forward + lm_head to get next token (last node only)

Input/output via files to avoid serialization overhead.
"""
import argparse, json, os, sys, time

try:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
    )
    from safetensors.torch import load_file
except ImportError as e:
    print(json.dumps({"error": str(e)})); sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["download","embed","forward","generate"])
    p.add_argument("--model-dir", required=True, help="Local dir for cached layer weights")
    p.add_argument("--coordinator", default="https://hyverk-coordinator.fly.dev")
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=28)
    p.add_argument("--input-file", default="", help="Input hidden states (binary torch tensor)")
    p.add_argument("--output-file", default="", help="Output hidden states")
    p.add_argument("--token-ids", default="", help="Comma-separated token IDs for embed mode")
    args = p.parse_args()

    if args.mode == "download":
        download_layers(args)
    elif args.mode == "embed":
        embed_tokens(args)
    elif args.mode == "forward":
        forward_layers(args)
    elif args.mode == "generate":
        generate_token(args)

def download_layers(args):
    """Download only the safetensors shards needed for our layers"""
    import urllib.request
    os.makedirs(args.model_dir, exist_ok=True)

    # Get model index from coordinator
    url = f"{args.coordinator}/api/v1/model/config"
    resp = json.loads(urllib.request.urlopen(url, timeout=30).read())
    if not resp.get("available"):
        print(json.dumps({"error": "Model not available on coordinator"})); return

    index = resp["index"]
    config = resp["config"]

    # Save config locally
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(args.model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    # Download tokenizer
    tok_url = f"{args.coordinator}/api/v1/model/shard/tokenizer.json"
    tok_path = os.path.join(args.model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Downloading tokenizer...", file=sys.stderr)
        urllib.request.urlretrieve(tok_url, tok_path)

    # Find which shards contain our layers
    wm = index.get("weight_map", {})
    needed_shards = set()
    for key, shard in wm.items():
        for l in range(args.layer_start, args.layer_end):
            if f"model.layers.{l}." in key:
                needed_shards.add(shard); break
        # First node needs embed_tokens
        if args.layer_start == 0 and key == "model.embed_tokens.weight":
            needed_shards.add(shard)
        # Last node needs norm + lm_head
        if args.layer_end >= config.get("num_hidden_layers", 28):
            if key in ("model.norm.weight", "lm_head.weight"):
                needed_shards.add(shard)

    # Download needed shards (with retry for partial downloads)
    for shard in sorted(needed_shards):
        path = os.path.join(args.model_dir, shard)
        if os.path.exists(path):
            # Verify file isn't a partial download (check size > 100MB)
            sz = os.path.getsize(path)
            if sz > 100_000_000:
                print(f"  {shard}: already cached ({sz/1e9:.1f}GB)", file=sys.stderr)
                continue
            else:
                print(f"  {shard}: incomplete ({sz} bytes), re-downloading...", file=sys.stderr)
                os.remove(path)
        shard_url = f"{args.coordinator}/api/v1/model/shard/{shard}"
        print(f"  {shard}: downloading...", file=sys.stderr)
        tmp_path = path + ".tmp"
        urllib.request.urlretrieve(shard_url, tmp_path)
        os.rename(tmp_path, path)
        sz = os.path.getsize(path) / 1e9
        print(f"  {shard}: {sz:.1f}GB", file=sys.stderr)

    print(json.dumps({
        "status": "ok",
        "shards": list(needed_shards),
        "layers": f"{args.layer_start}-{args.layer_end}",
    }))

def load_model_layers(args):
    """Load assigned layers from local cache"""
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"

    # Load weights
    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    use_quantize = (device.type == "cpu")
    if use_quantize:
        print("Using dynamic int8 quantization for CPU inference", file=sys.stderr)

    weights = {}
    loaded_shards = set()
    for key, shard in index["weight_map"].items():
        should_load = False
        for l in range(args.layer_start, args.layer_end):
            if f"model.layers.{l}." in key: should_load = True; break
        if args.layer_start == 0 and key == "model.embed_tokens.weight": should_load = True
        if args.layer_end >= config.num_hidden_layers:
            if key in ("model.norm.weight", "lm_head.weight"): should_load = True

        if should_load and shard not in loaded_shards:
            path = os.path.join(args.model_dir, shard)
            if os.path.exists(path):
                tensors = load_file(path, device="cpu")
                for k, v in tensors.items():
                    weights[k] = v.to(device=device, dtype=torch.float16)
                del tensors
                loaded_shards.add(shard)

    # Build layers
    rotary = Qwen2RotaryEmbedding(config).to(device)
    layers = []
    for i in range(args.layer_start, args.layer_end):
        layer = Qwen2DecoderLayer(config, i).to(dtype=torch.float16, device=device)
        prefix = f"model.layers.{i}."
        state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        if state:
            layer.load_state_dict(state, strict=False)
        if use_quantize:
            layer = layer.float()  # int8 quantization requires float32
            layer = torch.quantization.quantize_dynamic(layer, {torch.nn.Linear}, dtype=torch.qint8)
        layers.append(layer)

    # Embedding (first node)
    embed = None
    if "model.embed_tokens.weight" in weights:
        embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        embed.weight = torch.nn.Parameter(weights["model.embed_tokens.weight"], requires_grad=False)
        embed = embed.to(device)

    # Norm + lm_head (last node)
    norm = None
    lm_head = None
    if "model.norm.weight" in weights:
        norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(weights["model.norm.weight"], requires_grad=False)
        norm = norm.to(device)
    if "lm_head.weight" in weights:
        lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.weight = torch.nn.Parameter(weights["lm_head.weight"], requires_grad=False)
        lm_head = lm_head.to(device)

    return config, device, rotary, layers, embed, norm, lm_head

def embed_tokens(args):
    """First node: embed token IDs, forward through layers, save hidden states"""
    config, device, rotary, layers, embed, norm, lm_head = load_model_layers(args)

    token_ids = [int(x) for x in args.token_ids.split(",") if x]
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    hidden = embed(ids)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            hidden = layer(hidden, position_embeddings=position_embeddings, use_cache=False)

    # Save hidden states
    torch.save(hidden.cpu().half(), args.output_file)
    print(json.dumps({"shape": list(hidden.shape), "size": os.path.getsize(args.output_file)}))

def forward_layers(args):
    """Middle node: load hidden states, forward through layers, save result"""
    config, device, rotary, layers, embed, norm, lm_head = load_model_layers(args)

    hidden = torch.load(args.input_file, map_location=device, weights_only=True)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            hidden = layer(hidden, position_embeddings=position_embeddings, use_cache=False)

    torch.save(hidden.cpu().half(), args.output_file)
    print(json.dumps({"shape": list(hidden.shape), "size": os.path.getsize(args.output_file)}))

def generate_token(args):
    """Last node: forward through layers + norm + lm_head → token ID"""
    config, device, rotary, layers, embed, norm, lm_head = load_model_layers(args)

    hidden = torch.load(args.input_file, map_location=device, weights_only=True)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            hidden = layer(hidden, position_embeddings=position_embeddings, use_cache=False)
        if norm: hidden = norm(hidden)
        if lm_head:
            logits = lm_head.float()(hidden.float())
            token_id = torch.argmax(logits[0, -1]).item()
        else:
            token_id = 0

    print(json.dumps({"token_id": token_id}))

if __name__ == "__main__":
    main()
