#!/usr/bin/env python3
"""
End-to-end distributed inference test for hyverk.com.

Simulates the 2-node pipeline:
  Node 1 (first_layers): embed tokens → run layers 0..split-1
  Node 2 (last_layers):  run layers split..27 → norm → lm_head → sample

In distributed mode (--distributed), Node 1 POSTs its hidden states
to the Mac server, which runs Node 2. Both nodes are timed independently.

Usage (single-machine reference, Mac MPS):
  python3 inference/test_inference_e2e.py \
      --model-dir ~/.hyverk/qwen2.5-7b/inference_layers_0_28 \
      --device mps --split 14 --max-tokens 120

Usage (distributed: Mac = last node on :18100, Windows = first node):
  python3 inference/test_inference_e2e.py \
      --model-dir ~/.hyverk/qwen2.5-7b/inference_layers_0_28 \
      --device mps --split 14 --max-tokens 120 \
      --distributed --mac-url http://127.0.0.1:18100
"""
import argparse, json, os, sys, time, struct
import urllib.request

VENV_PYTHON = os.path.join(os.path.dirname(__file__), "..", ".venv", "lib")
sys.path.insert(0, VENV_PYTHON)

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from safetensors.torch import load_file

PROMPT = (
    "Write a Python function that implements a Least Recently Used (LRU) cache "
    "using an OrderedDict. The function should support get(key) and put(key, value) "
    "operations with O(1) time complexity. Include a capacity parameter and evict "
    "the least recently used item when full. Add clear docstrings and type hints."
)

# ── model helpers ─────────────────────────────────────────────────────────────

def load_shard_weights(model_dir: str, device: str):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    weights = {}
    for shard in shards:
        path = os.path.join(model_dir, shard)
        print(f"  loading {shard} ...", file=sys.stderr)
        w = load_file(path, device="cpu")
        weights.update(w)
    return weights

def build_model(model_dir: str, layer_start: int, layer_end: int, device: str):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = Qwen2Config(**{k: v for k, v in cfg_dict.items()
                         if k in Qwen2Config.__init__.__code__.co_varnames})
    cfg.num_hidden_layers = layer_end - layer_start
    model = Qwen2ForCausalLM(cfg)
    return model

# ── single-machine inference ──────────────────────────────────────────────────

def run_single(args):
    device = args.device
    split  = args.split
    model_dir = os.path.expanduser(args.model_dir)
    tok_path  = os.path.join(model_dir, "tokenizer.json")

    print(f"Device: {device} | split: {split} | max_tokens: {args.max_tokens}", file=sys.stderr)
    print(f"Prompt ({len(PROMPT)} chars):\n  {PROMPT[:80]}...\n", file=sys.stderr)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}", file=sys.stderr)

    # Load full model weights
    print("Loading weights...", file=sys.stderr)
    t0 = time.perf_counter()
    weights = load_shard_weights(model_dir, device)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = Qwen2Config(**{k: v for k, v in cfg_dict.items()
                         if hasattr(Qwen2Config, k)})

    model = Qwen2ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    model = model.to(torch.float16).to(device).eval()
    del weights
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f}ms", file=sys.stderr)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids.to(device), use_cache=False)
    print("Warmup done", file=sys.stderr)

    # Prefill
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids.to(device), use_cache=True)
    past = out.past_key_values
    logits = out.logits
    t_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000

    next_token = logits[0, -1].argmax().item()
    generated = [next_token]
    t_first_token_ms = (time.perf_counter() - t_prefill_start) * 1000

    # Decode loop
    decode_times = []
    for _ in range(args.max_tokens - 1):
        t_step = time.perf_counter()
        cur = torch.tensor([[next_token]], device=device)
        with torch.no_grad():
            out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax().item()
        generated.append(next_token)
        decode_times.append((time.perf_counter() - t_step) * 1000)
        if next_token == tokenizer.eos_token_id:
            break

    total_gen = len(generated)
    total_decode_ms = sum(decode_times)
    tps = total_gen / (total_decode_ms / 1000) if decode_times else 0
    p50 = sorted(decode_times)[len(decode_times)//2] if decode_times else 0
    p95 = sorted(decode_times)[int(len(decode_times)*0.95)] if decode_times else 0

    # Decode text
    text = tokenizer.decode(generated, skip_special_tokens=True)

    result = {
        "mode": "single_machine",
        "device": device,
        "split": split,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": total_gen,
        "prefill_ms": round(t_prefill_ms, 1),
        "time_to_first_token_ms": round(t_first_token_ms, 1),
        "decode_p50_ms": round(p50, 1),
        "decode_p95_ms": round(p95, 1),
        "decode_total_ms": round(total_decode_ms, 1),
        "tokens_per_second": round(tps, 1),
    }
    print("\n" + "="*60, file=sys.stderr)
    print(json.dumps(result, indent=2))
    print("="*60, file=sys.stderr)
    print("\nGENERATED OUTPUT:", file=sys.stderr)
    print(text)
    return result


# ── distributed inference (Mac as last node, server on :18100) ────────────────

def post_binary(url, tensor, mode, request_id, temperature=0.0):
    shape = list(tensor.shape)
    data = tensor.to(torch.float16).cpu().numpy().tobytes()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("X-Mode", mode)
    req.add_header("X-Request-Id", request_id)
    req.add_header("X-Shape", json.dumps(shape))
    req.add_header("X-Temperature", str(temperature))
    r = urllib.request.urlopen(req, timeout=60)
    if mode == "generate":
        return json.loads(r.read())
    shape_out = json.loads(r.headers["X-Shape"])
    buf = r.read()
    import numpy as np
    arr = np.frombuffer(buf, dtype=np.float16).reshape(shape_out)
    return torch.from_numpy(arr.copy())

def post_json(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=60)
    shape_out = json.loads(r.headers["X-Shape"])
    import numpy as np
    arr = np.frombuffer(r.read(), dtype=np.float16).reshape(shape_out)
    return torch.from_numpy(arr.copy())

def run_distributed(args):
    """
    Node 1 = this script (layers 0..split-1, runs locally or on Windows).
    Node 2 = Mac server at --mac-url (layers split..27 + lm_head).
    """
    device    = args.device
    split     = args.split
    mac_url   = args.mac_url
    model_dir = os.path.expanduser(args.model_dir)

    print(f"DISTRIBUTED: Node1={device} layers 0-{split} | Node2={mac_url} layers {split}-28", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}", file=sys.stderr)

    # Load first-node weights (layers 0..split-1 + embed_tokens)
    print("Loading Node1 weights...", file=sys.stderr)
    weights = load_shard_weights(model_dir, device)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = Qwen2Config(**{k: v for k, v in cfg_dict.items() if hasattr(Qwen2Config, k)})
    cfg.num_hidden_layers = split
    model1 = Qwen2ForCausalLM(cfg)
    model1.load_state_dict(weights, strict=False)
    model1 = model1.to(torch.float16).to(device).eval()
    del weights

    import uuid
    req_id = str(uuid.uuid4())

    # Prefill: embed + node1 layers
    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model1(input_ids.to(device), use_cache=True)
    hidden = out1.last_hidden_state  # shape [1, seq, hidden]
    past1  = out1.past_key_values
    t_node1_prefill = (time.perf_counter() - t0) * 1000

    # Send hidden states to Node2 via HTTP
    t_net0 = time.perf_counter()
    result2 = post_binary(mac_url, hidden, "generate", req_id)
    t_node2_ttft = (time.perf_counter() - t_net0) * 1000

    first_token = result2["token_id"]
    t_prefill_total = (time.perf_counter() - t0) * 1000
    print(f"Prefill: node1={t_node1_prefill:.0f}ms node2+net={t_node2_ttft:.0f}ms total={t_prefill_total:.0f}ms", file=sys.stderr)

    generated = [first_token]
    decode_times = []

    for _ in range(args.max_tokens - 1):
        t_step = time.perf_counter()
        cur = torch.tensor([[generated[-1]]], device=device)
        with torch.no_grad():
            out1 = model1(cur, past_key_values=past1, use_cache=True)
        past1  = out1.past_key_values
        hidden = out1.last_hidden_state

        result2 = post_binary(mac_url, hidden, "generate", req_id)
        next_tok = result2["token_id"]
        generated.append(next_tok)
        decode_times.append((time.perf_counter() - t_step) * 1000)
        if next_tok == tokenizer.eos_token_id:
            break

    # Clear KV on server
    try:
        urllib.request.urlopen(urllib.request.Request(mac_url,
            data=json.dumps({"mode":"clear_cache","request_id":req_id}).encode(),
            method="POST", headers={"Content-Type":"application/json"}), timeout=5)
    except Exception:
        pass

    total_gen = len(generated)
    tps = total_gen / (sum(decode_times) / 1000) if decode_times else 0
    p50 = sorted(decode_times)[len(decode_times)//2] if decode_times else 0
    p95 = sorted(decode_times)[int(len(decode_times)*0.95)] if decode_times else 0
    text = tokenizer.decode(generated, skip_special_tokens=True)

    result = {
        "mode": "distributed",
        "node1_device": device,
        "node1_layers": f"0-{split}",
        "node2_url": mac_url,
        "node2_layers": f"{split}-28",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": total_gen,
        "prefill_ms": round(t_prefill_total, 1),
        "time_to_first_token_ms": round(t_prefill_total, 1),
        "decode_p50_ms": round(p50, 1),
        "decode_p95_ms": round(p95, 1),
        "decode_total_ms": round(sum(decode_times), 1),
        "tokens_per_second": round(tps, 1),
    }
    print("\n" + "="*60, file=sys.stderr)
    print(json.dumps(result, indent=2))
    print("="*60, file=sys.stderr)
    print("\nGENERATED OUTPUT:", file=sys.stderr)
    print(text)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="~/.hyverk/qwen2.5-7b/inference_layers_0_28")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--split", type=int, default=14,
                   help="Layer split point: Node1 handles layers 0..split-1, Node2 handles split..27")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--distributed", action="store_true",
                   help="Run Node1 locally and forward to Mac server (Node2)")
    p.add_argument("--mac-url", default="http://127.0.0.1:18100",
                   help="URL of the Mac inference server (Node2)")
    args = p.parse_args()

    if args.distributed:
        run_distributed(args)
    else:
        run_single(args)

if __name__ == "__main__":
    main()
