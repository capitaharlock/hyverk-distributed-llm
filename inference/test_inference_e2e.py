#!/usr/bin/env python3
"""
End-to-end distributed inference test for hyverk.com — Qwen2.5-7B.

Single-machine mode (baseline):
  python3 inference/test_inference_e2e.py \
      --model-dir ~/.hyverk/qwen2.5-7b/inference_layers_0_28 \
      --device mps --max-tokens 120

Distributed mode (Win=Node1 CUDA layers 0-split, Mac=Node2 MPS layers split-28):
  python3 inference/test_inference_e2e.py \
      --model-dir ~/.hyverk/qwen2.5-7b/inference_layers_0_28 \
      --device cuda --split 14 --max-tokens 120 \
      --distributed --mac-url http://192.168.1.37:18100
"""
import argparse, json, os, sys, time, uuid
import urllib.request

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM
from safetensors.torch import load_file

PROMPT = (
    "Write a Python function that implements a Least Recently Used (LRU) cache "
    "using an OrderedDict. The function should support get(key) and put(key, value) "
    "operations with O(1) time complexity. Include a capacity parameter and evict "
    "the least recently used item when full. Add clear docstrings and type hints."
)


# ── weight loading ────────────────────────────────────────────────────────────

def load_weights(model_dir: str) -> dict:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    weights = {}
    for shard in sorted(set(weight_map.values())):
        print(f"  loading {shard}...", file=sys.stderr)
        weights.update(load_file(os.path.join(model_dir, shard), device="cpu"))
    return weights


def build_node1_model(model_dir: str, split: int, device: torch.device) -> Qwen2ForCausalLM:
    """Load embed_tokens + layers 0..split-1 (Node 1)."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.sliding_window = None
    cfg.use_sliding_window = False
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = split
    cfg.num_hidden_layers = split
    cfg._attn_implementation = "sdpa"

    model = Qwen2ForCausalLM(cfg)
    weights = load_weights(model_dir)
    # Keep only embed_tokens + layers 0..split-1 + lm_head (lm_head not used but harmless)
    node1_keys = {k: v for k, v in weights.items()
                  if "embed_tokens" in k
                  or any(f"model.layers.{i}." in k for i in range(split))}
    missing, _ = model.load_state_dict(node1_keys, strict=False)
    del weights
    return model.to(torch.float16).to(device).eval()


# ── HTTP helpers (talk to Mac Node2 server) ───────────────────────────────────

def post_hidden(url: str, hidden: torch.Tensor, mode: str, req_id: str):
    import numpy as np
    shape = list(hidden.shape)
    data = hidden.to(torch.float16).cpu().numpy().tobytes()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("X-Mode", mode)
    req.add_header("X-Request-Id", req_id)
    req.add_header("X-Shape", json.dumps(shape))
    req.add_header("X-Temperature", "0.0")
    r = urllib.request.urlopen(req, timeout=120)
    if mode == "generate":
        return json.loads(r.read())
    shape_out = json.loads(r.headers["X-Shape"])
    return torch.from_numpy(np.frombuffer(r.read(), dtype=np.float16).reshape(shape_out).copy())


def clear_kv(url: str, req_id: str):
    try:
        data = json.dumps({"mode": "clear_cache", "request_id": req_id}).encode()
        urllib.request.urlopen(urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"}), timeout=5)
    except Exception:
        pass


# ── single-machine inference ──────────────────────────────────────────────────

def run_single(args):
    model_dir = os.path.expanduser(args.model_dir)
    device    = torch.device(args.device)
    print(f"\nSINGLE-MACHINE | device={args.device} | max_tokens={args.max_tokens}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}", file=sys.stderr)

    print("Loading model...", file=sys.stderr)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.sliding_window = None
    cfg.use_sliding_window = False
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = cfg.num_hidden_layers
    cfg._attn_implementation = "sdpa"

    model = Qwen2ForCausalLM(cfg)
    weights = load_weights(model_dir)
    model.load_state_dict(weights, strict=False)
    del weights
    model = model.to(torch.float16).to(device).eval()
    print("Model ready.", file=sys.stderr)

    ids = input_ids.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(ids, use_cache=False)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[0, -1].argmax().item()
    t_ttft = (time.perf_counter() - t0) * 1000
    generated = [next_tok]

    # Decode
    decode_ms = []
    for _ in range(args.max_tokens - 1):
        ts = time.perf_counter()
        with torch.no_grad():
            out = model(torch.tensor([[next_tok]], device=device),
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax().item()
        generated.append(next_tok)
        decode_ms.append((time.perf_counter() - ts) * 1000)
        if next_tok == tokenizer.eos_token_id:
            break

    tps = len(generated) / (sum(decode_ms) / 1000) if decode_ms else 0
    sorted_ms = sorted(decode_ms)
    result = {
        "mode": "single_machine",
        "device": args.device,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated),
        "time_to_first_token_ms": round(t_ttft, 1),
        "decode_p50_ms": round(sorted_ms[len(sorted_ms) // 2], 1) if sorted_ms else 0,
        "decode_p95_ms": round(sorted_ms[int(len(sorted_ms) * 0.95)], 1) if sorted_ms else 0,
        "tokens_per_second": round(tps, 1),
    }
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n" + "=" * 60, file=sys.stderr)
    print(json.dumps(result, indent=2))
    print("=" * 60 + "\nGENERATED:", file=sys.stderr)
    print(text)


# ── distributed inference ─────────────────────────────────────────────────────

def run_distributed(args):
    model_dir = os.path.expanduser(args.model_dir)
    device    = torch.device(args.device)
    split     = args.split
    mac_url   = args.mac_url
    print(f"\nDISTRIBUTED | Node1={args.device} layers 0-{split} | Node2={mac_url} layers {split}-28", file=sys.stderr)

    # Verify Mac server is reachable
    try:
        h = json.loads(urllib.request.urlopen(f"{mac_url}/health", timeout=5).read())
        print(f"Mac Node2: {h['status']} | {h['device']} | layers={h['layers']}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Mac Node2 unreachable at {mac_url}: {e}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_tokens}", file=sys.stderr)

    print(f"Loading Node1 weights (layers 0-{split})...", file=sys.stderr)
    model1 = build_node1_model(model_dir, split, device)
    print("Node1 ready.", file=sys.stderr)

    req_id = str(uuid.uuid4())
    ids = input_ids.to(device)

    # Warmup Node1
    with torch.no_grad():
        _ = model1.model(ids, use_cache=False)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model1.model(ids, use_cache=True)
    hidden = out1.last_hidden_state
    past1  = out1.past_key_values
    t_node1 = (time.perf_counter() - t0) * 1000

    # Send to Mac Node2 → first token
    t_net = time.perf_counter()
    r2 = post_hidden(mac_url, hidden, "generate", req_id)
    t_node2 = (time.perf_counter() - t_net) * 1000

    next_tok = r2["token_id"]
    t_ttft = (time.perf_counter() - t0) * 1000
    print(f"Prefill: Node1={t_node1:.0f}ms | Node2+net={t_node2:.0f}ms | TTFT={t_ttft:.0f}ms", file=sys.stderr)
    generated = [next_tok]

    # Decode loop
    decode_ms = []
    node1_ms, node2_ms = [], []
    for i in range(args.max_tokens - 1):
        ts = time.perf_counter()
        with torch.no_grad():
            out1 = model1.model(torch.tensor([[next_tok]], device=device),
                                past_key_values=past1, use_cache=True)
        past1  = out1.past_key_values
        hidden = out1.last_hidden_state
        t1 = (time.perf_counter() - ts) * 1000

        t_n2 = time.perf_counter()
        r2 = post_hidden(mac_url, hidden, "generate", req_id)
        t2 = (time.perf_counter() - t_n2) * 1000

        next_tok = r2["token_id"]
        step_ms = (time.perf_counter() - ts) * 1000
        generated.append(next_tok)
        decode_ms.append(step_ms)
        node1_ms.append(t1)
        node2_ms.append(t2)

        if next_tok == tokenizer.eos_token_id:
            break
        if (i + 1) % 20 == 0:
            print(f"  step {i+1}: node1={t1:.1f}ms node2+net={t2:.1f}ms total={step_ms:.1f}ms", file=sys.stderr)

    clear_kv(mac_url, req_id)

    tps = len(generated) / (sum(decode_ms) / 1000) if decode_ms else 0
    sorted_ms = sorted(decode_ms)
    result = {
        "mode": "distributed",
        "node1_device": args.device,
        "node1_layers": f"0-{split}",
        "node2_url": mac_url,
        "node2_layers": f"{split}-28",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated),
        "time_to_first_token_ms": round(t_ttft, 1),
        "decode_p50_ms": round(sorted_ms[len(sorted_ms) // 2], 1) if sorted_ms else 0,
        "decode_p95_ms": round(sorted_ms[int(len(sorted_ms) * 0.95)], 1) if sorted_ms else 0,
        "decode_node1_mean_ms": round(sum(node1_ms) / len(node1_ms), 1) if node1_ms else 0,
        "decode_node2_mean_ms": round(sum(node2_ms) / len(node2_ms), 1) if node2_ms else 0,
        "tokens_per_second": round(tps, 1),
    }
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n" + "=" * 60, file=sys.stderr)
    print(json.dumps(result, indent=2))
    print("=" * 60 + "\nGENERATED:", file=sys.stderr)
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="~/.hyverk/qwen2.5-7b/inference_layers_0_28")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--split", type=int, default=14)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--mac-url", default="http://192.168.1.37:18100")
    args = p.parse_args()

    if args.distributed:
        run_distributed(args)
    else:
        run_single(args)

if __name__ == "__main__":
    main()
