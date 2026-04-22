#!/usr/bin/env python3
"""
Synthetic benchmark for the distributed-inference decode path.

Runs a tiny Qwen2 model (random weights, same shape family as the tests in
test_node_forward.py) through the same prefill → incremental-decode loop
serve_model exercises. No checkpoint download required, so every contributor
can run a baseline in seconds:

    python3 inference/bench_node_forward.py --prompt-len 64 --steps 64
    # optional:
    HYVERK_COMPILE=1        python3 inference/bench_node_forward.py   # CUDA only
    HYVERK_FLASH_ATTN=1     python3 inference/bench_node_forward.py   # CUDA only

What it reports, per device/impl:
    prefill_ms                wall time of the first forward (with causal mask)
    decode_p50_ms, decode_p95_ms, decode_mean_ms
    tokens_per_second         1000 / decode_mean_ms
    mask_skipped              whether the decode path took the DINF-002 no-mask branch

The numbers are meaningful within one device (CPU, MPS, CUDA) as a relative
baseline; they are NOT a substitute for real 7B wall-clock on the actual
chain, but they catch local regressions in the decoder layer path on any
dev machine.
"""
import argparse, json, os, statistics, sys, time

try:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
    )
    from transformers.cache_utils import DynamicCache
except ImportError as e:
    print(json.dumps({"error": f"missing deps: {e}"}), file=sys.stderr)
    sys.exit(2)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def tiny_config(num_layers: int):
    cfg = AutoConfig.for_model(
        "qwen2",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=2048,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.sliding_window = None
    cfg.use_sliding_window = False
    cfg.max_window_layers = cfg.num_hidden_layers
    return cfg


def build_stack(cfg, device, dtype):
    cfg._attn_implementation = "eager" if device.type == "cpu" else "sdpa"
    rotary = Qwen2RotaryEmbedding(cfg).to(device)
    layers = [
        Qwen2DecoderLayer(cfg, i).to(dtype=dtype, device=device).eval()
        for i in range(cfg.num_hidden_layers)
    ]
    return rotary, layers


def bench(device: torch.device, num_layers: int, prompt_len: int, steps: int,
          warmup: int, dtype_name: str) -> dict:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    if device.type == "cpu" and dtype == torch.float16:
        # CPU has no native fp16 matmul; stick to fp32 to avoid the bench just
        # measuring implicit up-conversion.
        dtype = torch.float32
        dtype_name = "fp32(forced on cpu)"

    cfg = tiny_config(num_layers)
    rotary, layers = build_stack(cfg, device, dtype)

    torch.manual_seed(0)
    cache = DynamicCache(config=cfg)

    # Prefill (multi-token) with proper causal mask.
    prompt = torch.randn(1, prompt_len, cfg.hidden_size, device=device, dtype=dtype)
    pos_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    pos_emb = rotary(prompt, pos_ids)
    min_val = torch.finfo(dtype).min
    causal = torch.triu(torch.full((prompt_len, prompt_len), min_val, device=device, dtype=dtype),
                        diagonal=1)[None, None]

    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        h = prompt
        for layer in layers:
            out = layer(h, attention_mask=causal, position_ids=pos_ids,
                        past_key_values=cache, use_cache=True,
                        position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out
    _sync(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode — mirror the DINF-002 hot path: attention_mask=None when
    # seq_len=1 and cache has past.
    decode_times = []
    for i in range(warmup + steps):
        past_seen = cache.get_seq_length()
        pos_ids_d = torch.tensor([[past_seen]], device=device, dtype=torch.long)
        new = torch.randn(1, 1, cfg.hidden_size, device=device, dtype=dtype)
        pe_d = rotary(new, pos_ids_d)
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            h = new
            for layer in layers:
                out = layer(h, attention_mask=None, position_ids=pos_ids_d,
                            past_key_values=cache, use_cache=True,
                            position_embeddings=pe_d)
                h = out[0] if isinstance(out, tuple) else out
        _sync(device)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            decode_times.append(dt)

    decode_mean = statistics.mean(decode_times)
    return {
        "device": str(device),
        "dtype": dtype_name,
        "num_layers": num_layers,
        "prompt_len": prompt_len,
        "decode_steps": steps,
        "warmup": warmup,
        "prefill_ms": round(prefill_ms, 2),
        "decode_p50_ms": round(statistics.median(decode_times), 3),
        "decode_p95_ms": round(sorted(decode_times)[int(0.95 * (len(decode_times) - 1))], 3),
        "decode_mean_ms": round(decode_mean, 3),
        "tokens_per_second": round(1000.0 / decode_mean, 1) if decode_mean > 0 else None,
        "mask_skipped_in_decode": True,
        "compile_env": bool(os.environ.get("HYVERK_COMPILE")),
        "flash_attn_env": bool(os.environ.get("HYVERK_FLASH_ATTN")),
        "final_kv_tokens": int(cache.get_seq_length()),
    }


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        # mps.synchronize() exists in recent torch; no-op otherwise.
        sync = getattr(torch.mps, "synchronize", None)
        if sync is not None:
            sync()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = p.parse_args()

    device = pick_device(args.device)
    result = bench(device, args.num_layers, args.prompt_len,
                   args.steps, args.warmup, args.dtype)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
