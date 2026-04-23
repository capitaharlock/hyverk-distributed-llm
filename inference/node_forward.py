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

DEBUG_NAN = bool(os.environ.get("HYVERK_DEBUG_NAN"))
COMPILE_LAYERS = bool(os.environ.get("HYVERK_COMPILE"))
COMPILE_MODE = os.environ.get("HYVERK_COMPILE_MODE", "reduce-overhead")
FLASH_ATTN = bool(os.environ.get("HYVERK_FLASH_ATTN"))


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


DEFAULT_TEMPERATURE = _env_float("HYVERK_TEMPERATURE", 0.0)
DEFAULT_TOP_P = _env_float("HYVERK_TOP_P", 1.0)
DEFAULT_TOP_K = _env_int("HYVERK_TOP_K", 0)


def _lru_evict(od, max_entries, log_prefix="KV cache"):
    """Pop least-recently-used entries from an OrderedDict until size <= max_entries.

    Relies on the standard OrderedDict contract: popitem(last=False) removes
    from the front (oldest / LRU). Callers are expected to move_to_end on every
    access so "LRU" tracks actual use, not just insertion order.

    Returns the list of evicted keys in eviction order (mostly useful for tests).
    """
    evicted = []
    while len(od) > max_entries:
        k, _ = od.popitem(last=False)
        evicted.append(k)
        if log_prefix:
            print(
                f"{log_prefix} LRU eviction: dropped {k} (over capacity {max_entries})",
                file=sys.stderr,
            )
    return evicted


def _sample_next_token(last_logits, temperature, top_p, top_k):
    """Pick the next token id from a 1-D logits vector (shape [vocab_size]).

    temperature <= 0 → deterministic argmax (the legacy behaviour).
    top_p  in (0, 1) → nucleus filter: keep the smallest set whose cumulative
                       probability ≥ top_p, mask the rest. top_p>=1 disables.
    top_k  > 0        → keep only the top-k logits. 0 disables.

    NaN/Inf handling and token-0 blocking happen before this call; we assume
    last_logits is already sanitised.
    """
    import torch  # local import — helper also used outside the serve closure
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(last_logits).item())

    lg = last_logits.float() / max(float(temperature), 1e-6)
    vocab = lg.numel()

    if top_k and 0 < top_k < vocab:
        kth = torch.topk(lg, top_k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(lg, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # keep[i] is True while the cumulative prob BEFORE position i is < top_p.
        # That keeps the first token always and every token while we're still
        # under the nucleus boundary.
        keep = (cum - probs) < top_p
        keep[0] = True
        sorted_logits = torch.where(
            keep, sorted_logits, torch.full_like(sorted_logits, float("-inf"))
        )
        # scatter back to original ordering
        lg = torch.full_like(lg, float("-inf"))
        lg.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(lg, dim=-1)
    # guard against a fully -inf row (shouldn't happen but be safe)
    if not torch.isfinite(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(last_logits).item())
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())

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
    p.add_argument("--mode", required=True, choices=["download","embed","forward","generate","serve"])
    p.add_argument("--model-dir", required=True, help="Local dir for cached layer weights")
    p.add_argument("--coordinator", default="https://hyverk-coordinator.fly.dev")
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=28)
    p.add_argument("--input-file", default="", help="Input hidden states (binary torch tensor)")
    p.add_argument("--output-file", default="", help="Output hidden states")
    p.add_argument("--token-ids", default="", help="Comma-separated token IDs for embed mode")
    p.add_argument("--port", type=int, default=18100, help="Port for serve mode")
    p.add_argument("--host", default="127.0.0.1", help="Bind address for serve mode (use 0.0.0.0 for LAN access)")
    p.add_argument("--device", default="", help="Force device: cpu, cuda, mps (default: auto-detect)")
    args = p.parse_args()

    if args.mode == "download":
        download_layers(args)
    elif args.mode == "serve":
        serve_model(args)
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
    if getattr(args, "device", ""):
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    # Sliding-window is disabled below; with full-window sequences we can use SDPA on GPU
    # (fused attention, same math as eager matmul attention — faster tokens/s).
    # Dynamic-quantized CPU path keeps eager for compatibility.
    # Disable sliding window — known to cause bugs when calling Qwen2DecoderLayer directly
    # (transformers issues #35896, #35924, #36361, #40126)
    config.sliding_window = None
    config.use_sliding_window = False
    if hasattr(config, 'max_window_layers'):
        config.max_window_layers = config.num_hidden_layers

    # Load weights
    with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    use_quantize = (device.type == "cpu")
    if use_quantize:
        # ARM (Apple Silicon) needs qnnpack; x86 uses fbgemm
        import platform
        backend = "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm"
        try:
            torch.backends.quantized.engine = backend
            print(f"Using dynamic int8 quantization for CPU inference (backend={backend})", file=sys.stderr)
        except Exception:
            use_quantize = False
            print("int8 quantization unavailable, falling back to fp16 CPU", file=sys.stderr)
        config._attn_implementation = "eager"
    elif device.type == "mps":
        # MPS + SDPA + explicit attention_mask hangs the MPS command queue on
        # certain PyTorch versions. Eager uses the manual QK^T + softmax path
        # which is stable on Apple Silicon.
        config._attn_implementation = "eager"
        # Qwen2.5-7B has unusually large Q/K projection biases (up to 442 in
        # layer 27). The fp16 Q@K^T dot product overflows Metal's fp16 ALU,
        # producing +inf → NaN in softmax. Patch transformers' eager attention
        # to upcast the matmul to fp32 and cast back, matching bfloat16 behaviour.
        import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv as _repeat_kv
        import torch.nn as _nn
        def _eager_attn_fp32(module, query, key, value, attention_mask,
                             scaling, dropout=0.0, **kwargs):
            ks = _repeat_kv(key, module.num_key_value_groups)
            vs = _repeat_kv(value, module.num_key_value_groups)
            aw = torch.matmul(query.float(), ks.float().transpose(2, 3)).to(query.dtype) * scaling
            if attention_mask is not None:
                aw = aw + attention_mask[:, :, :, :ks.shape[-2]]
            aw = _nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
            aw = _nn.functional.dropout(aw, p=dropout, training=module.training)
            out = torch.matmul(aw, vs)
            return out.transpose(1, 2).contiguous(), aw
        _qwen2_mod.eager_attention_forward = _eager_attn_fp32
        print("MPS: patched eager_attention_forward with fp32 Q@K^T upcast", file=sys.stderr)
    else:
        config._attn_implementation = "sdpa"
        # Optional FlashAttention-2 — strictly opt-in (HYVERK_FLASH_ATTN=1) because
        # it requires the flash-attn package AND a CUDA device. We fall back to SDPA
        # silently when either is missing, so a misconfigured env var never breaks
        # the node — it just logs a message.
        if FLASH_ATTN and device.type == "cuda":
            try:
                import flash_attn  # noqa: F401 — presence check
                config._attn_implementation = "flash_attention_2"
                ver = getattr(flash_attn, "__version__", "?")
                print(f"FlashAttention-2 enabled (flash-attn {ver})", file=sys.stderr)
            except ImportError:
                print(
                    "HYVERK_FLASH_ATTN set but flash-attn is not installed; "
                    "falling back to SDPA",
                    file=sys.stderr,
                )
        elif FLASH_ATTN:
            print(
                f"HYVERK_FLASH_ATTN set but device={device.type}; "
                f"FlashAttention-2 requires CUDA — using SDPA",
                file=sys.stderr,
            )

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

    # Optional torch.compile — CUDA only. Opt-in via HYVERK_COMPILE=1 because the
    # first decode step pays the compile cost (seconds) and MPS/CPU inductor
    # support is not reliable enough to default on. HYVERK_COMPILE_MODE overrides
    # the inductor mode (default "reduce-overhead" is tuned for small-batch decode).
    if COMPILE_LAYERS:
        if device.type == "cuda" and hasattr(torch, "compile"):
            try:
                layers = [
                    torch.compile(l, mode=COMPILE_MODE, fullgraph=False, dynamic=True)
                    for l in layers
                ]
                print(
                    f"torch.compile enabled on {len(layers)} layers (mode={COMPILE_MODE})",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"torch.compile failed, running eager: {e}", file=sys.stderr)
        else:
            print(
                f"HYVERK_COMPILE requested but device={device.type}; skipping "
                f"(CUDA only — MPS/CPU inductor is flaky)",
                file=sys.stderr,
            )

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
    lm_head_weight_fp32 = None
    if "lm_head.weight" in weights:
        lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.weight = torch.nn.Parameter(weights["lm_head.weight"], requires_grad=False)
        lm_head = lm_head.to(device)
        lm_head_weight_fp32 = lm_head.weight.data.to(dtype=torch.float32)

    return config, device, rotary, layers, embed, norm, lm_head, lm_head_weight_fp32

def serve_model(args):
    """Persistent HTTP server — loads model once, handles requests via HTTP."""
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    import struct, threading

    print(f"Loading model layers {args.layer_start}-{args.layer_end}...", file=sys.stderr)
    t0 = time.time()
    config, device, rotary, layers, embed, norm, lm_head, lm_head_weight_fp32 = load_model_layers(args)
    print(f"Model loaded in {time.time()-t0:.1f}s on {device}", file=sys.stderr)

    try:
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask
        try:
            from transformers.masking_utils import create_sliding_window_causal_mask
        except ImportError:
            create_sliding_window_causal_mask = None
    except ImportError:
        DynamicCache = None
        create_causal_mask = None
        create_sliding_window_causal_mask = None

    # request_id -> {"cache": DynamicCache, "last_access": ts}. OrderedDict gives
    # us O(1) LRU via move_to_end on every access. Old behaviour was FIFO via
    # next(iter(dict)), which would evict a long-running active request just
    # because it was created first. HYVERK_KV_MAX_ENTRIES tunes the budget.
    from collections import OrderedDict
    kv_cache_state = OrderedDict()
    MAX_CACHE_ENTRIES = max(1, _env_int("HYVERK_KV_MAX_ENTRIES", 8))

    def _touch(rid):
        """Mark request_id as most-recently-used (no-op if not present)."""
        if rid in kv_cache_state:
            kv_cache_state.move_to_end(rid)
            kv_cache_state[rid]["last_access"] = time.time()

    def _evict_if_over_capacity():
        """Drop the least-recently-used entries until we're at budget."""
        _lru_evict(kv_cache_state, MAX_CACHE_ENTRIES, log_prefix="KV cache")

    # Serialises all GPU work across concurrent HTTP threads. /health and other
    # non-model endpoints stay lock-free so they stay responsive during inference.
    model_lock = threading.Lock()
    started_at = time.time()

    # Idle timeout: drop KV entries not touched for N seconds. Prevents leaks
    # when a coordinator drops a request without sending InferenceEnd (client
    # crash, network drop). HYVERK_KV_IDLE_TIMEOUT_S=0 disables the reaper.
    # HYVERK_KV_REAPER_INTERVAL_S controls sweep frequency (min 5s).
    IDLE_TIMEOUT_S = max(0, _env_int("HYVERK_KV_IDLE_TIMEOUT_S", 300))
    REAPER_INTERVAL_S = max(5, _env_int("HYVERK_KV_REAPER_INTERVAL_S", 30))

    def _reap_idle_kv():
        """Background sweeper — drops entries untouched for IDLE_TIMEOUT_S."""
        while True:
            try:
                time.sleep(REAPER_INTERVAL_S)
                if IDLE_TIMEOUT_S <= 0 or not kv_cache_state:
                    continue
                now = time.time()
                # Grab the model_lock so we don't race with in-flight inference
                # mutating kv_cache_state. /health reads lock-free; that's OK —
                # it only snapshots counters, never dereferences the caches.
                if not model_lock.acquire(timeout=5):
                    continue
                try:
                    stale = [
                        rid for rid, entry in list(kv_cache_state.items())
                        if now - entry.get("last_access", now) > IDLE_TIMEOUT_S
                    ]
                    for rid in stale:
                        kv_cache_state.pop(rid, None)
                        print(
                            f"KV cache idle-timeout eviction: dropped {rid} "
                            f"(idle > {IDLE_TIMEOUT_S}s)",
                            file=sys.stderr,
                        )
                finally:
                    model_lock.release()
            except Exception as e:
                # Reaper crashing must not take the server down.
                print(f"KV reaper warning: {e}", file=sys.stderr)

    if IDLE_TIMEOUT_S > 0:
        t = threading.Thread(target=_reap_idle_kv, daemon=True, name="kv-reaper")
        t.start()
        print(
            f"KV cache idle-reaper: timeout={IDLE_TIMEOUT_S}s "
            f"interval={REAPER_INTERVAL_S}s",
            file=sys.stderr,
        )

    # Warmup pass — compile Metal/CUDA kernels before serving requests.
    # Includes lm_head to avoid 60s+ JIT stall on the first real request.
    print("Running warmup pass...", file=sys.stderr)
    with torch.inference_mode():
        dummy = torch.randn(1, 4, config.hidden_size, device=device, dtype=torch.float16)
        pos = torch.arange(4, device=device).unsqueeze(0)
        pe = rotary(dummy, pos)
        for layer in layers:
            out = layer(dummy, position_embeddings=pe, use_cache=True)
            dummy = out[0] if isinstance(out, tuple) else out
        if norm is not None:
            dummy = norm(dummy)
        if lm_head_weight_fp32 is not None:
            _ = torch.nn.functional.linear(dummy.float(), lm_head_weight_fp32)
        if device.type == "mps":
            torch.mps.synchronize()
    print("Warmup done", file=sys.stderr)

    def run_forward_legacy(hidden, request_id, is_generate):
        """Full attention mask (no KV) — fallback when DynamicCache is unavailable."""
        seq_len = hidden.shape[1]
        dtype = hidden.dtype
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        pos_emb = rotary(hidden, position_ids)
        min_val = torch.finfo(dtype).min
        causal_mask = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :]
        with torch.inference_mode():
            for layer in layers:
                try:
                    out = layer(
                        hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=pos_emb,
                    )
                except TypeError:
                    out = layer(
                        hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                        position_embeddings=pos_emb,
                    )
                hidden = out[0] if isinstance(out, tuple) else out
                if DEBUG_NAN and (torch.isnan(hidden).any() or torch.isinf(hidden).any()):
                    print("WARNING: NaN/Inf in hidden after layer", file=sys.stderr)
            if is_generate and norm:
                hidden = norm(hidden)
        if device.type == "mps":
            torch.mps.synchronize()
        return hidden

    def run_forward_kv(hidden, request_id, is_generate, reset_kv):
        """Prefill / decode with HuggingFace DynamicCache (matches Qwen2Model path)."""
        if not layers:
            with torch.inference_mode():
                if is_generate and norm:
                    hidden = norm(hidden)
            return hidden

        # Ensure contiguity — tensors received from numpy (binary HTTP body) or
        # another device may have non-standard strides that trigger MPS faults.
        hidden = hidden.contiguous()
        # Dynamic int8-quantized CPU layers require float32 activations; the KV
        # cache will also store float32 keys/values which avoids a decode-step
        # crash caused by the quantized attn reading fp16 from the cache.
        if device.type == "cpu":
            hidden = hidden.float()
        seq_len = hidden.shape[1]
        print(f"run_forward_kv: rid={request_id} seq={seq_len} past_seen=TBD is_generate={is_generate}", file=sys.stderr, flush=True)
        if reset_kv:
            kv_cache_state.pop(request_id, None)
        if request_id not in kv_cache_state:
            kv_cache_state[request_id] = {
                "cache": DynamicCache(config=config),
                "last_access": time.time(),
            }
            _evict_if_over_capacity()
        else:
            # LRU bump — this request is being actively used, keep it warm.
            _touch(request_id)
        past_key_values = kv_cache_state[request_id]["cache"]
        # Use layer_idx=args.layer_start so DynamicCache reads the correct slot.
        # Default (layer_idx=0) returns 0 when this node only holds layers 14+,
        # which breaks position embeddings on every decode step.
        past_seen = (past_key_values.get_seq_length(layer_idx=args.layer_start)
                     if past_key_values is not None else 0)
        print(f"run_forward_kv: past_seen={past_seen} device={device}", file=sys.stderr, flush=True)
        position_ids = torch.arange(
            past_seen, past_seen + seq_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        pos_emb = rotary(hidden, position_ids)

        # Pure decode (seq=1 appended to existing cache): no mask needed. A single new
        # query attends to cached keys + itself; there is no future position to hide.
        # Skipping create_causal_mask here saves the per-step tensor allocation that
        # dominated the decode-step cost on prefill-heavy runs.
        is_decode_step = (seq_len == 1 and past_seen > 0)

        cache_position = torch.arange(past_seen, past_seen + seq_len, device=device)

        causal_mask_mapping = None
        if create_causal_mask is not None and not is_decode_step:
            # Try the transformers ≥4.48 signature first (input_embeds + cache_position),
            # then fall back to older kwarg sets, then manual upper-triangular mask.
            mask = None
            for mask_kwargs in [
                dict(config=config, input_embeds=hidden, attention_mask=None,
                     cache_position=cache_position, past_key_values=past_key_values,
                     position_ids=position_ids),
                dict(config=config, inputs_embeds=hidden, attention_mask=None,
                     past_key_values=past_key_values, position_ids=position_ids),
                dict(inputs_embeds=hidden, attention_mask=None,
                     past_key_values=past_key_values, position_ids=position_ids),
            ]:
                try:
                    mask = create_causal_mask(**mask_kwargs)
                    break
                except TypeError:
                    continue
            if mask is None:
                # Manual fallback: upper-triangular float mask
                dtype = hidden.dtype
                min_val = torch.finfo(dtype).min
                total_len = past_seen + seq_len
                m = torch.zeros(1, 1, seq_len, total_len, dtype=dtype, device=device)
                m[:, :, :, past_seen:] = torch.triu(
                    torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device),
                    diagonal=1,
                )
                mask = m
            causal_mask_mapping = {"full_attention": mask}
            if getattr(config, "has_sliding_layers", False) and create_sliding_window_causal_mask:
                for mask_kwargs in [
                    dict(config=config, input_embeds=hidden, attention_mask=None,
                         cache_position=cache_position, past_key_values=past_key_values,
                         position_ids=position_ids),
                    dict(config=config, inputs_embeds=hidden, attention_mask=None,
                         past_key_values=past_key_values, position_ids=position_ids),
                ]:
                    try:
                        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
                        break
                    except TypeError:
                        continue

        nl = int(getattr(config, "num_hidden_layers", 28))
        layer_types = getattr(config, "layer_types", None)
        if not layer_types or len(layer_types) < nl:
            layer_types = ["full_attention"] * nl

        with torch.inference_mode():
            for idx, layer in enumerate(layers):
                gi = args.layer_start + idx
                lt = layer_types[gi] if gi < len(layer_types) else "full_attention"
                attn_mask = None
                if causal_mask_mapping is not None:
                    attn_mask = causal_mask_mapping.get(lt, causal_mask_mapping["full_attention"])
                print(f"  layer {gi} start", file=sys.stderr, flush=True)
                try:
                    out = layer(
                        hidden,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        position_embeddings=pos_emb,
                    )
                except TypeError:
                    out = layer(
                        hidden,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=pos_emb,
                    )
                hidden = out[0] if isinstance(out, tuple) else out
                print(f"  layer {gi} done", file=sys.stderr, flush=True)
                if DEBUG_NAN and (torch.isnan(hidden).any() or torch.isinf(hidden).any()):
                    print("WARNING: NaN/Inf in hidden after layer", file=sys.stderr)
            if is_generate and norm:
                hidden = norm(hidden)
        # Flush the MPS command queue after every forward pass. Without this,
        # the Metal command buffer fills up after ~20 decode steps and the
        # process crashes with no Python traceback.
        if device.type == "mps":
            torch.mps.synchronize()
        print("run_forward_kv: complete", file=sys.stderr, flush=True)
        return hidden

    def run_forward(hidden, request_id, is_generate, reset_kv=False):
        if DynamicCache is None or not layers:
            return run_forward_legacy(hidden, request_id, is_generate)
        return run_forward_kv(hidden, request_id, is_generate, reset_kv)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *a): pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/healthz", "/"):
                # Lock-free: reads only snapshot counters. Safe because Python dict
                # len() and DynamicCache.get_seq_length() are atomic for our usage.
                entries = []
                for rid, v in list(kv_cache_state.items()):
                    cache = v.get("cache")
                    try:
                        seen = cache.get_seq_length() if cache is not None else 0
                    except Exception:
                        seen = 0
                    entries.append({"request_id": rid, "kv_tokens": int(seen)})
                self._json(200, {
                    "status": "ready",
                    "device": str(device),
                    "layers": f"{args.layer_start}-{args.layer_end}",
                    "port": args.port,
                    "kv_incremental": DynamicCache is not None,
                    "active_requests": len(kv_cache_state),
                    "max_cache_entries": MAX_CACHE_ENTRIES,
                    "uptime_s": int(time.time() - started_at),
                    "kv_entries": entries,
                })
                return
            self._json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")
            mode = self.headers.get("X-Mode", "")
            shape_str = self.headers.get("X-Shape", "")
            request_id = self.headers.get("X-Request-Id", "")

            t0 = time.time()
            # Serialise GPU work. Parsing the body inside the lock is cheap and lets
            # us treat the model state (kv_cache_state, cache objects) as exclusive.
            acquired = model_lock.acquire(timeout=300)
            if not acquired:
                self._json(503, {"error": "model busy (lock timeout)"})
                return
            try:
                if content_type == "application/json":
                    body = self.rfile.read(length)
                    req = json.loads(body)
                    mode = req.get("mode", mode or "forward")
                    request_id = req.get("request_id", request_id)

                    if mode == "embed":
                        token_ids = req["token_ids"]
                        temperature = float(req.get("temperature", DEFAULT_TEMPERATURE))
                        top_p = float(req.get("top_p", DEFAULT_TOP_P))
                        top_k = int(req.get("top_k", DEFAULT_TOP_K))
                        with torch.inference_mode():
                            ids = torch.tensor([token_ids], device=device, dtype=torch.long)
                            hidden = embed(ids)
                            print(f"embed: after embed lookup dtype={hidden.dtype} shape={hidden.shape} max={hidden.abs().max().item():.4f} nan={torch.isnan(hidden).any().item()}", file=sys.stderr, flush=True)
                            hidden = run_forward(hidden, request_id, False, reset_kv=True)
                            print(f"embed: after forward dtype={hidden.dtype} shape={hidden.shape} max={hidden.abs().max().item():.4f} nan={torch.isnan(hidden).any().item()}", file=sys.stderr, flush=True)
                        data = hidden.cpu().half().numpy().tobytes()
                        shape = list(hidden.shape)
                        elapsed = time.time() - t0
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("X-Shape", json.dumps(shape))
                        self.send_header("X-Elapsed-Ms", str(int(elapsed * 1000)))
                        self.send_header("Content-Length", str(len(data)))
                        # X-Next-Token: predict next token from last hidden (if lm_head available).
                        # Lets single-node clients avoid a separate lm_head_only call.
                        if lm_head_weight_fp32 is not None and norm is not None:
                            with torch.inference_mode():
                                last_h = hidden[:, -1:, :]
                                h_n = norm(last_h)
                                print(f"embed: after norm dtype={h_n.dtype} max={h_n.abs().max().item():.4f} nan={torch.isnan(h_n).any().item()}", file=sys.stderr, flush=True)
                                logits = torch.nn.functional.linear(h_n.float(), lm_head_weight_fp32)
                                ll = logits[0, -1].cpu()
                                print(f"embed: logits nan={torch.isnan(ll).any().item()} max={ll[~torch.isnan(ll)].max().item() if (~torch.isnan(ll)).any() else 'all_nan'}", file=sys.stderr, flush=True)
                                nan_mask = torch.isnan(ll) | torch.isinf(ll)
                                if nan_mask.any():
                                    ll = torch.where(nan_mask, torch.tensor(-1e5), ll)
                                ll[0] = -1e5
                            self.send_header("X-Next-Token", str(_sample_next_token(ll, temperature, top_p, top_k)))
                        self.end_headers()
                        self.wfile.write(data)
                        return

                    if mode == "lm_head_only":
                        # Single-node decode: apply norm + lm_head to an existing
                        # hidden state (output of the last decoder layer) WITHOUT
                        # re-running any decoder layers. Use after embed/embed_step
                        # to predict the next token on a full-model node.
                        if lm_head_weight_fp32 is None:
                            self._json(503, {"error": "lm_head not available on this node"})
                            return
                        hidden_raw = req.get("hidden")  # optional inline float list
                        if hidden_raw is not None:
                            import numpy as np
                            arr = np.array(hidden_raw, dtype=np.float16).reshape(1, 1, config.hidden_size)
                            h = torch.from_numpy(arr.copy()).to(device)
                        else:
                            # Use last KV-cached hidden from the most recent forward pass
                            self._json(400, {"error": "lm_head_only requires 'hidden' field"})
                            return

                        def _fhdr(n, d):
                            v = self.headers.get(n); return float(v) if v else d
                        def _ihdr(n, d):
                            v = self.headers.get(n); return int(v) if v else d
                        temperature = _fhdr("X-Temperature", DEFAULT_TEMPERATURE)
                        top_p = _fhdr("X-Top-P", DEFAULT_TOP_P)
                        top_k = _ihdr("X-Top-K", DEFAULT_TOP_K)

                        with torch.inference_mode():
                            h_normed = norm(h) if norm else h
                            logits = torch.nn.functional.linear(h_normed.float(), lm_head_weight_fp32)
                            last_logits = logits[0, -1]
                            nan_mask = torch.isnan(last_logits) | torch.isinf(last_logits)
                            if nan_mask.any():
                                last_logits = torch.where(nan_mask, torch.tensor(-1e5, device=last_logits.device), last_logits)
                            last_logits[0] = -1e5
                            token_id = _sample_next_token(last_logits.cpu(), temperature, top_p, top_k)

                        elapsed = time.time() - t0
                        resp_body = json.dumps({"token_id": token_id, "elapsed_ms": int(elapsed*1000)}).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return

                    if mode == "embed_step":
                        if DynamicCache is None or embed is None:
                            err = json.dumps(
                                {"error": "embed_step requires transformers DynamicCache and embed weights"}
                            ).encode()
                            self.send_response(503)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(err)))
                            self.end_headers()
                            self.wfile.write(err)
                            return
                        if request_id not in kv_cache_state:
                            err = json.dumps(
                                {"error": "no KV for request_id; run embed (prefill) first"}
                            ).encode()
                            self.send_response(409)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(err)))
                            self.end_headers()
                            self.wfile.write(err)
                            return
                        tid = int(req["token_id"])
                        with torch.inference_mode():
                            ids = torch.tensor([[tid]], device=device, dtype=torch.long)
                            hidden = embed(ids)
                            hidden = run_forward(hidden, request_id, False, reset_kv=False)
                        data = hidden.cpu().half().numpy().tobytes()
                        shape = list(hidden.shape)
                        elapsed = time.time() - t0
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("X-Shape", json.dumps(shape))
                        self.send_header("X-Elapsed-Ms", str(int(elapsed * 1000)))
                        self.send_header("Content-Length", str(len(data)))
                        if lm_head_weight_fp32 is not None and norm is not None:
                            temperature = float(req.get("temperature", DEFAULT_TEMPERATURE))
                            top_p = float(req.get("top_p", DEFAULT_TOP_P))
                            top_k = int(req.get("top_k", DEFAULT_TOP_K))
                            with torch.inference_mode():
                                h_n = norm(hidden[:, -1:, :])
                                logits = torch.nn.functional.linear(h_n.float(), lm_head_weight_fp32)
                                ll = logits[0, -1].cpu()
                                nan_mask = torch.isnan(ll) | torch.isinf(ll)
                                if nan_mask.any():
                                    ll = torch.where(nan_mask, torch.tensor(-1e5), ll)
                                ll[0] = -1e5
                            self.send_header("X-Next-Token", str(_sample_next_token(ll, temperature, top_p, top_k)))
                        self.end_headers()
                        self.wfile.write(data)
                        return

                    if mode == "clear_cache":
                        rid = req.get("request_id", "")
                        kv_cache_state.pop(rid, None)
                        resp = json.dumps({"status": "ok"}).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp)))
                        self.end_headers()
                        self.wfile.write(resp)
                        return

                # Binary request — forward/generate mode
                raw = self.rfile.read(length)
                import numpy as np
                shape = json.loads(shape_str) if shape_str else [1, length // (config.hidden_size * 2), config.hidden_size]
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                is_generate = (mode == "generate")

                # Sampling params. Precedence:
                #   per-request header  >  HYVERK_* env default  >  argmax (0.0 / 1.0 / 0)
                # Coordinator can override per-request via X-Temperature etc.; operators
                # who want cluster-wide non-greedy behaviour without touching Rust today
                # set HYVERK_TEMPERATURE on the last-node box.
                def _fhdr(name, default):
                    v = self.headers.get(name)
                    try: return float(v) if v is not None and v != "" else default
                    except ValueError: return default
                def _ihdr(name, default):
                    v = self.headers.get(name)
                    try: return int(v) if v is not None and v != "" else default
                    except ValueError: return default
                temperature = _fhdr("X-Temperature", DEFAULT_TEMPERATURE)
                top_p       = _fhdr("X-Top-P", DEFAULT_TOP_P)
                top_k       = _ihdr("X-Top-K", DEFAULT_TOP_K)

                with torch.inference_mode():
                    hidden = torch.from_numpy(arr.copy()).to(device)
                    hidden = run_forward(hidden, request_id, is_generate, reset_kv=False)
                    if is_generate and lm_head_weight_fp32 is not None:
                        # Use the cached fp32 weight matrix; the old path called
                        # lm_head.float() every step, which copied the full vocab×hidden
                        # projection (~2 GB on Qwen2.5-7B) to fp32 on every token.
                        logits = torch.nn.functional.linear(hidden.float(), lm_head_weight_fp32)
                        last_logits = logits[0, -1]  # [vocab_size]

                        # Clean NaN/Inf from logits (caused by fp16 overflow in attention)
                        nan_mask = torch.isnan(last_logits) | torch.isinf(last_logits)
                        if nan_mask.any():
                            print(f"WARNING: {nan_mask.sum().item()} NaN/Inf in logits, cleaning", file=sys.stderr)
                            last_logits = torch.where(nan_mask, torch.tensor(-1e5, device=last_logits.device), last_logits)

                        # Block token 0 ("!") — Qwen2 vocab id 0 is "!" which is almost
                        # always a numerical artifact when argmax returns 0 from degraded states
                        last_logits[0] = -1e5

                        token_id = _sample_next_token(last_logits, temperature, top_p, top_k)

                if is_generate and lm_head_weight_fp32 is not None:
                    elapsed = time.time() - t0
                    resp = json.dumps({"token_id": token_id, "elapsed_ms": int(elapsed * 1000)}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                else:
                    data = hidden.cpu().half().numpy().tobytes()
                    shape = list(hidden.shape)
                    elapsed = time.time() - t0
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("X-Shape", json.dumps(shape))
                    self.send_header("X-Elapsed-Ms", str(int(elapsed * 1000)))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            finally:
                model_lock.release()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(json.dumps({
        "status": "ready",
        "port": args.port,
        "device": str(device),
        "layers": f"{args.layer_start}-{args.layer_end}",
        "kv_incremental": DynamicCache is not None,
    }))
    sys.stdout.flush()
    print(f"Serving on http://{args.host}:{args.port}", file=sys.stderr)
    server.serve_forever()

def embed_tokens(args):
    """First node: embed token IDs, forward through layers, save hidden states"""
    config, device, rotary, layers, embed, norm, lm_head, _ = load_model_layers(args)

    token_ids = [int(x) for x in args.token_ids.split(",") if x]
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    hidden = embed(ids)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            out = layer(hidden, position_embeddings=position_embeddings, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else out

    # Save hidden states
    torch.save(hidden.cpu().half(), args.output_file)
    print(json.dumps({"shape": list(hidden.shape), "size": os.path.getsize(args.output_file)}))

def forward_layers(args):
    """Middle node: load hidden states, forward through layers, save result"""
    config, device, rotary, layers, embed, norm, lm_head, _ = load_model_layers(args)

    hidden = torch.load(args.input_file, map_location=device, weights_only=True)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            out = layer(hidden, position_embeddings=position_embeddings, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else out

    torch.save(hidden.cpu().half(), args.output_file)
    print(json.dumps({"shape": list(hidden.shape), "size": os.path.getsize(args.output_file)}))

def generate_token(args):
    """Last node: forward through layers + norm + lm_head → token ID"""
    config, device, rotary, layers, embed, norm, lm_head, lm_head_weight_fp32 = load_model_layers(args)

    hidden = torch.load(args.input_file, map_location=device, weights_only=True)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        for layer in layers:
            out = layer(hidden, position_embeddings=position_embeddings, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else out
        if norm: hidden = norm(hidden)
        if lm_head_weight_fp32 is not None:
            logits = torch.nn.functional.linear(hidden.float(), lm_head_weight_fp32)
            token_id = torch.argmax(logits[0, -1]).item()
        else:
            token_id = 0

    print(json.dumps({"token_id": token_id}))

if __name__ == "__main__":
    main()
