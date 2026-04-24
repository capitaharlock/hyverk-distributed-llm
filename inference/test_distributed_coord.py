#!/usr/bin/env python3
"""
Distributed E2E test — coordinator mode.
Mac acts as coordinator, calls both servers over HTTP.
  Node1 (Windows, CUDA): layers 0-14 at WIN_URL
  Node2 (Mac, MPS):      layers 14-28 at MAC_URL

Flow per token:
  token_ids → Node1 (forward) → hidden_states → Node2 (generate/decode) → next_token_id
"""
import argparse, json, sys, time, uuid
import urllib.request
import numpy as np

WIN_URL = "http://192.168.1.41:18100"
MAC_URL = "http://127.0.0.1:18100"
MAX_TOKENS = 50

PROMPT_IDS = None  # filled after tokenize

def post(url, body_bytes, headers, timeout=60):
    req = urllib.request.Request(url, data=body_bytes, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=timeout)

def embed(url, token_ids, req_id):
    """Node1 prefill: JSON embed request → binary fp16 hidden states."""
    body = json.dumps({"mode": "embed", "token_ids": token_ids, "request_id": req_id}).encode()
    r = post(url, body, {"Content-Type": "application/json"})
    return r.headers, r.read()

def embed_step(url, token_id, req_id):
    """Node1 decode: JSON embed_step request → binary fp16 hidden states."""
    body = json.dumps({"mode": "embed_step", "token_id": token_id, "request_id": req_id}).encode()
    r = post(url, body, {"Content-Type": "application/json"})
    return r.headers, r.read()

def forward_hidden(url, hidden_bytes, shape, mode, req_id):
    """Node2: binary fp16 hidden states → JSON {token_id: N}."""
    r = post(url, hidden_bytes, {
        "Content-Type": "application/octet-stream",
        "X-Mode": mode,
        "X-Request-Id": req_id,
        "X-Shape": json.dumps(shape),
        "X-Temperature": "0.0",
    })
    resp_bytes = r.read()
    return r.headers, resp_bytes

def health(url):
    try:
        r = urllib.request.urlopen(url + "/health", timeout=5)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def main():
    print(f"Checking nodes...", file=sys.stderr)
    h1 = health(WIN_URL)
    h2 = health(MAC_URL)
    print(f"  Node1 (Win {WIN_URL}): {h1}", file=sys.stderr)
    print(f"  Node2 (Mac {MAC_URL}): {h2}", file=sys.stderr)

    # Tokenize via Mac server's tokenizer endpoint, or use fixed token IDs
    # Use a fixed prompt token sequence (Qwen2.5 tokenizer result for the LRU prompt)
    prompt = (
        "Write a Python function that implements a Least Recently Used (LRU) cache "
        "using an OrderedDict."
    )

    # Try to tokenize via Mac health endpoint info, fallback to sending raw text
    # Use the transformers tokenizer locally if available
    try:
        from transformers import AutoTokenizer
        import os
        tok = AutoTokenizer.from_pretrained(
            os.path.expanduser("~/.hyverk/qwen2.5-7b/inference_layers_0_28"))
        ids = tok(prompt, return_tensors="np").input_ids[0].tolist()
        print(f"  Prompt: {len(ids)} tokens", file=sys.stderr)
    except Exception as e:
        print(f"  Tokenizer unavailable ({e}), using hardcoded ids", file=sys.stderr)
        ids = [8504, 264, 13427, 729, 429, 23374, 264, 70297, 49776, 37589, 320,
               43, 49, 52, 8, 3721, 1667, 458, 11623, 27755, 13]

    req_id = str(uuid.uuid4())
    generated = []

    # ── Prefill ─────────────────────────────────────────────────────────────────
    print(f"\nPrefill → Node1 ({len(ids)} tokens)...", file=sys.stderr)
    t0 = time.perf_counter()
    hdrs1, bytes1 = forward(WIN_URL, ids, "forward", req_id)
    t_n1 = (time.perf_counter() - t0) * 1000

    shape1 = json.loads(hdrs1.get("X-Shape", "[1,1,3584]"))
    hidden = np.frombuffer(bytes1, dtype=np.float16).reshape(shape1)
    print(f"  Node1: {t_n1:.0f}ms  shape={shape1}", file=sys.stderr)

    print(f"Prefill → Node2 (generate)...", file=sys.stderr)
    t_n2s = time.perf_counter()
    hdrs2, bytes2 = forward(MAC_URL, hidden, "generate", req_id)
    t_n2 = (time.perf_counter() - t_n2s) * 1000
    t_ttft = (time.perf_counter() - t0) * 1000

    try:
        result = json.loads(bytes2)
        next_tok = result["token_id"]
    except Exception:
        next_tok = int(np.frombuffer(bytes2, dtype=np.uint32)[0])
    generated.append(next_tok)
    print(f"  Node2: {t_n2:.0f}ms  TTFT={t_ttft:.0f}ms  first_token={next_tok}", file=sys.stderr)

    # ── Decode loop ──────────────────────────────────────────────────────────────
    decode_ms = []
    print(f"\nDecode loop ({MAX_TOKENS-1} tokens)...", file=sys.stderr)
    for i in range(MAX_TOKENS - 1):
        ts = time.perf_counter()

        hdrs1, bytes1 = forward(WIN_URL, [next_tok], "decode", req_id)
        shape1 = json.loads(hdrs1.get("X-Shape", "[1,1,3584]"))
        hidden = np.frombuffer(bytes1, dtype=np.float16).reshape(shape1)

        hdrs2, bytes2 = forward(MAC_URL, hidden, "decode", req_id)
        try:
            result = json.loads(bytes2)
            next_tok = result["token_id"]
        except Exception:
            next_tok = int(np.frombuffer(bytes2, dtype=np.uint32)[0])

        step_ms = (time.perf_counter() - ts) * 1000
        decode_ms.append(step_ms)
        generated.append(next_tok)

        if i < 3 or i % 10 == 0:
            print(f"  step {i+1}: {step_ms:.0f}ms  tok={next_tok}", file=sys.stderr)

        if next_tok in (151643, 151645):  # EOS tokens for Qwen2.5
            print(f"  EOS at step {i+1}", file=sys.stderr)
            break

    # ── Results ──────────────────────────────────────────────────────────────────
    p50 = sorted(decode_ms)[len(decode_ms)//2] if decode_ms else 0
    tps = 1000 / p50 if p50 > 0 else 0
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"TTFT:          {t_ttft:.0f} ms", file=sys.stderr)
    print(f"Decode p50:    {p50:.0f} ms/tok", file=sys.stderr)
    print(f"Throughput:    {tps:.1f} tok/s", file=sys.stderr)
    print(f"Tokens gen:    {len(generated)}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Decode text
    try:
        from transformers import AutoTokenizer
        import os
        tok = AutoTokenizer.from_pretrained(
            os.path.expanduser("~/.hyverk/qwen2.5-7b/inference_layers_0_28"))
        text = tok.decode(generated, skip_special_tokens=True)
        print(f"\nGenerated text:\n{text}", file=sys.stderr)
    except Exception:
        print(f"\nGenerated token IDs: {generated}", file=sys.stderr)

if __name__ == "__main__":
    main()
