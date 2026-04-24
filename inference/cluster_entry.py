#!/usr/bin/env python3
"""
Hyverk Cluster Entry Server
============================
Single HTTP entry point for distributed inference.

  POST /generate   { "prompt": "...", "max_tokens": 256, "temperature": 0.7 }
  GET  /health

Internally chains:
  tokenize → Node1 (CUDA, layers 0-split) → Node2 (MPS, layers split-28) → detokenize

Future: quota enforcement per API key (tokens_in / tokens_out limits).
"""
import argparse, json, os, sys, time, uuid, struct
import urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_NODE1_URL  = os.environ.get("HYVERK_NODE1_URL", "http://192.168.1.41:18100")  # Win CUDA 0-14
DEFAULT_NODE2_URL  = os.environ.get("HYVERK_NODE2_URL", "http://127.0.0.1:18100")     # Mac MPS 14-28
DEFAULT_MODEL_DIR  = os.environ.get("HYVERK_MODEL_DIR",
                     os.path.expanduser("~/.hyverk/qwen2.5-7b/inference_layers_0_28"))
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are Hyverk, an expert coding assistant trained by the community."

# ── Tokenizer (loaded once at startup) ────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR)
    return _tokenizer

def build_prompt(user_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Node HTTP helpers ─────────────────────────────────────────────────────────

def _post(url: str, body: bytes, headers: dict, timeout: int = 120) -> tuple:
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    r = urllib.request.urlopen(req, timeout=timeout)
    return r.headers, r.read()

def node1_embed(url: str, token_ids: list, req_id: str) -> tuple:
    """Prefill on Node1: token_ids → fp16 hidden states."""
    body = json.dumps({"mode": "embed", "token_ids": token_ids, "request_id": req_id}).encode()
    hdrs, data = _post(url, body, {"Content-Type": "application/json"})
    shape = json.loads(hdrs.get("X-Shape") or hdrs.get("x-shape", "[1,1,3584]"))
    hidden = np.frombuffer(data, dtype=np.float16).reshape(shape)
    return hidden, hdrs

def node1_step(url: str, token_id: int, req_id: str) -> tuple:
    """Decode step on Node1: token_id → fp16 hidden states."""
    body = json.dumps({"mode": "embed_step", "token_id": token_id, "request_id": req_id}).encode()
    hdrs, data = _post(url, body, {"Content-Type": "application/json"})
    shape = json.loads(hdrs.get("X-Shape") or hdrs.get("x-shape", "[1,1,3584]"))
    hidden = np.frombuffer(data, dtype=np.float16).reshape(shape)
    return hidden, hdrs

def node2_generate(url: str, hidden: np.ndarray, mode: str, req_id: str, temperature: float) -> int:
    """Node2: fp16 hidden states → next token id."""
    body = hidden.astype(np.float16).tobytes()
    hdrs, data = _post(url, body, {
        "Content-Type": "application/octet-stream",
        "X-Mode": mode,
        "X-Request-Id": req_id,
        "X-Shape": json.dumps(list(hidden.shape)),
        "X-Temperature": str(temperature),
    })
    # Response is JSON {"token_id": N} or binary u32
    try:
        return json.loads(data)["token_id"]
    except Exception:
        return int(np.frombuffer(data, dtype=np.uint32)[0])

# ── Core inference pipeline ───────────────────────────────────────────────────

EOS_TOKENS = {151643, 151645}  # Qwen2.5 <|endoftext|> and <|im_end|>

def run_inference(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    node1_url: str = DEFAULT_NODE1_URL,
    node2_url: str = DEFAULT_NODE2_URL,
) -> dict:
    tok = get_tokenizer()
    full_prompt = build_prompt(prompt)
    token_ids = tok(full_prompt, return_tensors="np").input_ids[0].tolist()
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # Prefill
    hidden, _ = node1_embed(node1_url, token_ids, req_id)
    first_token = node2_generate(node2_url, hidden, "generate", req_id, temperature)
    t_ttft = (time.perf_counter() - t0) * 1000

    generated = [first_token]

    # Decode loop
    for _ in range(max_tokens - 1):
        if generated[-1] in EOS_TOKENS:
            break
        hidden, _ = node1_step(node1_url, generated[-1], req_id)
        next_tok = node2_generate(node2_url, hidden, "decode", req_id, temperature)
        generated.append(next_tok)

    t_total = (time.perf_counter() - t0) * 1000
    output_text = tok.decode(generated, skip_special_tokens=True)
    tokens_out = len(generated)
    tokens_in = len(token_ids)

    return {
        "text": output_text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "ttft_ms": round(t_ttft),
        "total_ms": round(t_total),
        "tok_s": round(tokens_out / (t_total / 1000), 1),
    }

# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    node1_url: str = DEFAULT_NODE1_URL
    node2_url: str = DEFAULT_NODE2_URL

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/"):
            # Check both nodes
            def ping(url):
                try:
                    r = urllib.request.urlopen(url + "/health", timeout=3)
                    return json.loads(r.read())
                except Exception as e:
                    return {"error": str(e)}

            self._send_json(200, {
                "status": "ok",
                "node1": ping(self.node1_url),
                "node2": ping(self.node2_url),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "use POST /generate"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = req.get("prompt", "").strip()
        if not prompt:
            self._send_json(400, {"error": "prompt required"})
            return

        max_tokens = int(req.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(req.get("temperature", DEFAULT_TEMPERATURE))

        # TODO (quota phase): check api_key token budget here
        # api_key = req.get("api_key") or self.headers.get("Authorization", "").removeprefix("Bearer ")
        # quota_check(api_key, max_tokens)

        try:
            result = run_inference(
                prompt, max_tokens, temperature,
                self.node1_url, self.node2_url,
            )
            self._send_json(200, result)
        except urllib.error.URLError as e:
            self._send_json(503, {"error": f"Node unreachable: {e}"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

        # TODO (quota phase): deduct tokens_in + tokens_out from api_key budget


def run_server(port: int, node1_url: str, node2_url: str):
    Handler.node1_url = node1_url
    Handler.node2_url = node2_url
    # Pre-load tokenizer
    print(f"Loading tokenizer from {DEFAULT_MODEL_DIR}...", flush=True)
    get_tokenizer()
    print(f"Cluster entry server on :{port}", flush=True)
    print(f"  Node1 (layers  0-14): {node1_url}", flush=True)
    print(f"  Node2 (layers 14-28): {node2_url}", flush=True)
    print(f"  POST /generate  {{prompt, max_tokens, temperature}}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hyverk cluster entry server")
    p.add_argument("--port",      type=int, default=18200)
    p.add_argument("--node1-url", default=DEFAULT_NODE1_URL, help="Node1 URL (CUDA, layers 0-14)")
    p.add_argument("--node2-url", default=DEFAULT_NODE2_URL, help="Node2 URL (MPS, layers 14-28)")
    args = p.parse_args()
    run_server(args.port, args.node1_url, args.node2_url)
