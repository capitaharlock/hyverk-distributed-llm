#!/usr/bin/env python3
"""
Hyverk Cluster Entry Server
============================
Single HTTP entry point for distributed inference.

  POST /generate   { "prompt": "...", "max_tokens": 256, "temperature": 0.7 }
  GET  /health     — also shows live node map

Supports 1, 2, or 3 nodes. Discovers ready nodes from the coordinator and
routes the pipeline automatically:
  tokenize → Node[0] embed (CUDA/MPS) → Node[1..n-1] forward → Node[-1] lm_head → token

Future: quota enforcement per API key (tokens_in / tokens_out limits).
"""
import argparse, json, os, sys, time, uuid
import urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

COORDINATOR_URL = os.environ.get("HYVERK_COORDINATOR", "http://127.0.0.1:17000")
DEFAULT_MODEL_DIR = os.environ.get("HYVERK_MODEL_DIR",
                    os.path.expanduser("~/.hyverk/qwen2.5-7b/inference_layers_0_28"))
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are Hyverk, an expert coding assistant trained by the community."

# Static fallback node map (used if coordinator unreachable).
# Each entry: {"url": "http://IP:18100", "layer_start": N, "layer_end": N, "name": "..."}
STATIC_NODES = [
    {"url": os.environ.get("HYVERK_NODE1_URL", "http://192.168.1.41:18100"),
     "layer_start": 0, "layer_end": 10, "name": "win-rtx4060"},
    {"url": os.environ.get("HYVERK_NODE2_URL", "http://127.0.0.1:18100"),
     "layer_start": 10, "layer_end": 28, "name": "mac-m4max"},
]

# ── Node discovery ────────────────────────────────────────────────────────────

# Node inference URL map: populated from coordinator registration data or env.
# Key = node_name, value = "http://IP:18100"
NODE_URL_OVERRIDES = {
    k[len("HYVERK_NODE_URL_"):].lower(): v
    for k, v in os.environ.items()
    if k.startswith("HYVERK_NODE_URL_")
}
# e.g. HYVERK_NODE_URL_MACBOOK_M4_MAX=http://127.0.0.1:18100

def _known_url(name: str):
    key = name.lower().replace("-", "_").replace(" ", "_")
    return NODE_URL_OVERRIDES.get(key)

def _default_url_for(name: str) -> str:
    """Best-guess URL: local if it's this machine, else None."""
    if "m4" in name.lower() or "macbook" in name.lower():
        return "http://127.0.0.1:18100"
    if "m1" in name.lower():
        return os.environ.get("HYVERK_M1_URL", "")
    if "rtx" in name.lower() or "desktop" in name.lower() or "win" in name.lower():
        return os.environ.get("HYVERK_WIN_URL", "http://192.168.1.41:18100")
    return ""

def get_ready_nodes() -> list:
    """Fetch ready nodes from coordinator, sorted by layer_start."""
    try:
        r = urllib.request.urlopen(f"{COORDINATOR_URL}/api/v1/cluster/status", timeout=5)
        data = json.loads(r.read())
        nodes = []
        for n in data.get("nodes", []):
            if n.get("state") != "ready":
                continue
            name = n.get("node_name", "")
            url = _known_url(name) or _default_url_for(name)
            if not url:
                continue
            nodes.append({
                "name": name,
                "url": url,
                "layer_start": n.get("layer_start", 0),
                "layer_end": n.get("layer_end", 28),
            })
        nodes.sort(key=lambda x: x["layer_start"])
        return nodes if nodes else STATIC_NODES
    except Exception:
        return STATIC_NODES

# ── Tokenizer ─────────────────────────────────────────────────────────────────

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

def node_embed(url: str, token_ids: list, req_id: str) -> np.ndarray:
    """First node: token_ids → fp16 hidden states via embed JSON mode."""
    body = json.dumps({"mode": "embed", "token_ids": token_ids, "request_id": req_id}).encode()
    hdrs, data = _post(url, body, {"Content-Type": "application/json"})
    shape = json.loads(hdrs.get("X-Shape") or hdrs.get("x-shape", "[1,1,3584]"))
    return np.frombuffer(data, dtype=np.float16).reshape(shape)

def node_step(url: str, token_id: int, req_id: str) -> np.ndarray:
    """First node decode step: token_id → fp16 hidden states."""
    body = json.dumps({"mode": "embed_step", "token_id": token_id, "request_id": req_id}).encode()
    hdrs, data = _post(url, body, {"Content-Type": "application/json"})
    shape = json.loads(hdrs.get("X-Shape") or hdrs.get("x-shape", "[1,1,3584]"))
    return np.frombuffer(data, dtype=np.float16).reshape(shape)

def node_forward(url: str, hidden: np.ndarray, req_id: str) -> np.ndarray:
    """Middle node: fp16 hidden states → fp16 hidden states."""
    body = hidden.astype(np.float16).tobytes()
    hdrs, data = _post(url, body, {
        "Content-Type": "application/octet-stream",
        "X-Mode": "forward",
        "X-Request-Id": req_id,
        "X-Shape": json.dumps(list(hidden.shape)),
        "X-Temperature": "0.0",
    })
    shape = json.loads(hdrs.get("X-Shape") or hdrs.get("x-shape", str(list(hidden.shape))))
    return np.frombuffer(data, dtype=np.float16).reshape(shape)

def node_generate(url: str, hidden: np.ndarray, mode: str, req_id: str, temperature: float) -> int:
    """Last node: fp16 hidden states → next token id."""
    body = hidden.astype(np.float16).tobytes()
    hdrs, data = _post(url, body, {
        "Content-Type": "application/octet-stream",
        "X-Mode": mode,
        "X-Request-Id": req_id,
        "X-Shape": json.dumps(list(hidden.shape)),
        "X-Temperature": str(temperature),
    })
    try:
        return json.loads(data)["token_id"]
    except Exception:
        return int(np.frombuffer(data, dtype=np.uint32)[0])

def node_health(url: str) -> dict:
    try:
        r = urllib.request.urlopen(url + "/health", timeout=3)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# ── Core inference pipeline ───────────────────────────────────────────────────

EOS_TOKENS = {151643, 151645}

def run_inference(prompt: str, max_tokens: int, temperature: float) -> dict:
    nodes = get_ready_nodes()
    if not nodes:
        raise RuntimeError("No ready nodes available")

    tok = get_tokenizer()
    full_prompt = build_prompt(prompt)
    token_ids = tok(full_prompt, return_tensors="np").input_ids[0].tolist()
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    first, *middle, last = nodes if len(nodes) > 1 else (nodes[0], [], nodes[0])
    if len(nodes) == 1:
        first = last = nodes[0]
        middle = []

    # Prefill: first node embeds tokens
    hidden = node_embed(first["url"], token_ids, req_id)
    t_ttft_start = time.perf_counter()

    # Middle nodes forward
    for node in middle:
        hidden = node_forward(node["url"], hidden, req_id)

    # Last node generates first token
    first_token = node_generate(last["url"], hidden, "generate", req_id, temperature)
    t_ttft = (time.perf_counter() - t0) * 1000
    generated = [first_token]

    # Decode loop
    for _ in range(max_tokens - 1):
        if generated[-1] in EOS_TOKENS:
            break
        hidden = node_step(first["url"], generated[-1], req_id)
        for node in middle:
            hidden = node_forward(node["url"], hidden, req_id)
        next_tok = node_generate(last["url"], hidden, "decode", req_id, temperature)
        generated.append(next_tok)

    t_total = (time.perf_counter() - t0) * 1000
    text = tok.decode(generated, skip_special_tokens=True)
    tps = len(generated) / (t_total / 1000)

    node_names = " → ".join(f"{n['name']}({n['layer_start']}-{n['layer_end']})" for n in nodes)
    return {
        "text": text,
        "tokens_in": len(token_ids),
        "tokens_out": len(generated),
        "ttft_ms": round(t_ttft),
        "total_ms": round(t_total),
        "tok_s": round(tps, 1),
        "pipeline": node_names,
    }

# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
            nodes = get_ready_nodes()
            health = {n["name"]: node_health(n["url"]) for n in nodes}
            pipeline = " → ".join(
                f"{n['name']}({n['layer_start']}-{n['layer_end']})" for n in nodes)
            self._send_json(200, {
                "status": "ok",
                "pipeline": pipeline,
                "nodes": health,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "use POST /generate"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
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

        try:
            result = run_inference(prompt, max_tokens, temperature)
            self._send_json(200, result)
        except urllib.error.URLError as e:
            self._send_json(503, {"error": f"Node unreachable: {e}"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

        # TODO (quota phase): deduct tokens_in + tokens_out from api_key budget


def run_server(port: int):
    print(f"Loading tokenizer...", flush=True)
    get_tokenizer()
    nodes = get_ready_nodes()
    print(f"Cluster entry on :{port}", flush=True)
    print(f"Pipeline: {' → '.join(n['name'] for n in nodes)}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18200)
    args = p.parse_args()
    run_server(args.port)
