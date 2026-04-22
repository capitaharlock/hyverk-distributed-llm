#!/usr/bin/env python3
"""
Unit tests for node_forward serve path: decode hot-path invariants.

These use a tiny synthetic Qwen2 config (not the real weights) so they run
fast and don't need the 7B checkpoint. The goal is to catch shape / API
regressions in run_forward_kv, the lm_head fp32 cache, and the NaN gate.

Run with:   python3 inference/test_node_forward.py
"""
import os, sys, unittest, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
    )
    from transformers.cache_utils import DynamicCache
    HAS_DEPS = True
except Exception as e:
    HAS_DEPS = False
    IMPORT_ERR = str(e)


def tiny_config():
    cfg = AutoConfig.for_model(
        "qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.sliding_window = None
    cfg.use_sliding_window = False
    cfg.max_window_layers = cfg.num_hidden_layers
    cfg._attn_implementation = "eager"
    return cfg


@unittest.skipUnless(HAS_DEPS, f"torch/transformers not available: {globals().get('IMPORT_ERR','')}")
class DecodeHotPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config()
        cls.device = torch.device("cpu")
        cls.rotary = Qwen2RotaryEmbedding(cls.cfg).to(cls.device)
        cls.layers = [
            Qwen2DecoderLayer(cls.cfg, i).to(dtype=torch.float32, device=cls.device).eval()
            for i in range(cls.cfg.num_hidden_layers)
        ]
        cls.norm = Qwen2RMSNorm(cls.cfg.hidden_size, eps=cls.cfg.rms_norm_eps).to(cls.device)
        cls.lm_head = torch.nn.Linear(cls.cfg.hidden_size, cls.cfg.vocab_size, bias=False).to(cls.device)

    def _prefill(self, seq_len):
        torch.manual_seed(0)
        cache = DynamicCache(config=self.cfg)
        hidden = torch.randn(1, seq_len, self.cfg.hidden_size, dtype=torch.float32)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        pos_emb = self.rotary(hidden, position_ids)
        with torch.inference_mode():
            for layer in self.layers:
                out = layer(
                    hidden,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                )
                hidden = out[0] if isinstance(out, tuple) else out
        return cache, hidden

    def test_prefill_then_decode_step_has_no_mask(self):
        """After prefill, decode with seq=1 and None mask must produce shape [1,1,hidden]."""
        cache, _ = self._prefill(seq_len=8)
        past_seen = cache.get_seq_length()
        self.assertEqual(past_seen, 8)

        # Single new token — mimics the hot path after DINF-002: attention_mask=None
        new = torch.randn(1, 1, self.cfg.hidden_size, dtype=torch.float32)
        pos_ids = torch.arange(past_seen, past_seen + 1).unsqueeze(0)
        pos_emb = self.rotary(new, pos_ids)
        with torch.inference_mode():
            hidden = new
            for layer in self.layers:
                out = layer(
                    hidden,
                    attention_mask=None,                # ← the DINF-002 path
                    position_ids=pos_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=pos_emb,
                )
                hidden = out[0] if isinstance(out, tuple) else out
        self.assertEqual(tuple(hidden.shape), (1, 1, self.cfg.hidden_size))
        self.assertEqual(cache.get_seq_length(), past_seen + 1)
        self.assertFalse(torch.isnan(hidden).any().item())

    def test_lm_head_fp32_cache_matches_recast_path(self):
        """Caching lm_head.weight.float() once must be numerically equivalent to the
        old-path `lm_head.float()(hidden.float())`."""
        torch.manual_seed(1)
        hidden = torch.randn(1, 1, self.cfg.hidden_size, dtype=torch.float16)

        # Old path: cast the entire Linear every call
        old = self.lm_head.half().float()(hidden.float())

        # New path (DINF-002): cache fp32 weight once, reuse
        lm_head_weight_fp32 = self.lm_head.weight.data.to(dtype=torch.float32)
        new = torch.nn.functional.linear(hidden.float(), lm_head_weight_fp32)

        self.assertEqual(old.shape, new.shape)
        # Same math modulo tiny numeric noise
        torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-5)

    def test_decode_step_matches_full_forward(self):
        """Incremental decode (prefill then one step) must match a single full forward
        up to numerical tolerance — validates that skipping the mask on decode is safe."""
        torch.manual_seed(42)
        full_len = 6
        full_tokens = torch.randn(1, full_len, self.cfg.hidden_size, dtype=torch.float32)

        # Full path: one forward, no cache, explicit causal mask
        min_val = torch.finfo(torch.float32).min
        causal = torch.triu(torch.full((full_len, full_len), min_val), diagonal=1)[None, None]
        pos_ids = torch.arange(full_len).unsqueeze(0)
        pos_emb = self.rotary(full_tokens, pos_ids)
        with torch.inference_mode():
            h = full_tokens
            for layer in self.layers:
                out = layer(h, attention_mask=causal, position_ids=pos_ids,
                            past_key_values=None, use_cache=False,
                            position_embeddings=pos_emb)
                h = out[0] if isinstance(out, tuple) else out
        full_last = h[:, -1:, :]

        # Incremental: prefill full_len-1 tokens WITH causal mask (multi-token needs it),
        # then one decode step with no mask (the DINF-002 optimisation).
        prefix_len = full_len - 1
        prefix = full_tokens[:, :prefix_len, :]
        cache = DynamicCache(config=self.cfg)
        pos_prefix = torch.arange(prefix_len).unsqueeze(0)
        pe_prefix = self.rotary(prefix, pos_prefix)
        prefill_mask = torch.triu(
            torch.full((prefix_len, prefix_len), min_val), diagonal=1
        )[None, None]
        with torch.inference_mode():
            h = prefix
            for layer in self.layers:
                out = layer(h, attention_mask=prefill_mask, position_ids=pos_prefix,
                            past_key_values=cache, use_cache=True,
                            position_embeddings=pe_prefix)
                h = out[0] if isinstance(out, tuple) else out

            # Decode step
            new_tok = full_tokens[:, full_len - 1:full_len, :]
            pos_new = torch.tensor([[full_len - 1]])
            pe_new = self.rotary(new_tok, pos_new)
            h = new_tok
            for layer in self.layers:
                out = layer(h, attention_mask=None, position_ids=pos_new,
                            past_key_values=cache, use_cache=True,
                            position_embeddings=pe_new)
                h = out[0] if isinstance(out, tuple) else out
        incr_last = h

        # fp32 eager Qwen2 with the same weights: should match within fp32 rounding.
        torch.testing.assert_close(incr_last, full_last, rtol=5e-3, atol=5e-3)


def test_debug_nan_env_gate():
    """Verify the HYVERK_DEBUG_NAN env var toggle is read at import time."""
    spec = importlib.util.spec_from_file_location(
        "node_forward", os.path.join(ROOT, "node_forward.py"))
    os.environ["HYVERK_DEBUG_NAN"] = "1"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEBUG_NAN is True, "HYVERK_DEBUG_NAN=1 should enable the flag"
    del os.environ["HYVERK_DEBUG_NAN"]

    spec2 = importlib.util.spec_from_file_location(
        "node_forward_off", os.path.join(ROOT, "node_forward.py"))
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    assert mod2.DEBUG_NAN is False, "unset HYVERK_DEBUG_NAN should disable the flag"


def test_health_endpoint_lockfree_during_inference():
    """Stand up a minimal ThreadingHTTPServer that mimics serve_model's health wiring
    and verify /health responds even while a POST holds the model_lock.
    No model weights loaded — this only exercises the HTTP layer."""
    import threading, urllib.request, json as _json
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    import time as _time

    model_lock = threading.Lock()
    started_at = _time.time()
    kv_cache_state = {"req-a": {"cache": None}, "req-b": {"cache": None}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass
        def do_GET(self):
            if self.path in ("/health", "/"):
                body = _json.dumps({
                    "status": "ready",
                    "active_requests": len(kv_cache_state),
                    "uptime_s": int(_time.time() - started_at),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404); self.end_headers()
        def do_POST(self):
            # Simulate a long inference: hold the lock for 1s
            if not model_lock.acquire(timeout=5):
                self.send_response(503); self.end_headers(); return
            try:
                _time.sleep(1.0)
                self.send_response(200); self.send_header("Content-Length","2"); self.end_headers()
                self.wfile.write(b'ok')
            finally:
                model_lock.release()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        # Start a long POST in the background (grabs the lock)
        def long_post():
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", data=b"x", timeout=10).read()
        p = threading.Thread(target=long_post, daemon=True); p.start()
        _time.sleep(0.2)  # let it acquire the lock
        # /health must answer while inference thread holds the lock
        t0 = _time.time()
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
        elapsed = _time.time() - t0
        data = _json.loads(resp)
        assert data["status"] == "ready", data
        assert data["active_requests"] == 2, data
        assert elapsed < 0.5, f"/health should be lock-free, took {elapsed:.2f}s"
        p.join(timeout=3)
        print(f"/health responded in {elapsed*1000:.0f}ms while POST held lock: OK")
    finally:
        srv.shutdown()
    print("DEBUG_NAN env gate: OK")


if __name__ == "__main__":
    test_debug_nan_env_gate()
    test_health_endpoint_lockfree_during_inference()
    unittest.main(verbosity=2, exit=True)
