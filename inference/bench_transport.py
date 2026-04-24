#!/usr/bin/env python3
"""
Benchmark HTTP vs TCP transport latency for hidden-state hops.

Usage:
    python3 inference/bench_transport.py [--seq SEQ] [--reps N] [--port PORT]

Sends a random fp16 hidden-state blob of shape [1, SEQ, 3584] to the
already-running inference server on both transports and reports p50/p95/p99.

Requires the server to be running with Stage 2 code (TCP on port+1).
"""
import argparse, json, socket, struct, statistics, time
import urllib.request, urllib.error
import numpy as np

HIDDEN_SIZE = 3584
DEFAULT_PORT = 18100


def _http_forward(hidden_bytes: bytes, seq: int, port: int, request_id: str) -> float:
    shape = f"[1,{seq},{HIDDEN_SIZE}]"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}",
        data=hidden_bytes,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Mode": "forward",
            "X-Shape": shape,
            "X-Request-Id": request_id,
            "Content-Length": str(len(hidden_bytes)),
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000


def _tcp_forward(hidden_bytes: bytes, seq: int, port: int, request_id: str) -> float:
    meta = json.dumps({
        "mode": "forward",
        "request_id": request_id,
        "shape": [1, seq, HIDDEN_SIZE],
    }).encode()
    total = len(meta) + len(hidden_bytes)
    frame = struct.pack("<II", total, len(meta)) + meta + hidden_bytes

    t0 = time.perf_counter()
    with socket.create_connection(("127.0.0.1", port + 1), timeout=10) as s:
        s.sendall(frame)
        # Read response header
        hdr = _recv_exact(s, 8)
        resp_total, resp_jlen = struct.unpack("<II", hdr)
        resp_binlen = resp_total - resp_jlen
        _recv_exact(s, resp_jlen)
        if resp_binlen > 0:
            _recv_exact(s, resp_binlen)
    return (time.perf_counter() - t0) * 1000


def _recv_exact(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise EOFError("TCP server closed connection")
        buf += chunk
    return buf


def _stats(label: str, samples: list[float]) -> None:
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]
    mean = statistics.mean(s)
    print(f"  {label:8s}  mean={mean:6.1f}ms  p50={p50:6.1f}ms  p95={p95:6.1f}ms  p99={p99:6.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64, help="Sequence length")
    ap.add_argument("--reps", type=int, default=50, help="Number of iterations per transport")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    blob = (np.random.randn(1, args.seq, HIDDEN_SIZE).astype(np.float16)).tobytes()
    print(f"Payload: seq={args.seq}  size={len(blob)/1024:.1f} KB  reps={args.reps}  warmup={args.warmup}")

    # Check server health
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as r:
            h = json.loads(r.read())
        print(f"Server: device={h['device']}  layers={h['layers']}")
        tcp_port = h.get("tcp_port", args.port + 1)
    except Exception as e:
        print(f"Server not reachable at port {args.port}: {e}")
        return

    # HTTP benchmark
    http_times = []
    for i in range(args.warmup + args.reps):
        rid = f"bench-http-{i}"
        try:
            ms = _http_forward(blob, args.seq, args.port, rid)
            if i >= args.warmup:
                http_times.append(ms)
        except Exception as e:
            print(f"  HTTP error rep {i}: {e}")

    # TCP benchmark
    tcp_times = []
    try:
        for i in range(args.warmup + args.reps):
            rid = f"bench-tcp-{i}"
            try:
                ms = _tcp_forward(blob, args.seq, args.port, rid)
                if i >= args.warmup:
                    tcp_times.append(ms)
            except Exception as e:
                print(f"  TCP error rep {i}: {e}")
                break
    except Exception as e:
        print(f"TCP unavailable (server may be old version): {e}")

    print(f"\nResults (seq={args.seq}, {args.reps} reps):")
    if http_times:
        _stats("HTTP", http_times)
    if tcp_times:
        _stats("TCP", tcp_times)
    if http_times and tcp_times:
        speedup = statistics.mean(http_times) / statistics.mean(tcp_times)
        print(f"\n  TCP speedup: {speedup:.2f}x vs HTTP")


if __name__ == "__main__":
    main()
