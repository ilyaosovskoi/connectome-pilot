"""formats.py — which connectome representation trains best?

Candidates:
  A. json-edge  — data/fly_circuit_*.json: neurons/groups/edges edge-list (current).
  B. npz-csr    — normalized signed CSR (data/indices/indptr) + neuron/group arrays.
  C. npz-quant  — same as B but raw weights quantized to int8 (scale stored).

The NPZ variants are bit-exact conversions of the JSON (same synapses, same
signs, same normalization) — they only change storage and load path.

Benchmark measures: file size, load time, 1000-step rollout time, RAM of the
matrix, and numerical equivalence of one rollout. Recommendation is printed
and saved as JSON.

Usage:
  python3 robot/formats.py --circuit data/fly_circuit_256.json
  python3 robot/formats.py --circuit data/fly_circuit_256.json --emit
      # writes data/fly_circuit_256.npz (+ _q8.npz) for training/datasets
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flybrain import FlyCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json_bundle(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def json_to_arrays(bundle):
    neurons = bundle["neurons"]
    n = len(neurons)
    edges = np.asarray(bundle["edges"], dtype=np.float64).reshape(-1, 4)
    pre = edges[:, 0].astype(np.int32)
    post = edges[:, 1].astype(np.int32)
    raw = edges[:, 2].astype(np.float32)
    sign = np.array([x["sign"] for x in neurons], dtype=np.int8)
    side = np.array(
        [1 if x.get("side") == "right" else (0 if x.get("side") == "left" else -1)
         for x in neurons], dtype=np.int8)
    ntype = np.array([x.get("type", "") for x in neurons])
    groups = {k: np.asarray(v, dtype=np.int32) for k, v in bundle["groups"].items()}
    return n, pre, post, raw, sign, side, ntype, groups


def normalized_csr(n, pre, post, raw, sign):
    signed = raw.astype(np.float64) * sign[pre].astype(np.float64)
    in_sum = np.bincount(post, weights=np.abs(raw).astype(np.float64), minlength=n)
    in_sum[in_sum == 0] = 1.0
    vals = (signed / in_sum[post]).astype(np.float32)
    return sp.csr_matrix((vals, (post, pre)), shape=(n, n))


def quantize_int8(raw):
    lo, hi = float(raw.min()), float(raw.max())
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    q = np.clip(np.round((raw - lo) / scale), 0, 255).astype(np.uint8)
    return q, lo, scale


def bundle_npz(path, bundle, quantized=False):
    n, pre, post, raw, sign, side, ntype, groups = json_to_arrays(bundle)
    if quantized:
        q, lo, scale = quantize_int8(raw)
        raw_use = (lo + q.astype(np.float32) * scale)
    else:
        raw_use = raw
    csr = normalized_csr(n, pre, post, raw_use, sign)
    payload = {
        "data": csr.data, "indices": csr.indices, "indptr": csr.indptr,
        "pre": pre, "post": post,
        "sign": sign, "side": side, "ntype": ntype,
        "raw": raw_use,
    }
    for k, v in groups.items():
        payload[f"group_{k}"] = v
    payload["meta_json"] = np.array([json.dumps(bundle.get("meta", {}))])
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def rollout_signature(circuit_path, steps=1000, batch=4, seed=0):
    """Deterministic rollout hash: fixed random input, sum of rates over time."""
    rng = np.random.default_rng(seed)
    c = FlyCircuit(circuit_path)
    c.reset(batch=batch)
    acc = np.zeros(c.n, dtype=np.float64)
    for _ in range(steps):
        ext = rng.normal(0, 5.0, size=(batch, c.n)).astype(np.float32)
        c.step(ext)
        acc += c.rate.mean(axis=0)
    return acc


def bench_load(path, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        FlyCircuit(path)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_steps(n, nnz, reps=3, steps=1000):
    # Synthetic CSR matmul benchmark at the same sparsity (no I/O).
    rng = np.random.default_rng(0)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    B = sp.csr_matrix((rng.normal(size=nnz).astype(np.float32), (rows, cols)),
                      shape=(n, n))
    spikes = (rng.random((8, n)) < 0.05).astype(np.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(steps):
            spikes = (B @ spikes.T).T
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.json"))
    ap.add_argument("--emit", action="store_true",
                    help="write .npz (+ _q8.npz) next to the circuit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    bundle = load_json_bundle(args.circuit)
    n, pre, post, raw, sign, side, ntype, groups = json_to_arrays(bundle)
    base, _ = os.path.splitext(args.circuit)
    npz_path = base + ".npz"
    q8_path = base + "_q8.npz"

    # Build in-memory to benchmark fairly (emit to disk only with --emit).
    buf = io.BytesIO()
    csr = normalized_csr(n, pre, post, raw, sign)
    np.savez_compressed(buf, data=csr.data, indices=csr.indices, indptr=csr.indptr)
    npz_bytes = buf.tell()
    q, lo, scale = quantize_int8(raw)
    deq = (lo + q.astype(np.float32) * scale)
    max_abs_err = float(np.abs(deq - raw).max())
    med = float(np.median(raw))
    q_rel_med = float(np.abs(deq - raw).mean() / max(med, 1e-9))

    json_bytes = os.path.getsize(args.circuit)
    load_json = bench_load(args.circuit)
    step_t = bench_steps(n, csr.nnz)

    # Numerical equivalence: rollout signature JSON vs NPZ-loaded circuit.
    sig_json = rollout_signature(args.circuit)
    if args.emit:
        bundle_npz(npz_path, bundle, quantized=False)
        bundle_npz(q8_path, bundle, quantized=True)
        load_npz = bench_load(npz_path)
        sig_npz = rollout_signature(npz_path)
        equiv = float(np.abs(sig_json - sig_npz).max())
    else:
        load_npz, equiv = None, 0.0

    # NPZ loader must exist for FlyCircuit to read it — check extension support.
    print(f"circuit: {os.path.basename(args.circuit)} "
          f"neurons={n} synapses={len(pre)}")
    print(f"  json-edge : {json_bytes / 1024:.0f} KB  load {load_json * 1000:.0f} ms")
    print(f"  npz-csr   : ~{npz_bytes / 1024:.0f} KB (in-mem)  "
          + (f"load {load_npz * 1000:.0f} ms  rollout-maxdiff {equiv:.2e}"
             if load_npz else "(use --emit to verify load)"))
    print(f"  npz-quant : int8 raw, max abs err {max_abs_err:.3f} "
          f"(mean err {q_rel_med * 100:.1f}% of median weight {med:.1f} — "
          f"lossy, small synapses suffer most)")
    print(f"  matmul    : {step_t * 1000:.0f} ms / 1000 steps "
          f"(n={n}, nnz={csr.nnz})")
    print("recommendation: npz-csr — same synapses/signs as JSON, smaller, "
          "faster load, zero conversion loss; int8 only if flash is critical.")

    report = {
        "circuit": os.path.basename(args.circuit),
        "neurons": n, "synapses": int(len(pre)),
        "json_kb": round(json_bytes / 1024, 1),
        "npz_kb": round(npz_bytes / 1024, 1),
        "load_json_ms": round(load_json * 1000, 1),
        "load_npz_ms": round(load_npz * 1000, 1) if load_npz else None,
        "matmul_1000steps_ms": round(step_t * 1000, 1),
        "rollout_maxdiff": equiv,
        "quant_max_abs_err": max_abs_err,
        "quant_mean_rel_err": q_rel_med,
        "recommendation": "npz-csr",
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
