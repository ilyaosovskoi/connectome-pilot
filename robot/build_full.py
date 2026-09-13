"""build_full.py — convert the WHOLE FlyWire v783 connectome to runtime NPZ.

No extraction, no budget cut: all 139,244 neurons, all ~15M synapses, same
normalization as the small circuits (see robot/formats.py). Output is big
(~300 MB) and stays OUT of git (see .gitignore) — it is a local experiment
artifact, not a deliverable.

Usage:
  python3 robot/build_full.py --out /tmp/fly_full.npz
  # then: FlyCircuit('/tmp/fly_full.npz') — LIF on the whole brain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
from extract_circuit import (  # noqa: E402
    OUTPUT_GROUPS,
    SENSOR_GROUPS,
    build_graph,
    load_annotations,
    match_group,
    sign_by_nt,
)

TARGET_SENSORS = ("odor", "loom")
TARGET_OUTPUTS = ("turn", "fwd", "bwd")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/fly_full.npz")
    ap.add_argument("--sensors", default=",".join(TARGET_SENSORS))
    ap.add_argument("--outputs", default=",".join(TARGET_OUTPUTS))
    args = ap.parse_args()

    t0 = time.time()
    print("annotations...", flush=True)
    ann = load_annotations()
    n = len(ann)
    print(f"cells: {n:,}", flush=True)
    print("graph (parquet)...", flush=True)
    ids, W, pre_all, post_all, w_all = build_graph(ann)
    print(f"matrix: {W.shape}, nnz={W.nnz:,} ({time.time() - t0:.0f}s)", flush=True)

    sign = sign_by_nt(ann)
    side_raw = ann["side"].fillna("").astype(str).str.lower()
    side = np.where(side_raw == "right", 1,
                    np.where(side_raw == "left", 0, -1)).astype(np.int8)
    ntype = ann["cell_type"].astype(str).to_numpy()

    sensor_names = [s.strip() for s in args.sensors.split(",") if s.strip()]
    output_names = [o.strip() for o in args.outputs.split(",") if o.strip()]
    groups = {}
    for name in sensor_names:
        groups[name] = match_group(ann, SENSOR_GROUPS[name]).astype(np.int32)
    for name in output_names:
        groups[name] = match_group(ann, OUTPUT_GROUPS[name]).astype(np.int32)
    print("groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()),
          flush=True)

    # Raw synapse counts in edge order (coo of the summed matrix).
    coo = W.tocoo()
    pre = coo.col.astype(np.int32)
    post = coo.row.astype(np.int32)
    raw = coo.data.astype(np.float32)

    # Same normalization as formats.normalized_csr (signed / postsynaptic sum).
    import scipy.sparse as sp
    signed = raw.astype(np.float64) * sign[pre].astype(np.float64)
    in_sum = np.bincount(post, weights=np.abs(raw).astype(np.float64),
                         minlength=n)
    in_sum[in_sum == 0] = 1.0
    vals = (signed / in_sum[post]).astype(np.float32)
    csr = sp.csr_matrix((vals, (post, pre)), shape=(n, n))

    meta = {"source": "FlyWire v783 full (no budget cut)",
            "neurons": n, "synapses": int(len(pre))}
    payload = {"data": csr.data, "indices": csr.indices, "indptr": csr.indptr,
               "pre": pre, "post": post, "raw": raw,
               "sign": sign.astype(np.int8), "side": side, "ntype": ntype,
               "meta_json": np.array([json.dumps(meta)])}
    for k, v in groups.items():
        payload[f"group_{k}"] = v
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.0f} MB "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
