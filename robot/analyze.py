"""analyze.py — understand the brain before training it.

Answers four questions on fixed seeds (reproducible):
  1. Which input gain clones the expert best AND drives best closed-loop?
  2. Which sense matters: what breaks when odor or loom input is zeroed?
  3. Which output neurons carry the behavior (top readout weights)?
  4. How active is each neuron group under expert drive?

Usage:
  python3 robot/analyze.py --circuit data/fly_circuit_256.npz --out /tmp/analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import BRAIN_STEPS, World, demo_stream, fit_readout, replay, tune_expert  # noqa: E402
from flybrain import FlyCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAINS = (10.0, 20.0, 40.0, 80.0)


def closed_loop(circuit, world, gain, readout, max_steps):
    from eval import rollout
    return rollout(circuit, world, gain, readout, reps=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.npz"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--test-episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--readout", choices=("dn", "all"), default="all")
    ap.add_argument("--norm-power", type=float, default=1.0,
                    help="drive normalization power: 1.0 linear fraction, "
                         "0.5 sublinear (wakes hub neurons like DNa)")
    ap.add_argument("--heading-noise", type=float, default=0.0,
                    help="uniform initial-heading noise (rad); pi = random heading")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train_w = World(args.episodes, args.max_steps, seed=100 + args.seed,
                    heading_noise=args.heading_noise)
    test_w = World(args.test_episodes, args.max_steps, seed=200 + args.seed,
                   heading_noise=args.heading_noise)
    eparams, _ = tune_expert(train_w)
    stream = demo_stream(train_w, eparams, 3)

    # 1. Gain sweep.
    print(f"{'gain':>6} {'cloneMSE':>9} {'success':>8} {'crash':>7} {'fitness':>8}")
    rows = []
    readouts = {}
    for g in GAINS:
        c = FlyCircuit(args.circuit, norm_power=args.norm_power)
        c.set_readout(args.readout)
        feats, targets = replay(c, stream, g)
        readout = fit_readout(feats, targets)
        mu, sd, W = readout
        X = np.concatenate([(feats - mu) / sd,
                            np.ones((len(feats), 1), np.float32)], axis=1)
        mse = float(((X @ W.T - targets) ** 2).mean())
        res = closed_loop(c, test_w, g, readout, args.max_steps)
        readouts[g] = readout
        rows.append({"gain": g, "mse": mse,
                     "success": res["success"], "crash": res["crash"],
                     "fitness": res["fitness"]})
        print(f"{g:>6.0f} {mse:>9.4f} {res['success']:>7.0%} {res['crash']:>6.0%} "
              f"{res['fitness']:>8.3f}", flush=True)
    # Clone MSE via fit_with_gain internals (recompute cheaply):
    from eval import fit_with_gain
    c0 = FlyCircuit(args.circuit, norm_power=args.norm_power)
    c0.set_readout(args.readout)
    best_gain, _, best_mse, _ = fit_with_gain(c0, stream)
    print(f"best clone gain: x{best_gain:g} (MSE {best_mse:.4f})")
    best = max(rows, key=lambda r: r["fitness"])
    print(f"best closed-loop gain: x{best['gain']:.0f} "
          f"(fitness {best['fitness']:.3f})")

    # 2. Modality ablation at best gain.
    g = best["gain"]
    readout = readouts[g]
    base_c = FlyCircuit(args.circuit, norm_power=args.norm_power)
    base_c.set_readout(args.readout)
    base = closed_loop(base_c, test_w, g, readout, args.max_steps)
    # NOTE: fresh circuit each time (rollout mutates state, reset inside anyway).
    results = {"full": {k: base[k] for k in ("success", "crash", "fitness")}}
    for knocked in ("odor", "loom"):
        c = FlyCircuit(args.circuit, norm_power=args.norm_power)
        c.set_readout(args.readout)
        # Zero the knocked group by emptying its index lists for this run.
        saved = {k: c.groups[k] for k in (f"{knocked}",) if knocked in c.groups}
        for k in saved:
            c.groups[k] = np.array([], dtype=np.int32)
        c.group_left = {k: v[c.side[v] == 0] for k, v in c.groups.items()}
        c.group_right = {k: v[c.side[v] == 1] for k, v in c.groups.items()}
        r = closed_loop(c, test_w, g, readout, args.max_steps)
        results[f"no_{knocked}"] = {k: r[k] for k in ("success", "crash", "fitness")}
        for k in saved:
            c.groups[k] = saved[k]
    print("ablation:", json.dumps(results))

    # 3. Top readout weights -> neuron types.
    mu, sd, W = readout
    c = FlyCircuit(args.circuit, norm_power=args.norm_power)
    c.set_readout(args.readout)
    feat_idx = c.feat_idx
    order = np.argsort(-np.abs(W).mean(axis=0)[:-1])[:5]
    tops = []
    for j in order:
        ni = int(feat_idx[int(j)])
        tops.append({"neuron": ni, "type": c.types[ni],
                     "w_left": float(W[0][j]), "w_right": float(W[1][j])})
    print("top readout neurons:", json.dumps(tops))

    # 4. Group firing rates under expert drive.
    c.reset(batch=1)
    acc, cnt = {}, {}
    for i, (ol, orr, ll, lr, _, _) in enumerate(stream[:200]):
        ext = c.sensor_current(g, g, ol[:1], orr[:1], ll[:1], lr[:1])
        for _ in range(BRAIN_STEPS):
            c.step(ext)
    rates = {}
    for name in ("odor", "loom", "turn", "fwd", "bwd"):
        idx = c.groups.get(name, np.array([], dtype=np.int32))
        rates[name] = float(c.rate[:, idx].mean()) if len(idx) else 0.0
    print("mean rates:", json.dumps({k: round(v, 4) for k, v in rates.items()}))

    report = {"circuit": os.path.basename(args.circuit), "gains": rows,
              "best_gain_clone": best_gain, "best_gain_closed": best["gain"],
              "ablation": results, "top_neurons": tops, "rates": rates,
              "expert": list(eparams)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
