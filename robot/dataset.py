"""dataset.py — sensorimotor dataset in NPZ format for connectome training.

Records expert demonstrations as (sensors -> action) pairs plus the frozen
brain's features and dense progress rewards, so any learner (readout,
plastic synapses, external models) can train without re-running the world:

  sensors  (T, 4)  odor_l, odor_r, loom_l, loom_r in [0, ~2]
  actions  (T, 2)  expert left/right wheel commands in [-1, 1]
  features (T, F)  brain firing-rate features at each tick (readout input)
  reward   (T,)    dense progress reward (fraction of distance closed)
  episode  (T,)    episode id per tick
  split    (T,)    0=train 1=test (split by episode seed)

Usage:
  python3 robot/dataset.py --circuit data/fly_circuit_256.npz --out data/sensorimotor.npz
  python3 robot/dataset.py --circuit data/fly_circuit_256.npz --episodes 20 --test-episodes 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import BRAIN_STEPS, World, expert_wheels, tune_expert  # noqa: E402
from flybrain import FlyCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def record_world(circuit, world, eparams, gain, collect_features=True):
    sens, acts, feats, rews, epids = [], [], [], [], []
    m = world.m
    for ep in range(m):
        world.reset()
        # Isolate episode ep by masking others as done.
        world.done[:] = True
        world.done[ep] = False
        circuit.reset(batch=1)
        prev_d = float(np.linalg.norm(world.p[ep] - world.goal[ep]))
        d0 = max(prev_d, 1e-6)
        for _ in range(world.max_steps):
            ol, orr, ll, lr = world.sensors()
            # Single-episode vectors:
            o = (float(ol[ep]), float(orr[ep]))
            l = (float(ll[ep]), float(lr[ep]))
            left, right = expert_wheels(
                np.array([o[0]]), np.array([o[1]]),
                np.array([l[0]]), np.array([l[1]]), *eparams)
            lv, rv = float(left[0]), float(right[0])
            sens.append([o[0], o[1], l[0], l[1]])
            acts.append([lv, rv])
            if collect_features:
                ext = circuit.sensor_current(
                    gain, gain, np.array([o[0]]), np.array([o[1]]),
                    np.array([l[0]]), np.array([l[1]]))
                for _ in range(BRAIN_STEPS):
                    circuit.step(ext)
                feats.append(circuit.features()[0].copy())
            else:
                feats.append(np.zeros(0, dtype=np.float32))
            world.step(np.array([lv if i == ep else 0.0 for i in range(m)]),
                       np.array([rv if i == ep else 0.0 for i in range(m)]))
            d = float(np.linalg.norm(world.p[ep] - world.goal[ep]))
            rews.append((prev_d - d) / d0)
            prev_d = d
            epids.append(ep)
            if bool(world.done[ep]):
                break
        world.done[:] = False
    return (np.asarray(sens, np.float32), np.asarray(acts, np.float32),
            np.asarray(feats, np.float32), np.asarray(rews, np.float32),
            np.asarray(epids, np.int32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.npz"))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--test-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--gain", type=float, default=80.0)
    ap.add_argument("--readout", choices=("dn", "all"), default="all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sensorimotor.npz"))
    args = ap.parse_args()

    circuit = FlyCircuit(args.circuit)
    circuit.set_readout(args.readout)

    probe = World(args.episodes, args.max_steps, seed=100 + args.seed)
    eparams, efit = tune_expert(probe)
    print(f"expert params={eparams} train-fitness={efit:.3f}", flush=True)

    train_w = World(args.episodes, args.max_steps, seed=100 + args.seed)
    test_w = World(args.test_episodes, args.max_steps, seed=200 + args.seed)
    ts, ta, tf, tr, te = record_world(circuit, train_w, eparams, args.gain)
    vs, va, vf, vr, ve = record_world(circuit, test_w, eparams, args.gain)
    ve = ve + args.episodes

    sensors = np.concatenate([ts, vs])
    actions = np.concatenate([ta, va])
    features = np.concatenate([tf, vf])
    reward = np.concatenate([tr, vr])
    episode = np.concatenate([te, ve])
    split = np.concatenate([np.zeros(len(ts), np.int8), np.ones(len(vs), np.int8)])

    meta = {
        "circuit": os.path.basename(args.circuit),
        "readout": args.readout,
        "gain": args.gain,
        "expert_params": list(eparams),
        "train_episodes": args.episodes,
        "test_episodes": args.test_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "columns_sensors": ["odor_l", "odor_r", "loom_l", "loom_r"],
        "columns_actions": ["left", "right"],
    }
    np.savez_compressed(args.out, sensors=sensors, actions=actions,
                        features=features, reward=reward, episode=episode,
                        split=split, meta_json=np.array([json.dumps(meta)]))
    print(f"dataset -> {args.out}: {len(sensors)} ticks "
          f"(train {len(ts)}, test {len(vs)}), "
          f"features F={features.shape[1]}, {os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
