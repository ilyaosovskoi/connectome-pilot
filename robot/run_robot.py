"""run_robot.py — the connectome as a robot: same brain, several bodies.

Loads a circuit + linear readout (from a train.py checkpoint or freshly fit
on rover demos) and runs it closed-loop on each embodiment in embodiments.py.

Usage:
  python3 robot/run_robot.py --circuit data/fly_circuit_256.npz --bodies rover,arm
  python3 robot/run_robot.py --circuit data/fly_circuit_256.npz --checkpoint /tmp/train2.json
  python3 robot/run_robot.py --circuit data/fly_circuit_256.npz --episodes 10 --max-steps 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodiments import BODIES  # noqa: E402
from eval import BRAIN_STEPS, World, demo_stream, fit_with_gain, tune_expert  # noqa: E402
from flybrain import FlyCircuit  # noqa: E402
from plastic import PlasticFlyCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_brain(args):
    if args.checkpoint:
        with open(args.checkpoint, encoding="utf-8") as fh:
            ck = json.load(fh)
        circuit_path = os.path.join(ROOT, "data", ck.get("circuit", "fly_circuit_256.npz"))
        if not os.path.exists(circuit_path):
            circuit_path = args.circuit
        brain = PlasticFlyCircuit(path=circuit_path, scope=ck.get("scope", "dn-exc"),
                                  norm_power=float(ck.get("norm_power", 1.0)))
        brain.set_readout(ck.get("readout_kind", "all"))
        brain.set_plastic_weights(np.asarray(ck["plastic_weights"], dtype=np.float64))
        if ck.get("homeo_target", 0) > 0:
            brain.enable_homeostasis(target=float(ck["homeo_target"]))
        r = ck["readout"]
        readout = (np.asarray(r["mu"], np.float32), np.asarray(r["sd"], np.float32),
                   np.asarray(r["W"], np.float32))
        if "gain_odor" in ck:
            gains = (float(ck["gain_odor"]), float(ck["gain_loom"]))
        else:
            gains = (float(ck.get("gain", 80.0)),) * 2
        print(f"loaded checkpoint {args.checkpoint} "
              f"(circuit {os.path.basename(circuit_path)}, "
              f"odor x{gains[0]:g} loom x{gains[1]:g})")
        return brain, gains, readout
    brain = FlyCircuit(args.circuit, norm_power=args.norm_power)
    brain.set_readout(args.readout)
    probe = World(8, args.max_steps, seed=100 + args.seed)
    eparams, _ = tune_expert(probe)
    stream = demo_stream(probe, eparams, 3)
    gain, readout, mse, _ = fit_with_gain(brain, stream)
    print(f"fresh readout on rover demos: gain x{gain:g}, clone-MSE {mse:.4f}")
    return brain, (gain, gain), readout


def arm_demos(episodes=8, max_steps=400, seed=100):
    """Expert demonstration stream for the arm (same tick format as eval)."""
    from embodiments import Arm, arm_expert_action
    stream = []
    for ep in range(episodes):
        body = Arm(max_steps=max_steps, seed=seed, episode=ep)
        body.reset()
        done = False
        while not done:
            ol, orr, ll, lr = (np.array([x]) for x in body.obs())
            a = arm_expert_action(body)
            stream.append((ol, orr, ll, lr, np.array([a[0]]), np.array([a[1]])))
            _, _, done, _ = body.step(a[0], a[1])
    return stream


def fit_body_readout(brain, body_name, max_steps, seed):
    """Fit a fresh readout for one body from its own expert demos."""
    if body_name == "arm":
        stream = arm_demos(seed=seed)
    else:
        probe = World(8, max_steps, seed=seed)
        eparams, _ = tune_expert(probe)
        stream = demo_stream(probe, eparams, 3)
    gain, readout, mse, _ = fit_with_gain(brain, stream)
    print(f"[{body_name}] body-specific readout: {len(stream)} demo ticks, "
          f"gain x{gain:g}, clone-MSE {mse:.4f}", flush=True)
    return gain, readout


def run_body(brain, gains, readout, body_name, episodes, max_steps, seed):
    go, gl = (gains if isinstance(gains, (tuple, list)) else (gains, gains))
    mu, sd, W = readout
    cls = BODIES[body_name]
    n_ok, n_crash, prog, total_r = 0, 0, [], 0.0
    for ep in range(episodes):
        body = cls(max_steps=max_steps, seed=seed, episode=ep)
        ol, orr, ll, lr = body.reset()
        brain.reset(batch=1)
        done = False
        while not done:
            ext = brain.sensor_current(
                go, gl, np.array([ol]), np.array([orr]),
                np.array([ll]), np.array([lr]))
            for _ in range(BRAIN_STEPS):
                brain.step(ext)
            X = np.concatenate([(brain.features() - mu) / sd,
                                np.ones((1, 1), np.float32)], axis=1)
            cmd = (X @ W.T)[0]
            (ol, orr, ll, lr), reward, done, info = body.step(
                float(np.clip(cmd[0], -1, 1)), float(np.clip(cmd[1], -1, 1)))
            total_r += reward
        n_ok += info["success"]
        n_crash += info.get("crash", False)
        prog.append(info["progress"])
    return {"body": body_name, "episodes": episodes,
            "success": n_ok / episodes, "crash": n_crash / episodes,
            "progress": float(np.mean(prog)), "mean_reward": total_r / episodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.npz"))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--bodies", default="rover,arm")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--readout", choices=("dn", "all"), default="all")
    ap.add_argument("--norm-power", type=float, default=1.0)
    ap.add_argument("--fit-body", action="store_true",
                    help="fit a separate readout per body from its own expert "
                         "(vs zero-shot transfer of the rover readout)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for b in args.bodies.split(","):
        if b.strip() not in BODIES:
            sys.exit(f"unknown body '{b.strip()}', choose from {sorted(BODIES)}")

    brain, gains, readout = load_brain(args)
    rows = []
    for body_name in args.bodies.split(","):
        body_name = body_name.strip()
        if not body_name:
            continue
        g, r = (fit_body_readout(brain, body_name, args.max_steps, 100 + args.seed)
                if args.fit_body and not args.checkpoint else (gains, readout))
        row = run_body(brain, g, r, body_name,
                       args.episodes, args.max_steps, args.seed)
        rows.append(row)
        print(f"[{body_name:<6}] success={row['success'] * 100:>3.0f}% "
              f"crash={row['crash'] * 100:>3.0f}% progress={row['progress']:.2f} "
              f"mean-reward={row['mean_reward']:.2f}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"circuit": os.path.basename(args.circuit),
                       "checkpoint": args.checkpoint,
                       "rows": rows}, fh, indent=2)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
