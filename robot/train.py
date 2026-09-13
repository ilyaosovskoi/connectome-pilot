"""train.py — automated training of plastic connectome synapses.

Pipeline (fully automatic, one command):
  1. Tune a Braitenberg expert on the training worlds.
  2. Fit a linear readout by behavioral cloning (frozen-connectome baseline).
  3. Evaluate the frozen baseline on held-out test worlds.
  4. Train plastic synapses with dopamine-style R-STDP (see plastic.py):
     dense step reward (progress toward goal) + terminal bonus/penalty.
  5. Re-fit the readout on top of the tuned connectome, re-evaluate.
  6. Save the best checkpoint (plastic weights + readout + report).

Usage:
  python3 robot/train.py --circuit data/fly_circuit_256.json --episodes 12
  python3 robot/train.py --circuit data/fly_circuit_256.json --scope all-exc --lr 0.02 --out data/trained_256.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import (  # noqa: E402
    BRAIN_STEPS,
    World,
    demo_stream,
    fit_with_gain,
    rollout,
    run_expert,
    tune_expert,
)
from plastic import PlasticFlyCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dense progress rewards are ~1e-3 per step; scale so RPE lands in the
# 0.05..1.0 range where R-STDP actually moves synapse counts (3..30).
REWARD_SCALE = 50.0


def rollout_with_plasticity(circuit, world, gain, readout, train: bool,
                            max_steps: int):
    """Single-episode closed loop. Returns (result, mean_step_reward).

    When train=True, accumulates eligibility every control step and applies
    a dense progress reward; terminal success/crash gives a bonus/penalty.
    """
    mu, sd, W = readout
    circuit.reset(batch=world.m)
    circuit.reset_trace()
    world.reset()
    prev_d = np.linalg.norm(world.p - world.goal, axis=1)
    d0 = np.maximum(prev_d, 1e-6)
    step_rewards = []
    for _ in range(max_steps):
        ol, orr, ll, lr = world.sensors()
        ext = circuit.sensor_current(gain, gain, ol, orr, ll, lr)
        for _ in range(BRAIN_STEPS):
            circuit.step(ext)
        X = np.concatenate([(circuit.features() - mu) / sd,
                            np.ones((world.m, 1), np.float32)], axis=1)
        cmd = X @ W.T
        left = np.clip(cmd[:, 0], -1, 1)
        right = np.clip(cmd[:, 1], -1, 1)
        world.step(left, right)
        d = np.linalg.norm(world.p - world.goal, axis=1)
        # Dense reward: fraction of initial distance closed this step.
        r = ((prev_d - d) / d0).astype(np.float64)
        prev_d = d
        if train:
            circuit.observe(circuit.spikes, circuit.rate)
            circuit.apply_reward(float(r.mean()) * REWARD_SCALE)
            step_rewards.append(float(r.mean()))
        if bool(world.done.all()):
            break
    res = world.result()
    bonus = 0.0
    if train:
        if bool(world.success.all()):
            bonus = 1.0
        elif bool(world.crash.all()):
            bonus = -1.0
        circuit.apply_reward(bonus * REWARD_SCALE)
    return res, (float(np.mean(step_rewards)) if step_rewards else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.json"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--test-episodes", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--demo-rollouts", type=int, default=3)
    ap.add_argument("--scope", choices=("dn-exc", "all-exc", "all"), default="dn-exc")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--val-every", type=int, default=2,
                    help="validate on held-out worlds every K episodes, keep best")
    ap.add_argument("--readout", choices=("dn", "all"), default="all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    train_seeds = rng.integers(0, 10_000, size=args.episodes)
    test = World(args.test_episodes, args.max_steps, seed=200 + args.seed)

    # 1. Expert + demonstrations (shared, fixed).
    probe = World(args.episodes, args.max_steps, seed=100 + args.seed)
    eparams, efit = tune_expert(probe)
    print(f"expert: k_odor={eparams[0]} k_loom={eparams[1]} v={eparams[2]} "
          f"train-fitness={efit:.3f}", flush=True)
    stream = demo_stream(probe, eparams, args.demo_rollouts)
    print(f"demos: {len(stream)} ticks x {probe.m} robots", flush=True)

    # 2-3. Frozen baseline: fit readout, evaluate.
    base = PlasticFlyCircuit(path=args.circuit, scope=args.scope, lr=args.lr)
    base.set_readout(args.readout)
    gain, readout, mse, _ = fit_with_gain(base, stream)
    frozen_res = rollout(base, test, gain, readout, reps=2)
    print(f"frozen baseline: success={frozen_res['success']:.2f} "
          f"crash={frozen_res['crash']:.2f} fitness={frozen_res['fitness']:.3f} "
          f"(gain x{gain:g}, clone-MSE {mse:.4f})", flush=True)

    # 4. Plastic training on fresh single-episode worlds, with held-out
    # validation for model selection (train-episode fitness is not comparable
    # to test fitness — different worlds).
    val = World(4, args.max_steps, seed=300 + args.seed)
    best_fit, best_w, history = frozen_res["fitness"], None, []
    for ep in range(args.episodes):
        w = World(1, args.max_steps, seed=int(train_seeds[ep]))
        res, rbar = rollout_with_plasticity(base, w, gain, readout,
                                            train=True, max_steps=args.max_steps)
        entry = {"episode": ep, "fitness": res["fitness"],
                 "success": res["success"], "crash": res["crash"],
                 "mean_step_r": rbar}
        if (ep + 1) % args.val_every == 0 or ep == args.episodes - 1:
            v = rollout(base, val, gain, readout, reps=1)
            entry["val_fitness"] = v["fitness"]
            entry["val_success"] = v["success"]
            if v["fitness"] > best_fit:
                best_fit = v["fitness"]
                best_w = base.get_plastic_weights().copy()
                entry["best"] = True
        history.append(entry)
        msg = (f"  train ep {ep + 1}/{args.episodes}: fitness={res['fitness']:.3f} "
               f"rbar={rbar:+.4f}")
        if "val_fitness" in entry:
            msg += f" val={entry['val_fitness']:.3f}"
            msg += " *BEST*" if entry.get("best") else ""
        print(msg, flush=True)
    if best_w is not None:
        base.set_plastic_weights(best_w)
        print(f"restored best validation weights (val fitness {best_fit:.3f})",
              flush=True)
    else:
        print("no validation improvement over frozen baseline; "
              "keeping final weights", flush=True)

    # 5. Re-fit readout on tuned connectome, re-evaluate.
    gain2, readout2, mse2, _ = fit_with_gain(base, stream)
    tuned_res = rollout(base, test, gain2, readout2, reps=2)
    print(f"tuned connectome: success={tuned_res['success']:.2f} "
          f"crash={tuned_res['crash']:.2f} fitness={tuned_res['fitness']:.3f} "
          f"(gain x{gain2:g}, clone-MSE {mse2:.4f})", flush=True)
    print(f"delta fitness: {tuned_res['fitness'] - frozen_res['fitness']:+.3f} "
          f"in {time.time() - t0:.0f}s", flush=True)

    if args.out:
        mu, sd, W = readout2
        payload = {
            "circuit": os.path.basename(args.circuit),
            "scope": args.scope,
            "lr": args.lr,
            "readout_kind": args.readout,
            "gain": gain2,
            "plastic_weights": base.get_plastic_weights().tolist(),
            "readout": {"mu": mu.tolist(), "sd": sd.tolist(), "W": W.tolist()},
            "expert_params": list(eparams),
            "frozen": {k: v for k, v in frozen_res.items()
                       if k in ("success", "crash", "fitness", "progress")},
            "tuned": {k: v for k, v in tuned_res.items()
                      if k in ("success", "crash", "fitness", "progress")},
            "history": history,
            "seed": args.seed,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
