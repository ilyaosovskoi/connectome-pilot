"""train.py — automated training of plastic connectome synapses.

Pipeline (fully automatic, one command):
  1. Tune a Braitenberg expert on the training worlds.
  2. Select input gains (odor x loom) by CLOSED-LOOP validation fitness —
     clone MSE picks the wrong gain (verified: it favors saturating drive).
  3. Fit a linear readout by behavioral cloning (frozen-connectome baseline).
  4. Train plastic synapses with dopamine-style R-STDP (see plastic.py):
     dense progress reward + loom (hazard) shaping + terminal bonus/penalty.
  5. DAgger: collect on-policy states, label with the expert, re-fit readout.
  6. Re-evaluate on held-out test worlds, save best checkpoint.

Usage:
  python3 robot/train.py --circuit data/fly_circuit_256.npz --episodes 12
  python3 robot/train.py --circuit data/fly_circuit_256.npz --readout dn --dagger 2
"""

from __future__ import annotations

import argparse
import itertools
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
    expert_wheels,
    fit_readout,
    tune_expert,
)
from plastic import PlasticFlyCircuit  # noqa: E402
from novelty import NoveltyBonus  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dense progress rewards are ~1e-3 per step; scale so RPE lands in the
# 0.05..1.0 range where R-STDP actually moves synapse counts (3..30).
REWARD_SCALE = 50.0
GAIN_GRID = (10.0, 20.0, 40.0, 80.0)


def replay_split(circuit, stream, go, gl, stride=1):
    """Brain features for a sensor stream under per-modality gains."""
    circuit.reset(batch=len(stream[0][0]))
    feats, targets = [], []
    for i, (ol, orr, ll, lr, left, right) in enumerate(stream):
        ext = circuit.sensor_current(go, gl, ol, orr, ll, lr)
        for _ in range(BRAIN_STEPS):
            circuit.step(ext)
        if i % stride:
            continue
        feats.append(circuit.features().copy())
        targets.append(np.stack([left, right], 1))
    return np.concatenate(feats), np.concatenate(targets)


def drive(circuit, gains, readout, ol, orr, ll, lr, m):
    """One closed-loop control step. Returns (left, right)."""
    go, gl = gains
    mu, sd, W = readout
    ext = circuit.sensor_current(go, gl, ol, orr, ll, lr)
    for _ in range(BRAIN_STEPS):
        circuit.step(ext)
    X = np.concatenate([(circuit.features() - mu) / sd,
                        np.ones((m, 1), np.float32)], axis=1)
    cmd = X @ W.T
    return np.clip(cmd[:, 0], -1, 1), np.clip(cmd[:, 1], -1, 1)


def rollout_split(circuit, world, gains, readout, max_steps, reps=1, collect=False):
    """Closed-loop eval under per-modality gains. Optionally records states."""
    outs, stream = [], []
    for _ in range(reps):
        circuit.reset(batch=world.m)
        world.reset()
        for _ in range(max_steps):
            ol, orr, ll, lr = world.sensors()
            if collect:
                stream.append((ol, orr, ll, lr))
            left, right = drive(circuit, gains, readout, ol, orr, ll, lr, world.m)
            if world.step(left, right).all():
                break
        outs.append(world.result())
    res = {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}
    return (res, stream) if collect else res


def fit_for_gains(circuit, stream, gains):
    go, gl = gains
    feats, targets = replay_split(circuit, stream, go, gl)
    mu, sd, W = fit_readout(feats, targets)
    X = np.concatenate([(feats - mu) / sd, np.ones((len(feats), 1), np.float32)], 1)
    mse = float(((X @ W.T - targets) ** 2).mean())
    return (mu, sd, W), mse


def select_gains(circuit, stream, val, max_steps, grid=None):
    """Pick (odor, loom) gains by closed-loop validation fitness."""
    grid = grid or GAIN_GRID
    best = None
    for go, gl in itertools.product(grid, grid):
        readout, mse = fit_for_gains(circuit, stream, (go, gl))
        res = rollout_split(circuit, val, (go, gl), readout, max_steps)
        print(f"    gains odor x{go:g} loom x{gl:g}: clone-MSE {mse:.4f} "
              f"val fitness {res['fitness']:.3f} "
              f"(success {res['success']:.0%}, crash {res['crash']:.0%})", flush=True)
        if best is None or res["fitness"] > best[0]:
            best = (res["fitness"], (go, gl), readout, mse, res)
    return best[1], best[2], best[3], best[4]


def rollout_plastic(circuit, world, gains, readout, max_steps, train: bool,
                    crash_coef: float = 0.5, imitate_coef: float = 0.0,
                    eparams=None, episodic: bool = True,
                    novelty=None, novelty_coef: float = 0.0):
    """Single-episode loop. Returns (result, episodic return, parts).

    Reward channels (logged separately, learned jointly):
      task    — progress + hazard shaping + teacher imitation + terminal
                bonus. This is "what is right" (PAM) and "what is wrong" (PPL1).
      novelty — 1/sqrt(visits) for the sensor state. Curiosity only; it can
                never outshout the task because the critic (V_baseline)
                adapts to its mean and only deviations teach.

    episodic=True (default): accumulate eligibility all episode, ONE
    REINFORCE-style update at the end (see PlasticFlyCircuit.apply_episodic).
    episodic=False: per-step dopamine updates (noisy on slow robots).
    """
    go, gl = gains
    mu, sd, W = readout
    circuit.reset(batch=world.m)
    circuit.reset_trace()
    if episodic:
        # Episodic credit assignment needs episode-long memory: coincidences
        # from the first seconds must survive until the final update.
        saved_decay, circuit.decay = circuit.decay, 0.999
    world.reset()
    prev_d = np.linalg.norm(world.p - world.goal, axis=1)
    d0 = np.maximum(prev_d, 1e-6)
    ep_task, ep_novel = 0.0, 0.0
    for _ in range(max_steps):
        ol, orr, ll, lr = world.sensors()
        ext = circuit.sensor_current(go, gl, ol, orr, ll, lr)
        for _ in range(BRAIN_STEPS):
            circuit.step(ext)
        X = np.concatenate([(circuit.features() - mu) / sd,
                            np.ones((world.m, 1), np.float32)], axis=1)
        cmd = X @ W.T
        left, right = np.clip(cmd[:, 0], -1, 1), np.clip(cmd[:, 1], -1, 1)
        world.step(left, right)
        d = np.linalg.norm(world.p - world.goal, axis=1)
        # TASK channel: progress + hazard shaping + teacher imitation.
        loom = np.stack([np.asarray(ll), np.asarray(lr)], 1).mean()
        r_task = ((prev_d - d) / d0).astype(np.float64) - crash_coef * float(loom) / 100.0
        if imitate_coef > 0 and eparams is not None:
            el, er = expert_wheels(ol, orr, ll, lr, *eparams)
            imit = 1.0 - (np.abs(left - el) + np.abs(right - er)) / 4.0
            r_task = r_task + imitate_coef * imit
        # NOVELTY channel: curiosity about unvisited sensor states.
        r_novel = 0.0
        if novelty is not None and novelty_coef > 0 and world.m == 1:
            r_novel = novelty.bonus(ol[0], orr[0], ll[0], lr[0])
        prev_d = d
        if train:
            circuit.observe(circuit.spikes, circuit.rate, circuit.spikes)
            if not episodic:
                circuit.apply_reward((float(r_task.mean())
                                      + novelty_coef * r_novel) * REWARD_SCALE)
            ep_task += float(r_task.mean())
            ep_novel += r_novel
        if bool(world.done.all()):
            break
    res = world.result()
    if episodic:
        circuit.decay = saved_decay
    rpe = 0.0
    if train:
        bonus = 1.0 if bool(world.success.all()) else (
            -1.0 if bool(world.crash.all()) else 0.0)
        total = (ep_task + novelty_coef * ep_novel) * REWARD_SCALE + bonus * REWARD_SCALE
        if episodic:
            rpe = circuit.apply_episodic(total)
        else:
            rpe = circuit.apply_reward(total)
    parts = {"task": ep_task, "novelty": ep_novel,
             "pam": max(rpe, 0.0), "ppl": max(-rpe, 0.0)}
    return res, ep_task + novelty_coef * ep_novel, parts


def label_stream(extra, eparams):
    out = []
    for ol, orr, ll, lr in extra:
        left, right = expert_wheels(ol, orr, ll, lr, *eparams)
        out.append((ol, orr, ll, lr, left, right))
    return out


def report_synapses(brain, top: int = 5):
    """Proof that conductance changed: most strengthened/weakened synapses.

    Prints pre-type -> post-type, raw count before -> after (x factor).
    Raw count / postsynaptic input sum IS the effective conductance, so a
    factor here is a conductance change, not a bookkeeping number.
    """
    w1 = brain.get_plastic_weights()
    w0 = brain.w_init
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(w0 > 0, w1 / np.maximum(w0, 1e-9), 1.0)
    order = np.argsort(factor)
    types = getattr(brain, "types", [])
    def name(i):
        return types[int(i)] if types and int(i) < len(types) else f"n{int(i)}"
    changed = int((np.abs(factor - 1.0) > 0.01).sum())
    print(f"plastic synapses changed >1%: {changed}/{len(w0)}", flush=True)
    for tag, idx in (("strengthened", order[-top:][::-1]),
                     ("weakened", order[:top])):
        rows = []
        for k in idx:
            if abs(factor[int(k)] - 1.0) <= 0.01:
                continue
            ei = brain.plastic_idx[int(k)]
            rows.append(f"{name(brain.edge_pre[ei])}->{name(brain.edge_post[ei])} "
                        f"{w0[int(k)]:.0f}->{w1[int(k)]:.0f} (x{factor[int(k)]:.2f})")
        if rows:
            print(f"  {tag}: " + "; ".join(rows), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default=os.path.join(ROOT, "data", "fly_circuit_256.npz"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--test-episodes", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--demo-rollouts", type=int, default=3)
    ap.add_argument("--dagger", type=int, default=1,
                    help="on-policy collect+label+refit rounds after plasticity")
    ap.add_argument("--gains", type=str, default="10,20,40,80",
                    help="comma-separated gain grid (used for odor x loom)")
    ap.add_argument("--crash-coef", type=float, default=0.5,
                    help="weight of dense loom (hazard) penalty in step reward")
    ap.add_argument("--imitate", type=float, default=0.0,
                    help="dense teacher bonus for matching expert wheels each step")
    ap.add_argument("--novelty", type=float, default=0.0,
                    help="curiosity weight: bonus 1/sqrt(visits) per sensor state. "
                         "Explores; the task channel still defines right vs wrong.")
    ap.add_argument("--homeo-target", type=float, default=5.0,
                    help="homeostatic target rate (Hz-ish); 0 disables. Wakes "
                         "silent descending neurons so plasticity can reach them.")
    ap.add_argument("--scope", choices=("dn-exc", "all-exc", "all"), default="dn-exc")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--readout", choices=("dn", "all"), default="dn")
    ap.add_argument("--norm-power", type=float, default=1.0,
                    help="drive normalization power: 1.0 linear fraction, "
                         "0.5 sublinear (wakes hub neurons like DNa)")
    ap.add_argument("--heading-noise", type=float, default=np.pi,
                    help="target initial-heading noise (rad). With --curriculum, "
                         "ramps up to this value.")
    ap.add_argument("--curriculum", type=str, default="",
                    help="comma-separated heading-noise stages, e.g. '0,1.5,3.14'. "
                         "Plastic episodes split evenly across stages; selection "
                         "always on the final (target) noise.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    target_hn = args.heading_noise
    stages = [float(x) for x in args.curriculum.split(",") if x.strip()] or [target_hn]
    hn = stages[-1]  # selection and eval always run on the target task
    rng = np.random.default_rng(args.seed)
    train_seeds = rng.integers(0, 10_000, size=args.episodes)
    test = World(args.test_episodes, args.max_steps, seed=200 + args.seed,
                 heading_noise=hn)
    val = World(8, args.max_steps, seed=300 + args.seed, heading_noise=hn)

    # 1. Expert + demonstrations.
    probe = World(args.episodes, args.max_steps, seed=100 + args.seed,
                  heading_noise=hn)
    eparams, efit = tune_expert(probe)
    print(f"expert: k_odor={eparams[0]} k_loom={eparams[1]} v={eparams[2]} "
          f"train-fitness={efit:.3f}", flush=True)
    stream = list(demo_stream(probe, eparams, args.demo_rollouts))
    print(f"demos: {len(stream)} ticks x {probe.m} robots", flush=True)

    brain = PlasticFlyCircuit(path=args.circuit, scope=args.scope, lr=args.lr,
                              norm_power=args.norm_power)
    brain.set_readout(args.readout)
    if args.homeo_target > 0:
        brain.enable_homeostasis(target=args.homeo_target)
        print(f"homeostasis on (target {args.homeo_target:g} Hz)", flush=True)

    # 2. Gain selection by closed-loop validation (NOT clone MSE).
    print("gain selection (closed-loop val):", flush=True)
    grid = tuple(float(x) for x in args.gains.split(",") if x.strip())
    gains, readout, mse, _ = select_gains(brain, stream, val, args.max_steps,
                                          grid=grid)
    frozen_res = rollout_split(brain, test, gains, readout, args.max_steps, reps=2)
    print(f"frozen baseline: success={frozen_res['success']:.0%} "
          f"crash={frozen_res['crash']:.0%} fitness={frozen_res['fitness']:.3f} "
          f"(odor x{gains[0]:g} loom x{gains[1]:g}, clone-MSE {mse:.4f})", flush=True)

    # 3. Plastic episodes with validation-based model selection.
    # With a curriculum, episodes are split across noise stages (easy first);
    # validation always measures the TARGET task.
    best_fit, best_w, history = frozen_res["fitness"], None, []
    per_stage = max(1, args.episodes // len(stages))
    novel = NoveltyBonus() if args.novelty > 0 else None
    for ep in range(args.episodes):
        stage_hn = stages[min(ep // per_stage, len(stages) - 1)]
        w = World(1, args.max_steps, seed=int(train_seeds[ep]),
                  heading_noise=stage_hn)
        res, eret, parts = rollout_plastic(
            brain, w, gains, readout, args.max_steps, train=True,
            crash_coef=args.crash_coef, imitate_coef=args.imitate,
            eparams=eparams, novelty=novel, novelty_coef=args.novelty)
        entry = {"episode": ep, "stage_hn": stage_hn,
                 "fitness": res["fitness"], "task": round(parts["task"], 3),
                 "novelty": round(parts["novelty"], 3),
                 "pam": round(parts["pam"], 3), "ppl": round(parts["ppl"], 3)}
        if (ep + 1) % args.val_every == 0 or ep == args.episodes - 1:
            v = rollout_split(brain, val, gains, readout, args.max_steps)
            entry.update(val_fitness=v["fitness"], val_success=v["success"])
            if v["fitness"] > best_fit:
                best_fit = v["fitness"]
                best_w = brain.get_plastic_weights().copy()
                entry["best"] = True
        history.append(entry)
        msg = (f"  train ep {ep + 1}/{args.episodes}: fitness={res['fitness']:.3f} "
               f"task={parts['task']:+.2f} nov={parts['novelty']:.1f} "
               f"PAM={parts['pam']:.2f}/PPL={parts['ppl']:.2f}")
        if "val_fitness" in entry:
            msg += f" val={entry['val_fitness']:.3f}" + (" *BEST*" if entry.get("best") else "")
        print(msg, flush=True)
    if best_w is not None:
        brain.set_plastic_weights(best_w)
        print(f"restored best weights (val {best_fit:.3f})", flush=True)
    else:
        print("no val improvement over frozen; keeping final weights", flush=True)

    # 4. DAgger: on-policy states labeled by expert. Accept each round ONLY
    # on validation improvement — a blind refit can hurt (verified).
    for it in range(args.dagger):
        _, extra = rollout_split(brain, probe, gains, readout,
                                 args.max_steps, collect=True)
        stream = stream + label_stream(extra, eparams)
        candidate, mse = fit_for_gains(brain, stream, gains)
        v = rollout_split(brain, val, gains, candidate, args.max_steps)
        if v["fitness"] > best_fit:
            best_fit = v["fitness"]
            readout = candidate
            print(f"  dagger {it + 1}/{args.dagger}: data={len(stream)} ticks "
                  f"val fitness={v['fitness']:.3f} (MSE {mse:.4f}) *ACCEPTED*",
                  flush=True)
        else:
            # Roll back the data too: bad states stay out.
            stream = stream[:-len(extra)]
            print(f"  dagger {it + 1}/{args.dagger}: val fitness={v['fitness']:.3f} "
                  f"<= best {best_fit:.3f}, round rejected", flush=True)

    tuned_res = rollout_split(brain, test, gains, readout, args.max_steps, reps=2)
    print(f"tuned: success={tuned_res['success']:.0%} crash={tuned_res['crash']:.0%} "
          f"fitness={tuned_res['fitness']:.3f}", flush=True)
    print(f"delta fitness: {tuned_res['fitness'] - frozen_res['fitness']:+.3f} "
          f"in {time.time() - t0:.0f}s", flush=True)
    report_synapses(brain)
    if novel is not None:
        print(f"novelty coverage: {novel.coverage()} distinct sensor states",
              flush=True)

    if args.out:
        mu, sd, W = readout
        payload = {
            "circuit": os.path.basename(args.circuit),
            "scope": args.scope, "lr": args.lr,
            "readout_kind": args.readout,
            "gain_odor": gains[0], "gain_loom": gains[1],
            "dagger_rounds": args.dagger, "crash_coef": args.crash_coef,
            "imitate": args.imitate, "novelty": args.novelty,
            "norm_power": args.norm_power,
            "homeo_target": args.homeo_target,
            "plastic_weights": brain.get_plastic_weights().tolist(),
            "readout": {"mu": mu.tolist(), "sd": sd.tolist(), "W": W.tolist()},
            "expert_params": list(eparams),
            "frozen": {k: frozen_res[k] for k in ("success", "crash", "fitness")},
            "tuned": {k: tuned_res[k] for k in ("success", "crash", "fitness")},
            "history": history, "seed": args.seed,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
