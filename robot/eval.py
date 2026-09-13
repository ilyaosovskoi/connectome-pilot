"""eval.py — robot with a fly brain: seeks an odor source, avoids obstacles.

Idea. The connectome is NOT trained: all 8,264 synapses are real and frozen.
The brain acts as a reservoir; only a tiny readout from the real motor output
— 12 descending neurons (DNa01/DNa02 for turning, DNp09 forward, MDN backward)
to two wheels — is learned. 26 numbers. This mirrors the living brain, which
talks to the body only through descending neurons.

Robot task (differential drive, 20x20 m arena):
  - reach the odor source — ORN_* (smell) sensors;
  - don't crash — LPLC2/LC4 (looming/shadow) sensors; on the robot these are
    range-finder proxies.

Ablation ladder — same setup, same training, same episodes; only the wiring
inside the reservoir changes:

  real         — real FlyWire v783 synapses (data/fly_circuit.json);
  shuffled     — same neurons, weights and fan-out, but shuffled targets;
  random       — same weights, but random sources and targets (degrees broken);
  braitenberg  — two lines of hand-written chemotaxis, no network at all;
  nobrain      — random walk.

The readout is trained by behavioral cloning of a Braitenberg expert (ridge
regression), identically for every condition. If real wiring learns better at
equal size and equal data, the wiring is what makes the difference.

Run:
  python3 robot/eval.py
  python3 robot/eval.py --circuits data/fly_circuit_256.json,data/fly_circuit.json
  python3 robot/eval.py --episodes 12 --generations 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flybrain import FlyCircuit, RandomCircuit, ShuffledCircuit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CIRCUIT = os.path.join(ROOT, "data", "fly_circuit.json")

ARENA = 20.0
DT = 0.05                 # с, шаг управления
BRAIN_STEPS = 10          # шагов LIF на шаг управления (шаг мозга 5 мс)
V_MAX = 1.0               # м/с
OMEGA_MAX = 3.0           # рад/с
ANTENNA_ANGLE = 25.0      # градусы отклонения антенн от курса
ANTENNA_LEN = 0.45        # м
GOAL_R = 1.0              # радиус «дошёл»
N_OBST = 4
OBST_R = 1.6
PROX_RANGE = 3.5          # м — дальность «зрения»
GAIN_GRID = (10.0, 20.0, 40.0, 80.0)   # сетка калибровки входа


class World:
    """Векторизованный мир: сразу M эпизодов, чтобы прогоны шли пачками."""

    def __init__(self, episodes: int, max_steps: int, seed: int = 0,
                 heading_noise: float = 0.0):
        rng = np.random.default_rng(seed)
        self.m = episodes
        self.max_steps = max_steps
        self.heading_noise = float(heading_noise)
        self._rng = np.random.default_rng(seed + 999)
        self.start = rng.uniform(2.0, ARENA - 2.0, size=(episodes, 2))
        self.goal = rng.uniform(3.0, ARENA - 3.0, size=(episodes, 2))
        far = np.linalg.norm(self.start - self.goal, axis=1) < 8.0
        while far.any():
            self.goal[far] = rng.uniform(3.0, ARENA - 3.0, size=(int(far.sum()), 2))
            far = np.linalg.norm(self.start - self.goal, axis=1) < 8.0
        self.obst = np.stack([
            np.stack([rng.uniform(2.0, ARENA - 2.0, episodes),
                      rng.uniform(2.0, ARENA - 2.0, episodes)], axis=1)
            for _ in range(N_OBST)], axis=1)
        for k in range(N_OBST):
            bad = (np.linalg.norm(self.obst[:, k] - self.start, axis=1) < OBST_R + 1.5) \
                | (np.linalg.norm(self.obst[:, k] - self.goal, axis=1) < OBST_R + 2.0)
            while bad.any():
                nb = int(bad.sum())
                self.obst[bad, k] = np.stack([rng.uniform(2.0, ARENA - 2.0, nb),
                                              rng.uniform(2.0, ARENA - 2.0, nb)], axis=1)
                bad = (np.linalg.norm(self.obst[:, k] - self.start, axis=1) < OBST_R + 1.5) \
                    | (np.linalg.norm(self.obst[:, k] - self.goal, axis=1) < OBST_R + 2.0)
        self.wind = rng.normal(size=(episodes, 2))
        self.wind /= np.linalg.norm(self.wind, axis=1, keepdims=True)
        self.reset()

    def reset(self):
        self.p = self.start.copy()
        self.th = np.arctan2(self.goal[:, 1] - self.p[:, 1], self.goal[:, 0] - self.p[:, 0])
        if self.heading_noise > 0:
            self.th = self.th + self._rng.uniform(-self.heading_noise,
                                                  self.heading_noise, size=self.m)
        self.steps = np.zeros(self.m, dtype=np.int64)
        self.success = np.zeros(self.m, dtype=bool)
        self.crash = np.zeros(self.m, dtype=bool)
        self.done = np.zeros(self.m, dtype=bool)          # финишировавшие замирают
        self.d0 = np.linalg.norm(self.p - self.goal, axis=1)
        self.dmin = self.d0.copy()
        self.prev_prox = np.zeros((self.m, 2), dtype=np.float32)

    # ── сенсоры ──

    def _antennae(self):
        off = np.radians(ANTENNA_ANGLE)
        out = []
        for ang in (self.th + off, self.th - off):
            out.append(self.p + np.stack([np.cos(ang), np.sin(ang)], 1) * ANTENNA_LEN)
        return out[0], out[1]

    def _odor_at(self, pts):
        d = np.linalg.norm(pts - self.goal, axis=1)
        v = self.goal - pts
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        align = np.clip((v * self.wind).sum(1), -1, 1)
        return np.exp(-d / 5.0) * (0.55 + 0.45 * align)

    def _obst_dist(self, pts):
        """Расстояние от точки до поверхности каждого препятствия (m, K)."""
        return np.linalg.norm(pts[:, None, :] - self.obst, axis=2) - OBST_R

    def _prox_at(self, pts):
        """Близость препятствия: 1 — вплотную, 0 — дальше PROX_RANGE."""
        d = self._obst_dist(pts)
        wall = np.minimum.reduce([pts[:, 0], ARENA - pts[:, 0], pts[:, 1], ARENA - pts[:, 1]])
        d = np.concatenate([d, wall[:, None]], axis=1)
        return np.clip(1.0 - d.min(axis=1) / PROX_RANGE, 0.0, 1.0)

    def sensors(self):
        pl, pr = self._antennae()
        odor_l, odor_r = self._odor_at(pl), self._odor_at(pr)
        prox = np.stack([self._prox_at(pl), self._prox_at(pr)], 1)
        # «наезд» — это не только близость, но и скорость сближения (как у LC4/LPLC2)
        closing = np.clip((prox - self.prev_prox) / DT, 0.0, None)
        self.prev_prox = prox
        loom = np.clip(prox * (0.6 + 1.2 * closing), 0.0, 2.0)
        if self.done.any():                     # доехавшие не чувствуют мир
            odor_l = np.where(self.done, 0.0, odor_l)
            odor_r = np.where(self.done, 0.0, odor_r)
            loom = np.where(self.done[:, None], 0.0, loom)
        return odor_l, odor_r, loom[:, 0], loom[:, 1]

    # ── динамика ──

    def step(self, left, right):
        active = ~self.done
        left, right = np.where(active, left, 0.0), np.where(active, right, 0.0)
        v = (left + right) * 0.5 * V_MAX
        omega = (right - left) * OMEGA_MAX
        self.th += omega * DT
        self.p = self.p + np.stack([np.cos(self.th), np.sin(self.th)], 1) * (v * DT)[:, None]
        self.p[:, 0] = np.clip(self.p[:, 0], 0.3, ARENA - 0.3)
        self.p[:, 1] = np.clip(self.p[:, 1], 0.3, ARENA - 0.3)

        d = np.linalg.norm(self.p - self.goal, axis=1)
        self.dmin = np.where(active, np.minimum(self.dmin, d), self.dmin)
        self.steps += active
        # эпизод кончается сразу на финише или на столкновении
        self.success |= active & (d < GOAL_R)
        self.crash |= active & (self._obst_dist(self.p).min(axis=1) < 0.25)
        self.done |= self.success | self.crash | (self.steps >= self.max_steps)
        return self.done

    def result(self):
        progress = np.clip((self.d0 - self.dmin) / np.maximum(self.d0, 1e-6), 0, 1)
        return {
            "success": float(self.success.mean()),
            "crash": float(self.crash.mean()),
            "progress": float(progress.mean()),
            "steps": float(self.steps[self.success].mean()) if self.success.any() else float("nan"),
            "fitness": float((self.success * 1.0 - 0.5 * self.crash + 0.3 * progress).mean()),
        }


# ─────────────────────────── эксперт и его демонстрации ───────────────────────────

def expert_wheels(ol, orr, ll, lr, k_odor, k_loom, v):
    turn = k_odor * (orr - ol) - k_loom * (lr - ll)
    return np.clip(v + turn, -1, 1), np.clip(v - turn, -1, 1)


def run_expert(world: World, k_odor, k_loom, v):
    world.reset()
    for _ in range(world.max_steps):
        ol, orr, ll, lr = world.sensors()
        if world.step(*expert_wheels(ol, orr, ll, lr, k_odor, k_loom, v)).all():
            break
    return world.result()


def tune_expert(world: World):
    """Перебор 3 констант эксперта на обучающих эпизодах — честная планка."""
    best, bp = -1.0, (3.5, 1.0, 0.9)
    for k_odor in (0.5, 1.0, 2.0, 3.5, 5.0):
        for k_loom in (0.0, 0.5, 1.0, 2.0, 3.0):
            for v in (0.4, 0.6, 0.9):
                f = run_expert(world, k_odor, k_loom, v)["fitness"]
                if f > best:
                    best, bp = f, (k_odor, k_loom, v)
    return bp, best


def demo_stream(world: World, params, rollouts: int = 3):
    """Записываем сенсорный поток и команды эксперта — одно и то же для всех условий."""
    stream = []
    for _ in range(rollouts):
        world.reset()
        for _ in range(world.max_steps):
            ol, orr, ll, lr = world.sensors()
            left, right = expert_wheels(ol, orr, ll, lr, *params)
            stream.append((ol, orr, ll, lr, left, right))
            if world.step(left, right).all():
                break
    return stream


def replay(circuit: FlyCircuit, stream, gain: float, stride: int = 1):
    """Прогон сенсорного потока через контур. gain масштабирует обе модальности."""
    circuit.reset(batch=len(stream[0][0]))
    feats, targets = [], []
    for i, (ol, orr, ll, lr, left, right) in enumerate(stream):
        ext = circuit.sensor_current(gain, gain, ol, orr, ll, lr)
        for _ in range(BRAIN_STEPS):
            circuit.step(ext)
        if i % stride:
            continue
        feats.append(circuit.features().copy())
        targets.append(np.stack([left, right], 1))
    return np.concatenate(feats), np.concatenate(targets)


def fit_readout(feats: np.ndarray, targets: np.ndarray, ridge: float = 1e-3):
    """Ридж-регрессия: частоты нейронов (+ сдвиг) → 2 колеса. Она и есть «обучение»."""
    mu = feats.mean(axis=0)
    # Пол для sd: у молчащих нейронов разброс ~1e-8, и без пола любой численный шум
    # в их частоте после нормировки превращался в десятки и забивал весь читатель.
    sd = feats.std(axis=0)
    floor = max(1e-6, 1e-3 * float(sd.max()))
    sd = np.maximum(sd, floor)
    X = np.concatenate([(feats - mu) / sd, np.ones((len(feats), 1), np.float32)], axis=1)
    A = X.T @ X + ridge * len(X) * np.eye(X.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, X.T @ targets)
    return mu.astype(np.float32), sd.astype(np.float32), W.T.astype(np.float32)


def fit_with_gain(circuit: FlyCircuit, stream, grid=GAIN_GRID):
    """Единственная настройка у всех условий: усиление входа выбирается так,
    чтобы с нисходящих нейронов ЛУЧШЕ ВСЕГО считывалась команда эксперта.

    Это не обучение поведения — это выбор одного скаляра по ошибке клонирования
    на тех же самых демонстрациях. Процедура, сетка и данные идентичны для
    настоящей проводки, перемешанной и случайной, поэтому сравнение честное.
    """
    stride = 1 if len(circuit.feat_idx) <= 64 else 2      # не раздуваем датасет
    best = None
    for g in grid:
        feats, targets = replay(circuit, stream, g, stride)
        readout = fit_readout(feats, targets)
        mu, sd, W = readout
        X = np.concatenate([(feats - mu) / sd, np.ones((len(feats), 1), np.float32)], axis=1)
        mse = float(((X @ W.T - targets) ** 2).mean())
        if best is None or mse < best[0]:
            best = (mse, g, readout, float(feats.var(axis=0).mean()))
    return best[1], best[2], best[0], best[3]


def rollout(circuit: FlyCircuit, world: World, gain: float, readout, reps: int = 1,
            collect: bool = False):
    """Замкнутый контур. Если collect — заодно записываем сенсорный поток робота.

    Записанный поток затем размечается командами эксперта — это и есть DAgger:
    политика сама заезжает туда, где ошибается, а мы добавляем эти состояния в датасет.
    """
    readout = readout[1] if isinstance(readout, tuple) and len(readout) == 4 else readout
    mu, sd, W = readout
    outs, stream = [], []
    for _ in range(reps):
        circuit.reset(batch=world.m)
        world.reset()
        for _ in range(world.max_steps):
            ol, orr, ll, lr = world.sensors()
            if collect:
                stream.append((ol, orr, ll, lr))
            ext = circuit.sensor_current(gain, gain, ol, orr, ll, lr)
            for _ in range(BRAIN_STEPS):
                circuit.step(ext)
            X = np.concatenate([(circuit.features() - mu) / sd,
                                np.ones((world.m, 1), np.float32)], axis=1)
            cmd = X @ W.T
            if world.step(np.clip(cmd[:, 0], -1, 1), np.clip(cmd[:, 1], -1, 1)).all():
                break
        outs.append(world.result())
    res = {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}
    return (res, stream) if collect else res


def label_with_expert(stream, params):
    """Разметка потока робота командами эксперта (od и loom → два колеса)."""
    k_odor, k_loom, v = params
    out = []
    for ol, orr, ll, lr in stream:
        left, right = expert_wheels(ol, orr, ll, lr, k_odor, k_loom, v)
        out.append((ol, orr, ll, lr, left, right))
    return out


def train_policy(circuit: FlyCircuit, train: World, base_stream, eparams,
                 dagger_iters: int = 2):
    """Обучение читателя: демонстрации эксперта + итерации DAgger.

    Один и тот же бюджет итераций и одинаковый объём данных у всех условий.
    """
    stream = list(base_stream)
    gain, readout, mse, resp = fit_with_gain(circuit, stream)
    for _ in range(dagger_iters):
        _, extra = rollout(circuit, train, gain, readout, reps=1, collect=True)
        stream = stream + label_with_expert(extra, eparams)
        gain, readout, mse, resp = fit_with_gain(circuit, stream)
    return gain, readout, mse, resp, len(stream)


def run_nobrain(world: World, seed: int = 0, reps: int = 3):
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(reps):
        world.reset()
        th = rng.normal(0, 0.3, world.m)
        for _ in range(world.max_steps):
            th += rng.normal(0, 0.6, world.m)
            if world.step(np.clip(0.5 + 0.4 * np.tanh(th), -1, 1),
                          np.clip(0.5 - 0.4 * np.tanh(th), -1, 1)).all():
                break
        outs.append(world.result())
    return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}


# ─────────────────────────── основной прогон ───────────────────────────

BUILDERS = {
    "real": lambda p, s: FlyCircuit(p),
    "shuffled": lambda p, s: ShuffledCircuit(p, seed=s),
    "random": lambda p, s: RandomCircuit(p, seed=s),
}


def evaluate_circuit(path: str, train: World, test: World, stream, eparams, seed: int,
                     reps: int, dagger: int = 2, readout: str = "all",
                     conditions=("real", "shuffled", "random")):
    rows = {}
    for name in conditions:
        circ = BUILDERS[name](path, seed)
        circ.set_readout(readout)
        gain, read_two, mse, resp, train_ticks = train_policy(circ, train, stream, eparams)
        res = rollout(circ, test, gain, read_two, reps)
        res["gain"] = gain
        res["clone_mse"] = mse
        res["train_ticks"] = train_ticks
        res["response"] = resp
        res["params"] = int(read_two[2].size)         # только числа читателя
        st = circ.stats()
        res["neurons"] = st["neurons"]
        res["synapses"] = st["synapses"]
        res["readout"] = {"mu": read_two[0].tolist(), "sd": read_two[1].tolist(),
                          "W": read_two[2].tolist(), "feat": int(len(circ.feat_idx))}
        rows[name] = res
        print(f"  [{name:<9}] вход ×{gain:>4.0f} · клон MSE {mse:.4f} · "
              f"данных {train_ticks} тактов · дошёл {res['success'] * 100:>3.0f}% · "
              f"врезался {res['crash'] * 100:>3.0f}% · fitness {res['fitness']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", default=DEFAULT_CIRCUIT,
                    help="один или несколько контуров через запятую (real оценивается для каждого)")
    ap.add_argument("--episodes", type=int, default=8, help="эпизодов на обучении")
    ap.add_argument("--test-episodes", type=int, default=24, help="эпизодов на тесте")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--demo-rollouts", type=int, default=3)
    ap.add_argument("--dagger", type=int, default=2, help="итераций DAgger")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--readout", choices=("dn", "all"), default="all",
                    help="dn — только 12 нисходящих нейронов, all — всё состояние мозга")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = [p.strip() for p in args.circuits.split(",") if p.strip()]
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"нет {p} — сначала: python3 scripts/extract_circuit.py")

    t0 = time.time()
    train = World(args.episodes, args.max_steps, seed=100 + args.seed)
    test = World(args.test_episodes, args.max_steps, seed=200 + args.seed)

    print("── эксперт (планка для сравнения) ──")
    eparams, efitness = tune_expert(train)
    print(f"  лучший Braitenberg: k_odor={eparams[0]}, k_loom={eparams[1]}, v={eparams[2]}, "
          f"fitness на обучении {efitness:.3f}")
    print("  на тесте: " + json.dumps({k: round(v, 3) if v == v else None
                                      for k, v in run_expert(test, *eparams).items()}))
    stream = demo_stream(train, eparams, args.demo_rollouts)
    print(f"  демонстраций эксперта: {len(stream)} тактов × {train.m} роботов")
    print("── без мозга ──")
    nobrain = run_nobrain(test, args.seed, args.reps)
    print(f"  случайное блуждание: дошёл {nobrain['success'] * 100:.0f}%, "
          f"fitness {nobrain['fitness']:.3f}")

    all_rows = {}
    for path in paths:
        print(f"\n── условие: {os.path.basename(path)} ──")
        rows = evaluate_circuit(path, train, test, stream, eparams, args.seed, args.reps,
                                args.dagger, args.readout)
        all_rows[path] = rows

    print("\n" + "=" * 86)
    print(f"{'контур':<22}{'проводка':<11}{'нейронов':>9}{'синапсов':>10}"
          f"{'обучено':>9}{'дошёл':>8}{'врезался':>10}{'fitness':>9}")
    print("-" * 86)
    for path, rows in all_rows.items():
        first = True
        for name, r in rows.items():
            label = os.path.basename(path).replace("fly_circuit", "").replace(".json", "") or "1024"
            print(f"{(label if first else ''):<22}{name:<11}{r['neurons']:>9}{r['synapses']:>10,}"
                  f"{r['params']:>9}{r['success'] * 100:>7.0f}%{r['crash'] * 100:>9.0f}%"
                  f"{r['fitness']:>9.3f}")
            first = False
    print("-" * 86)
    e = run_expert(test, *eparams)
    print(f"{'braitenberg (без сети)':<33}{'-':>9}{'-':>10}{3:>9}"
          f"{e['success'] * 100:>7.0f}%{e['crash'] * 100:>9.0f}%{e['fitness']:>9.3f}")
    print(f"{'nobrain (случайно)':<33}{'-':>9}{'-':>10}{0:>9}"
          f"{nobrain['success'] * 100:>7.0f}%{nobrain['crash'] * 100:>9.0f}%"
          f"{nobrain['fitness']:>9.3f}")
    print("=" * 86)
    print(f"\nвремя: {time.time() - t0:.0f} с")

    if args.out:
        payload = {
            "expert": {"params": list(eparams), "fitness": efitness},
            "nobrain": nobrain,
            "braitenberg_test": e,
            "circuits": {p: {k: {kk: (vv if vv == vv else None) for kk, vv in v.items()}
                             for k, v in rows.items()} for p, rows in all_rows.items()},
            "settings": {"readout": args.readout, "episodes": args.episodes,
                         "test_episodes": args.test_episodes,
                         "max_steps": args.max_steps, "demo_rollouts": args.demo_rollouts,
                         "dagger": args.dagger, "reps": args.reps, "seed": args.seed},
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"записано в {args.out}")


if __name__ == "__main__":
    main()
