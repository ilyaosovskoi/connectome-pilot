"""flybrain.py — fruit-fly brain runtime for robots: real neurons, real synapses.

The circuit comes from data/fly_circuit.json (built by scripts/extract_circuit.py).
Unlike the browser model (app.js), connections here are NOT random: this is a
FlyWire v783 subgraph with real synapse counts and real signs
(acetylcholine +, GABA/glutamate -), and the neurons are real brain cells:
ORN_* (smell), LPLC2/LC4 (looming/shadow), T4/T5 (motion), KC (mushroom body),
DNa01/DNa02 (turning), DNp09 (forward), MDN/DNp01 (backward).

Robot sensors drive current into side-split sensory neurons (left/right).
Descending neurons drive the differential-drive wheels.

The engine is batch-vectorized: one step() call advances M episodes at once
(needed for output-gain training — hundreds of rollouts in one go).

What is trained: NOTHING inside the network. All weights, signs and fan-out
come from the connectome, frozen. Only a tiny readout from the 12 descending
neurons to two wheels is learned — 26 numbers (2 x (12 + 1)). That matches the
biology: the brain talks to the body only through descending neurons
(the fly has ~1300 of them).
"""

from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CIRCUIT = os.path.join(ROOT, "data", "fly_circuit.json")

# LIF из Shiu et al. 2024 (та же физиология, что в app.js)
V_REST = -52.0
V_TH = -45.0
V_RESET = -54.0
TAU_M = 20.0        # мс
REFRACT = 2.2       # мс
TAU_SYN = 5.0       # мс
TAU_RATE = 120.0    # мс — окно чтения выхода
I_SCALE = 25.0      # мВ на единицу нормированного синаптического драйва
# Нормировка: драйв = доля полного входа клетки, пришедшая за такт. Один спайк
# слабого партнёра не должен стрелять сам по себе — иначе сеть просто орет.

OUT_GROUPS = ("turn", "fwd", "bwd")   # единственный выход мозга к телу


class FlyCircuit:
    """Контур + LIF. Умеет шагать батч роботов одновременно."""

    def __init__(self, path: str = DEFAULT_CIRCUIT, dt: float = 5.0, edges=None):
        if path is not None and str(path).endswith(".npz"):
            self._init_from_npz(str(path), dt)
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.meta = data.get("meta", {})
        neurons = data["neurons"]
        edges = np.asarray(data["edges"] if edges is None else edges,
                           dtype=np.float64).reshape(-1, 4)

        self.n = len(neurons)
        self.dt = float(dt)
        self.ids = np.array([x["id"] for x in neurons], dtype=np.int64)
        self.types = [x["type"] for x in neurons]
        self.sign = np.array([x["sign"] for x in neurons], dtype=np.float32)
        self.side = np.array([1 if x["side"] == "right" else (0 if x["side"] == "left" else -1)
                              for x in neurons], dtype=np.int8)

        pre = edges[:, 0].astype(np.int32)
        post = edges[:, 1].astype(np.int32)
        w = edges[:, 2].astype(np.float32)
        # знак связи — знак ПРЕсинаптической клетки (её медиатор)
        signed = w * self.sign[pre]

        self.groups = {k: np.asarray(v, dtype=np.int32) for k, v in data["groups"].items()}
        # сторона для групп, которые латерализованы (запах и наезд)
        self.group_left = {k: v[self.side[v] == 0] for k, v in self.groups.items()}
        self.group_right = {k: v[self.side[v] == 1] for k, v in self.groups.items()}
        # моторный выход мозга — 12 нисходящих нейронов; их и читает биологичный вариант
        self.out_idx = np.concatenate([self.groups[g] for g in OUT_GROUPS if g in self.groups]) \
            if any(g in self.groups for g in OUT_GROUPS) else np.array([], dtype=np.int32)
        self.set_readout("dn")

        # Нормировка на полный синаптический вход постсинаптической клетки:
        # тогда "драйв" — доля её входа, а не абсолютное число синапсов.
        in_sum = np.bincount(post, weights=np.abs(w), minlength=self.n).astype(np.float32)
        in_sum[in_sum == 0] = 1.0
        B = sp.coo_matrix((signed / in_sum[post], (post, pre)), shape=(self.n, self.n)).tocsr()
        B.sum_duplicates()
        self.B = B

        self.reset(batch=1)

    def _init_from_npz(self, path: str, dt: float) -> None:
        """Load a pre-converted NPZ bundle (see robot/formats.py --emit).

        Same synapses, signs and normalization as the JSON edge-list,
        but faster to load and smaller on disk.
        """
        z = np.load(path, allow_pickle=True)
        csr = sp.csr_matrix(
            (np.asarray(z["data"], dtype=np.float32),
             np.asarray(z["indices"], dtype=np.int32),
             np.asarray(z["indptr"], dtype=np.int32)))
        self.n = int(csr.shape[0])
        self.dt = float(dt)
        self.B = csr
        self.meta = json.loads(str(z["meta_json"][0])) if "meta_json" in z else {}
        self.sign = np.asarray(z["sign"], dtype=np.float32)
        self.side = np.asarray(z["side"], dtype=np.int8)
        self.types = [str(t) for t in np.asarray(z["ntype"]).tolist()]
        self.ids = np.arange(self.n, dtype=np.int64)  # NPZ stores topology, not FlyWire IDs
        self.groups = {k[6:]: np.asarray(z[k], dtype=np.int32)
                       for k in z.files if k.startswith("group_")}
        self.group_left = {k: v[self.side[v] == 0] for k, v in self.groups.items()}
        self.group_right = {k: v[self.side[v] == 1] for k, v in self.groups.items()}
        self.out_idx = np.concatenate([self.groups[g] for g in OUT_GROUPS if g in self.groups]) \
            if any(g in self.groups for g in OUT_GROUPS) else np.array([], dtype=np.int32)
        self.set_readout("dn")
        self.reset(batch=1)

    # ── состояние ──

    def reset(self, batch: int = 1) -> None:
        m = batch
        self.m = m
        self.V = np.full((m, self.n), V_REST, dtype=np.float32)
        self.ref = np.zeros((m, self.n), dtype=np.float32)
        self.Isyn = np.zeros((m, self.n), dtype=np.float32)
        self.spikes = np.zeros((m, self.n), dtype=np.float32)
        self.rate = np.zeros((m, self.n), dtype=np.float32)
        self.bias = np.zeros(self.n, dtype=np.float32)
        self.t_ms = 0.0
        # NOTE: homeostasis config survives reset (bias itself restarts at 0).
        self.homeo = getattr(self, "homeo", None)

    def enable_homeostasis(self, target: float = 5.0, eta: float = 0.05,
                           max_bias: float = 10.0) -> None:
        """Homeostatic excitability: neurons drift toward a target firing rate.

        Silent cells (e.g. turn DNs that never reach threshold) slowly gain
        bias current; overactive cells lose it. Real neurons do this; here it
        keeps every pathway recruitable for plasticity instead of frozen-silent.
        """
        self.homeo = {"target": float(target), "eta": float(eta),
                      "max_bias": float(max_bias)}

    def sensor_current(self, g_odor: float, g_loom: float,
                       odor_l, odor_r, loom_l, loom_r) -> np.ndarray:
        """Мир снаружи: запах и близость препятствия слева/справа → ток в сенсорику."""
        m = self.m
        ext = np.zeros((m, self.n), dtype=np.float32)
        odor = np.stack([np.asarray(odor_l, np.float32), np.asarray(odor_r, np.float32)], 1)
        loom = np.stack([np.asarray(loom_l, np.float32), np.asarray(loom_r, np.float32)], 1)

        def inject(idx_left, idx_right, sig, gain):
            if len(idx_left):
                ext[:, idx_left] += sig[:, :1] * gain
            if len(idx_right):
                ext[:, idx_right] += sig[:, 1:] * gain

        inject(self.group_left.get("odor", []), self.group_right.get("odor", []), odor, g_odor)
        inject(self.group_left.get("loom", []), self.group_right.get("loom", []), loom, g_loom)
        return ext

    def step(self, ext: np.ndarray) -> None:
        """Один LIF-шаг (dt мс) для всего батча. ext — внешний ток (m × n)."""
        dt = self.dt
        if self.homeo is not None:
            ext = ext + self.bias[None, :].astype(np.float32)
        # синаптический драйв = доля входа клетки, пришедшая за такт (n × m → m × n)
        drive = (self.B @ self.spikes.T).T
        self.Isyn = self.Isyn * np.exp(-dt / TAU_SYN) + drive

        active = self.ref <= 0.0
        dV = (-(self.V - V_REST) + self.Isyn * I_SCALE + ext) * (dt / TAU_M)
        self.V = np.where(active, self.V + dV, V_RESET)

        fired = active & (self.V >= V_TH)
        self.V = np.where(fired, V_RESET, self.V)
        self.ref = np.where(fired, REFRACT, np.maximum(self.ref - dt, 0.0))
        self.spikes = fired.astype(np.float32)
        self.rate *= np.exp(-dt / TAU_RATE)
        self.rate += self.spikes * (dt / TAU_RATE) * (1000.0 / dt)   # масштаб ~Гц
        if self.homeo is not None:
            err = self.homeo["target"] - self.rate.mean(axis=0)
            self.bias = np.clip(self.bias + self.homeo["eta"] * err,
                                -self.homeo["max_bias"],
                                self.homeo["max_bias"]).astype(np.float32)
        self.t_ms += dt

    # ── выход на колёса ──

    def group_rate(self, name: str) -> np.ndarray:
        idx = self.groups.get(name, np.array([], dtype=np.int32))
        if len(idx) == 0:
            return np.zeros(self.m, dtype=np.float32)
        return self.rate[:, idx].mean(axis=1)

    def side_rate(self, name: str, right: bool) -> np.ndarray:
        idx = self.group_right.get(name, []) if right else self.group_left.get(name, [])
        if len(idx) == 0:
            return np.zeros(self.m, dtype=np.float32)
        return self.rate[:, idx].mean(axis=1)

    def set_readout(self, kind: str = "dn") -> None:
        """Кто доступен читателю: только нисходящие нейроны (биология) или весь мозг."""
        if kind == "dn":
            self.feat_idx = self.out_idx
        elif kind == "all":
            self.feat_idx = np.arange(self.n, dtype=np.int32)
        else:
            raise ValueError(f"неизвестный читатель: {kind}")
        self.readout_kind = kind

    def features(self) -> np.ndarray:
        """Состояние мозга, доступное читателю: частоты выбранных нейронов."""
        return self.rate[:, self.feat_idx]

    def wheels(self, Wout: np.ndarray):
        """Единственное обученное место: 2×(n_out+1) чисел (читатель DN → колёса)."""
        X = np.concatenate([self.features(), np.ones((self.m, 1), np.float32)], axis=1)
        cmd = X @ Wout.T
        return np.clip(cmd[:, 0], -1, 1).astype(np.float32), \
            np.clip(cmd[:, 1], -1, 1).astype(np.float32)

    def stats(self) -> dict:
        return {
            "neurons": self.n,
            "synapses": int(self.B.nnz),
            "groups": {k: int(len(v)) for k, v in self.groups.items()},
            "inhibitory_neurons": int((self.sign < 0).sum()),
            "output_neurons": int(len(self.out_idx)),
            "readout_kind": self.readout_kind,
            "readout_inputs": int(len(self.feat_idx)),
            "fidelity": self.meta.get("fidelity"),
        }


class ShuffledCircuit(FlyCircuit):
    """Абляция структуры: те же нейроны, те же веса, но цели синапсов перепутаны.

    Проверяет главное: даёт ли точность именно НАСТОЯЩАЯ проводка, или хватило бы
    любого графа с тем же числом связей и тем же распределением весов. При таком
    перемешивании сохраняются и веер каждого нейрона, и набор весов, и знаки —
    ломается только то, КУДА связи ведут.
    """

    def __init__(self, path: str = DEFAULT_CIRCUIT, dt: float = 5.0, seed: int = 0):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        rng = np.random.default_rng(seed)
        post = np.array([e[1] for e in data["edges"]], dtype=np.int64)
        shuffled = rng.permutation(post)
        new_edges = [[e[0], int(shuffled[i]), e[2], e[3]] for i, e in enumerate(data["edges"])]
        super().__init__(path, dt, edges=new_edges)


class RandomCircuit(FlyCircuit):
    """Нижняя ступень абляции: те же веса, но случайные и источники, и цели.

    Ломается не только структура, но и распределение степеней — остаётся лишь
    размер и набор весов. Если и это работает не хуже настоящей проводки, значит
    коннектом здесь ни при чём.
    """

    def __init__(self, path: str = DEFAULT_CIRCUIT, dt: float = 5.0, seed: int = 0):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        rng = np.random.default_rng(seed)
        n = len(data["neurons"])
        pre = np.array([e[0] for e in data["edges"]], dtype=np.int64)
        new_edges = [[int(pre[i]), int(rng.integers(0, n)), e[2], e[3]]
                     for i, e in enumerate(data["edges"])]
        super().__init__(path, dt, edges=new_edges)
