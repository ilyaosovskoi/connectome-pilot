"""plastic.py — trainable synapses on top of the frozen FlyWire connectome.

The base FlyCircuit freezes all weights and only trains a linear readout.
This module adds a small, stable plastic subset with dopamine-style
reward-modulated updates:

    e <- decay * e + outer(post_rate, pre_spikes)   (eligibility trace)
    W <- clip(W + lr * RPE * e)                     (RPE = r - V_baseline)

Default plastic set: excitatory synapses onto descending (DN) output neurons.
For the 256-neuron circuit that is ~60 weights — small enough to stay stable,
large enough to steer behavior. Use scope="all-exc" to open all excitatory
synapses (~1000 on circuit-256).

The connectome is used as initialization, not as a frozen reservoir:
real wiring + real weights as the starting point, then task-driven plasticity.
"""

from __future__ import annotations

import numpy as np

from flybrain import FlyCircuit


class PlasticFlyCircuit(FlyCircuit):
    """FlyCircuit with a trainable synaptic subset and eligibility traces."""

    def __init__(
        self,
        path=None,
        dt: float = 5.0,
        edges=None,
        scope: str = "dn-exc",
        lr: float = 0.01,
        decay: float = 0.9,
        w_min: float = 0.5,
        w_max: float = 30.0,
    ):
        super().__init__(path=path, dt=dt, edges=edges)
        assert scope in ("dn-exc", "all-exc", "all")
        self.scope = scope
        self.lr = float(lr)
        self.decay = float(decay)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

        # Raw (unsigned) weights per stored edge, in CSR order.
        coo = self.B.tocoo()
        self.edge_pre = coo.col.astype(np.int32)
        self.edge_post = coo.row.astype(np.int32)
        # self.B stores signed/in_sum-normalized values; recover raw magnitude:
        # raw = |B| * in_sum[post]
        in_sum = np.zeros(self.n, dtype=np.float32)
        np.add.at(in_sum, self.edge_post, np.abs(coo.data))
        # in_sum here is in normalized units; instead keep explicit raw weights
        # from the source file for clean plasticity. Rebuild from edges arg or file.
        self._rebuild_raw_weights(path, edges)

        out_set = set(self.out_idx.tolist())
        mask = np.ones(len(self.edge_pre), dtype=bool)
        if scope == "dn-exc":
            mask = np.array(
                [(q in out_set) and self.sign[p] > 0
                 for p, q in zip(self.edge_pre, self.edge_post)]
            )
        elif scope == "all-exc":
            mask = np.array([self.sign[p] > 0 for p in self.edge_pre])
        self.plastic_mask = mask
        self.plastic_idx = np.nonzero(mask)[0].astype(np.int64)
        # Scale-aware plasticity: raw FlyWire counts span ~3..1500 synapses,
        # so absolute steps/clips would erase the strongest pathways.
        # Each weight moves proportionally to its own initial value and is
        # clipped to [0.2x, 5x] of it.
        self.w_init = np.array(
            [self._raw_lookup.get((int(self.edge_pre[i]), int(self.edge_post[i])), 1.0)
             for i in self.plastic_idx], dtype=np.float64)
        self.w_init[self.w_init <= 0] = 1.0
        # Eligibility trace per plastic edge.
        self.elig = np.zeros(len(self.plastic_idx), dtype=np.float32)
        # Map plastic position -> position in CSR data array.
        # CSR data order == order of coo after tocsr? Rebuild mapping safely:
        csr = self.B.tocsr()
        csr_coo = csr.tocoo()
        # Build lookup (post, pre) -> data position
        self._data_pos = np.zeros(len(self.edge_pre), dtype=np.int64)
        lookup = {}
        for pos, (r, c) in enumerate(zip(csr_coo.row, csr_coo.col)):
            lookup[(int(r), int(c))] = pos
        # edge_pre/post are in coo order of original B; find each in csr order
        for i, (r, c) in enumerate(zip(self.edge_post, self.edge_pre)):
            self._data_pos[i] = lookup[(int(r), int(c))]
        self._csr = csr
        self.B = csr
        self.V_baseline = 0.0

    def _rebuild_raw_weights(self, path, edges):
        import json as _json
        if edges is not None:
            arr = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
            self._raw_lookup = {
                (int(a), int(b)): float(w)
                for a, b, w, _ in arr.tolist()
            }
        elif path is not None and str(path).endswith(".npz"):
            z = np.load(path, allow_pickle=True)
            pre = np.asarray(z["pre"]).tolist()
            post = np.asarray(z["post"]).tolist()
            raw = np.asarray(z["raw"], dtype=np.float32).tolist()
            self._raw_lookup = {
                (int(a), int(b)): float(w)
                for a, b, w in zip(pre, post, raw)
            }
        else:
            with open(path or self._default_path(), encoding="utf-8") as fh:
                data = _json.load(fh)
            self._raw_lookup = {
                (int(a), int(b)): float(w) for a, b, w, _ in data["edges"]
            }

    def _default_path(self):
        import os as _os
        return _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "fly_circuit.json",
        )

    @property
    def n_plastic(self) -> int:
        return int(len(self.plastic_idx))

    def reset_trace(self):
        self.elig.fill(0.0)

    def observe(self, pre_spikes: np.ndarray, post_rate: np.ndarray,
                post_spikes: np.ndarray | None = None):
        """Accumulate eligibility from one control step (batch dim collapsed by mean).

        Classic R-STDP mix: a spike-spike term (recruits quiet neurons) plus a
        small rate term (refines active ones). Pure rate gating freezes silent
        postsynaptic cells forever — verified failure mode on turn DNs.
        """
        pre = np.asarray(pre_spikes, dtype=np.float32)
        if pre.ndim == 2:
            pre = pre.mean(axis=0)
        post = np.asarray(post_rate, dtype=np.float32)
        if post.ndim == 2:
            post = post.mean(axis=0)
        if post_spikes is None:
            post_spikes = (post > 0).astype(np.float32)
        else:
            post_spikes = np.asarray(post_spikes, dtype=np.float32)
            if post_spikes.ndim == 2:
                post_spikes = post_spikes.mean(axis=0)
        pe = pre[self.edge_pre[self.plastic_idx]]
        po_rate = post[self.edge_post[self.plastic_idx]]
        po_spk = post_spikes[self.edge_post[self.plastic_idx]]
        self.elig = self.decay * self.elig + (pe * (po_spk + 0.05 * po_rate)).astype(np.float32)

    def _refresh_matrix(self):
        """Recompute normalized signed CSR values from raw weights."""
        data = self._csr.data
        pre_all = self.edge_pre
        post_all = self.edge_post
        raw_all = np.array(
            [self._raw_lookup.get((int(p), int(q)), 1.0)
             for p, q in zip(pre_all, post_all)],
            dtype=np.float64,
        )
        in_sum = np.zeros(self.n, dtype=np.float64)
        np.add.at(in_sum, post_all, np.abs(raw_all))
        in_sum[in_sum == 0] = 1.0
        signed = raw_all * np.array([self.sign[p] for p in pre_all], dtype=np.float64)
        normed = (signed / in_sum[post_all]).astype(np.float32)
        new_data = np.empty_like(data)
        for i in range(len(pre_all)):
            new_data[self._data_pos[i]] = normed[i]
        self._csr.data = new_data
        self.B = self._csr

    def apply_episodic(self, ret: float) -> float:
        """One REINFORCE-style update at episode end.

        Eligibility accumulated over the whole episode (see observe()) is
        normalized so the most-coactive synapse takes the full step; the step
        is a bounded fraction of each weight's own initial value. This fits
        slow robots: 700 sparse control steps build one dense co-activity
        picture, then a single return judges it.
        """
        rpe = float(ret) - self.V_baseline
        self.V_baseline += 0.2 * rpe
        if self.n_plastic == 0:
            return rpe
        norm = float(np.abs(self.elig).max()) or 1.0
        step = (self.lr * np.tanh(rpe / 2.0)
                * (self.elig / norm) * self.w_init * 0.2)
        for k, ei in enumerate(self.plastic_idx):
            key = (int(self.edge_pre[ei]), int(self.edge_post[ei]))
            lo, hi = 0.2 * self.w_init[k], 5.0 * self.w_init[k]
            cur = self._raw_lookup.get(key, 1.0) + float(step[k])
            self._raw_lookup[key] = min(max(cur, lo), hi)
        self._refresh_matrix()
        self.elig.fill(0.0)
        return rpe

    def apply_reward(self, reward: float) -> float:
        """Per-step dopamine update (same math, smaller step). Noisy on slow
        robots — prefer apply_episodic() there. Kept for fine-timescale tasks."""
        rpe = float(reward) - self.V_baseline
        self.V_baseline += 0.05 * rpe  # slow critic
        if abs(rpe) < 1e-6 or self.n_plastic == 0:
            return rpe
        step = (self.lr * np.tanh(rpe / 2.0)
                * np.clip(self.elig, -3.0, 3.0) * self.w_init * 0.02)
        for k, ei in enumerate(self.plastic_idx):
            key = (int(self.edge_pre[ei]), int(self.edge_post[ei]))
            lo, hi = 0.2 * self.w_init[k], 5.0 * self.w_init[k]
            cur = self._raw_lookup.get(key, 1.0) + float(step[k])
            self._raw_lookup[key] = min(max(cur, lo), hi)
        self._refresh_matrix()
        # Decay trace after consuming (standard R-STDP).
        self.elig *= 0.5
        return rpe

    def get_plastic_weights(self) -> np.ndarray:
        return np.array(
            [self._raw_lookup.get((int(self.edge_pre[i]), int(self.edge_post[i])), 1.0)
             for i in self.plastic_idx],
            dtype=np.float32,
        )

    def set_plastic_weights(self, w: np.ndarray):
        w = np.asarray(w, dtype=np.float64)
        assert len(w) == self.n_plastic
        for k, ei in enumerate(self.plastic_idx):
            key = (int(self.edge_pre[ei]), int(self.edge_post[ei]))
            lo, hi = 0.2 * self.w_init[k], 5.0 * self.w_init[k]
            self._raw_lookup[key] = float(min(max(w[k], lo), hi))
        self._refresh_matrix()
        self.elig.fill(0.0)
