"""novelty.py — count-based intrinsic bonus (curiosity).

The brain keeps a visit-count over a discretized sensor state. First visits
pay 1.0, familiar states pay ~0:

    bonus(s) = 1 / sqrt(count(s))   (computed BEFORE incrementing)

Why counts and not a learned predictor: 4 sensors, tiny bins, zero training,
fully reproducible. It answers "have I been here before", which is all a
rover needs to stop pacing the same corner.

Combine with the task reward, never alone (novelty-only agents learn to
spin in interesting corners):
    reward = task_reward + novelty_coef * novelty_bonus
"""

from __future__ import annotations

import numpy as np


class NoveltyBonus:
    """Visit counts over binned (odor_l, odor_r, loom_l, loom_r)."""

    def __init__(self, n_bins: int = 8, lo: float = 0.0, hi: float = 2.0):
        self.n_bins = int(n_bins)
        self.lo = float(lo)
        self.hi = float(hi)
        self.counts: dict[tuple, int] = {}

    def _key(self, sensors) -> tuple:
        v = np.clip(np.asarray(sensors, dtype=float).ravel()[:4],
                    self.lo, self.hi)
        idx = ((v - self.lo) / (self.hi - self.lo) * self.n_bins).astype(int)
        idx = np.clip(idx, 0, self.n_bins - 1)
        return tuple(idx.tolist())

    def bonus(self, ol, orr, ll, lr) -> float:
        """Bonus for the current state (scalar float sensors)."""
        key = self._key((float(ol), float(orr), float(ll), float(lr)))
        n = self.counts.get(key, 0)
        self.counts[key] = n + 1
        return 1.0 / (n + 1) ** 0.5

    def coverage(self) -> int:
        """How many distinct states visited — exploration in one number."""
        return len(self.counts)
