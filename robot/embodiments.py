"""embodiments.py — robot bodies sharing one sensor/motor interface.

Every body speaks the same 4-channel language the fly connectome understands:
  sensors: odor_l, odor_r (attraction gradient, lateralized)
           loom_l, loom_r (hazard proximity, lateralized)
  motors:  left, right in [-1, 1] (differential-drive style command)

Bodies implemented:
  Rover — differential-drive mobile robot seeking an odor source among
          obstacles (wraps the vectorized World from eval.py, one episode).
  Arm   — 2-link planar arm reaching a target with its end-effector.
          Wheel commands map to joint velocities; odor channels encode the
          bearing/distance to the target, loom channels encode joint-limit
          proximity. Same brain, different body.

This is the point: the connectome is a small adaptive controller, not a
rover-only script. If it only worked on one body, it would be a script.
"""

from __future__ import annotations

import numpy as np


class Rover:
    """Mobile robot: 20x20 m arena, odor source, 4 obstacles."""

    name = "rover"

    def __init__(self, max_steps=400, seed=0, episode=0):
        from eval import World
        self._World = World
        self.max_steps = max_steps
        self.seed = seed
        self.episode = episode
        self.world = None

    def reset(self):
        self.world = self._World(1, self.max_steps, seed=self.seed + self.episode)
        self.world.reset()
        # Isolate our single episode (index 0).
        self._d0 = float(np.linalg.norm(self.world.p[0] - self.world.goal[0]))
        self._prev = self._d0
        return self.obs()

    def obs(self):
        ol, orr, ll, lr = self.world.sensors()
        return float(ol[0]), float(orr[0]), float(ll[0]), float(lr[0])

    def step(self, left, right):
        w = self.world
        l = np.array([np.clip(left, -1, 1)], dtype=np.float64)
        r = np.array([np.clip(right, -1, 1)], dtype=np.float64)
        w.step(l, r)
        d = float(np.linalg.norm(w.p[0] - w.goal[0]))
        reward = (self._prev - d) / max(self._d0, 1e-6)
        self._prev = d
        done = bool(w.done[0])
        info = {"success": bool(w.success[0]), "crash": bool(w.crash[0]),
                "dist": d, "progress": (self._d0 - min(d, self._d0)) / max(self._d0, 1e-6)}
        if info["success"]:
            reward += 1.0
        elif info["crash"]:
            reward -= 1.0
        return self.obs(), reward, done, info


class Arm:
    """2-link planar arm: touch the target with the end-effector.

    Geometry: base at origin, L1 = L2 = 1.0 m. Joint limits +-170 deg.
    Sensors: odor strength falls with end-effector distance; lateralization
    comes from the bearing of the target relative to the arm plane normal
    (+25/-25 deg virtual antennae, same convention as the rover).
    Loom channels report joint-limit proximity (left = joint 1, right = joint 2).
    Motors: left/right wheel command -> joint 1/2 angular velocity.
    """

    name = "arm"
    L1 = 1.0
    L2 = 1.0
    LIM = np.radians(170.0)
    V_MAX = 1.5  # rad/s at full command
    DT = 0.05
    GOAL_R = 0.12

    def __init__(self, max_steps=400, seed=0, episode=0):
        self.max_steps = max_steps
        self.seed = seed
        self.episode = episode

    def reset(self):
        rng = np.random.default_rng(self.seed + self.episode)
        self.th = rng.uniform(-1.2, 1.2, size=2)
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.5, 1.8)
        self.target = np.array([np.cos(ang), np.sin(ang)]) * rad
        # Reachable: clamp into the annulus the arm can touch.
        if np.linalg.norm(self.target) > 1.95:
            self.target *= 1.95 / np.linalg.norm(self.target)
        self.steps = 0
        self._d0 = self._dist()
        self._prev = self._d0
        self._dmin = self._d0
        return self.obs()

    def _fk(self):
        x = self.L1 * np.cos(self.th[0]) + self.L2 * np.cos(self.th[0] + self.th[1])
        y = self.L1 * np.sin(self.th[0]) + self.L2 * np.sin(self.th[0] + self.th[1])
        return np.array([x, y])

    def _dist(self):
        return float(np.linalg.norm(self._fk() - self.target))

    def obs(self):
        ee = self._fk()
        d = self._dist()
        bearing = np.arctan2(self.target[1] - ee[1], self.target[0] - ee[0])
        off = np.radians(25.0)
        base = float(np.exp(-d / 0.6))
        odor_l = base * (0.55 + 0.45 * np.cos(bearing - off))
        odor_r = base * (0.55 + 0.45 * np.cos(bearing + off))
        # Joint-limit proximity per joint.
        loom_l = float(np.clip((abs(self.th[0]) - 2.2) / 0.77, 0.0, 1.0))
        loom_r = float(np.clip((abs(self.th[1]) - 2.2) / 0.77, 0.0, 1.0))
        return odor_l, odor_r, loom_l, loom_r

    def step(self, left, right):
        self.th[0] += np.clip(left, -1, 1) * self.V_MAX * self.DT
        self.th[1] += np.clip(right, -1, 1) * self.V_MAX * self.DT
        self.th = np.clip(self.th, -self.LIM, self.LIM)
        self.steps += 1
        d = self._dist()
        self._dmin = min(self._dmin, d)
        reward = (self._prev - d) / max(self._d0, 1e-6)
        self._prev = d
        success = d < self.GOAL_R
        done = success or self.steps >= self.max_steps
        info = {"success": bool(success), "crash": False, "dist": d,
                "progress": (self._d0 - min(self._dmin, self._d0)) / max(self._d0, 1e-6)}
        if success:
            reward += 1.0
        return self.obs(), reward, done, info


BODIES = {"rover": Rover, "arm": Arm}


def arm_expert_action(body: "Arm"):
    """Jacobian-transpose reaching expert: full-strength step down the
    distance gradient. Returns (left, right) in [-1, 1]."""
    d0 = body._dist()
    grad = []
    for j in range(2):
        old = body.th[j]
        body.th[j] = old + 1e-3
        d1 = body._dist()
        body.th[j] = old
        grad.append((d1 - d0) / 1e-3)
    grad = np.asarray(grad)
    n = float(np.linalg.norm(grad))
    v = (-grad / max(n, 1e-9)).tolist() if n > 1e-9 else [0.0, 0.0]
    return float(np.clip(v[0], -1, 1)), float(np.clip(v[1], -1, 1))
