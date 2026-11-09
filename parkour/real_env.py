"""Thin client for the real game: fast resets, course building, true state.

Key speed-up over the original project: an episode restart is a teleport inside
the running mission (about 0.04 s) instead of a full mission reset (about 15 s).
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

os.environ.setdefault("MINEDOJO_HEADLESS", "1")
os.environ.setdefault("MINEDOJO_HIDE_GUI", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from minedojo.sim import MineDojoSim  # noqa: E402

from . import courses as C  # noqa: E402
from .features import DECISION_TICKS, FALL_MARGIN, YAW_CHOICES, build_obs, course_arrays, supported_by  # noqa: E402

PITCH = 40.0
BLOCKS = ["quartz_block", "gold_block", "emerald_block", "lapis_block", "redstone_block", "diamond_block", "iron_block"]
CLEAR_MIN, CLEAR_MAX = (-40, 1, -8), (40, 9, 72)


class RealGame:
    def __init__(self, image_size: int = 128, max_platforms: int = 14):
        self.image_size = image_size
        self.max_platforms = max_platforms
        self.env = MineDojoSim(
            image_size=(image_size, image_size),
            generate_world_type="flat",
            flat_world_seed_string="1;7;1",
            drawing_str='<DrawCuboid x1="0" y1="4" z1="0" x2="0" y2="4" z2="0" type="quartz_block"/>',
            start_position={"x": 0.5, "y": 5.0, "z": 0.5, "yaw": 0.0, "pitch": PITCH},
            start_time=6000,
            allow_time_passage=False,
            allow_mob_spawn=False,
            seed=0,
        )
        self.obs = None
        self.course: Optional[List[C.Platform]] = None
        self.placed: List[C.Platform] = []
        self.centers = self.halves = self.counts = None
        self.cur = 0
        self.prev_pos = None
        self.disp = np.zeros((1, 3))
        self.prev_action = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
        self.yaw = 0.0
        self._pending_sat = True

    # Low-level -------------------------------------------------------------
    def _chat(self, cmd: str):
        """Send a chat command. Goes through step() directly because MineDojo's
        execute_cmd whitelist rejects /fill and /effect."""
        action = self.env.action_space.no_op()
        action["chat"] = cmd
        self.obs, _, _, _ = self.env.step(action)
        return self.obs

    def _noop(self, ticks: int = 1):
        for _ in range(ticks):
            self.obs, _, _, _ = self.env.step(self.env.action_space.no_op())
        return self.obs

    def state(self):
        loc = self.obs["location_stats"]
        pos = np.array(loc["pos"], dtype=float).reshape(3)
        yaw = float(np.array(loc["yaw"]).item())
        pitch = float(np.array(loc["pitch"]).item())
        return pos, yaw, pitch

    def frame(self):
        """RGB frame, shape (H, W, 3), uint8."""
        return self.obs["rgb"].transpose(1, 2, 0)

    # Episode management ----------------------------------------------------
    def hard_reset(self):
        self.obs = self.env.reset()
        self._chat("/gamerule sendCommandFeedback false")
        self._chat("/gamerule doFireTick false")
        self._chat("/effect @p minecraft:saturation 1000000 255 true")
        self._chat("/effect @p minecraft:resistance 1000000 255 true")
        self.placed = [(0, 4, 0, 1, 1)]
        return self.obs

    def _clear(self):
        for x, y, z, wx, wz in self.placed:
            if wx == 1 and wz == 1:
                self._chat(f"/setblock {x} {y} {z} air")
            else:
                self._chat(f"/fill {x} {y} {z} {x + wx - 1} {y} {z + wz - 1} air")
        self.placed = []

    def build(self, course: List[C.Platform], block: str = "quartz_block"):
        # Stand near the origin first so every course chunk is loaded.
        self._chat("/tp @p 0.5 12 0.5")
        self._clear()
        for command in C.course_commands(course, block):
            self._chat(command)
        self.placed = list(course)
        self.course = course
        self.centers, self.halves, self.counts = course_arrays([course], self.max_platforms)

    def respawn(self, offset=(0.0, 0.0), yaw: float = 0.0, settle: int = 6):
        """Teleport to the start platform, zeroing all motion."""
        sx, sy, sz = C.spawn_point(self.course)
        x, z = sx + offset[0], sz + offset[1]
        self._chat(f"/tp @p {x:.4f} {sy:.4f} {z:.4f} {yaw:.3f} {PITCH:.3f}")
        for _ in range(30):
            pos, y_, p_ = self.state()
            if abs(pos[0] - x) < 1e-3 and abs(pos[2] - z) < 1e-3 and abs(pos[1] - sy) < 1e-3:
                break
            self._noop()
        self._noop(settle)
        pos, yaw_now, _ = self.state()
        self.prev_pos = pos.copy()
        self.disp = np.zeros((1, 3))
        self.cur = 0
        self.yaw = yaw_now
        self.prev_action = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
        return self.obs

    # Control ---------------------------------------------------------------
    def teacher_obs(self):
        pos, yaw, _ = self.state()
        on_ground = supported_by(pos[None], self.centers, self.halves, self.counts) >= 0
        return build_obs(pos[None], self.disp, np.array([yaw]), on_ground, self.prev_action,
                         np.array([self.cur]), self.centers, self.halves, self.counts)

    def step(self, action, capture=None):
        """Apply one decision (forward, jump, turn index). Returns (done, info)."""
        forward, jump, turn = int(action[0]), int(action[1]), int(action[2])
        yaw_delta = float(YAW_CHOICES[turn]) / DECISION_TICKS
        info = {"finished": False, "fell": False}
        done = False
        for _ in range(DECISION_TICKS):
            a = self.env.action_space.no_op()
            a["forward"] = forward
            a["sprint"] = forward
            a["jump"] = jump
            a["camera"] = np.array([0.0, yaw_delta], dtype=np.float32)
            self.obs, _, env_done, _ = self.env.step(a)
            if env_done:
                info["env_done"] = True
                done = True
            pos, yaw, _ = self.state()
            self.disp = (pos - self.prev_pos)[None]
            self.prev_pos = pos.copy()
            self.yaw = yaw
            standing = supported_by(pos[None], self.centers, self.halves, self.counts)[0]
            if standing > self.cur:
                self.cur = int(standing)
            if capture is not None:
                capture(self)
            target = min(self.cur + 1, int(self.counts[0]) - 1)
            top = min(self.centers[0, self.cur, 1], self.centers[0, target, 1])
            if pos[1] < top - FALL_MARGIN:
                info["fell"] = True
                done = True
                break
            if self.cur >= self.counts[0] - 1:
                info["finished"] = True
                done = True
                break
        self.prev_action = np.array([[forward, jump, turn]], dtype=np.float32)
        info["platform"] = self.cur
        return done, info

    def close(self):
        self.env.close()


def preprocess_rgb(frame_rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """Downscale an RGB frame to the student's input: channels-first uint8."""
    small = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return small.transpose(2, 0, 1)
