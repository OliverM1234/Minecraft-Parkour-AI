"""Fast batched model of the block-game player kinematics used by the teacher.

There is no rendering here: this is a state-only integrator (position, motion,
yaw) with axis-aligned box collisions, written so thousands of parallel
episodes run on a laptop CPU. The constants and the order of operations follow
the Minecraft 1.11.2 movement code, and `scripts/validate_sim.py` compares this
model tick-by-tick against traces recorded from the real game.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 0.08
Y_DRAG = 0.98000001907
GROUND_DRAG = 0.6 * 0.91
AIR_DRAG = 0.91
JUMP_VELOCITY = 0.42
SPRINT_JUMP_BOOST = 0.2
WALK_SPEED = 0.1
SPRINT_MULTIPLIER = 1.3
AIR_ACCEL = 0.02
SPRINT_AIR_ACCEL = 0.026
INPUT_SCALE = 0.98
MOTION_EPS = 0.003
HALF_WIDTH = 0.3
PLAYER_HEIGHT = 1.8
JUMP_COOLDOWN = 10
DEG = np.pi / 180.0


class Kinematics:
    """Batched player state for ``n`` independent worlds of up to ``m`` boxes."""

    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.yaw = np.zeros(n)
        self.on_ground = np.zeros(n, bool)
        self.sprinting = np.zeros(n, bool)
        self.prev_sprinting = np.zeros(n, bool)
        self.collided_h = np.zeros(n, bool)
        self.jump_ticks = np.zeros(n, int)
        self.box_min = np.zeros((n, m, 3))
        self.box_max = np.zeros((n, m, 3))
        self.box_active = np.zeros((n, m), bool)

    def place(self, i, pos, yaw):
        self.pos[i] = pos
        self.vel[i] = (0.0, -GRAVITY * Y_DRAG, 0.0)  # resting on the ground
        self.yaw[i] = yaw
        self.on_ground[i] = True
        self.sprinting[i] = False
        self.prev_sprinting[i] = False
        self.collided_h[i] = False
        self.jump_ticks[i] = 0

    def _clip(self, axis, delta, lo, hi):
        """Longest movement along ``axis`` that does not enter any box."""
        a, b = [k for k in range(3) if k != axis]
        overlap = (
            self.box_active
            & (self.box_max[..., a] > lo[:, None, a])
            & (self.box_min[..., a] < hi[:, None, a])
            & (self.box_max[..., b] > lo[:, None, b])
            & (self.box_min[..., b] < hi[:, None, b])
        )
        out = delta.copy()
        positive = delta > 0
        if positive.any():
            gap = self.box_min[..., axis] - hi[:, None, axis]
            valid = overlap & (gap >= 0)
            gap = np.where(valid, gap, np.inf).min(axis=1)
            out[positive] = np.minimum(delta[positive], gap[positive])
        negative = delta < 0
        if negative.any():
            gap = self.box_max[..., axis] - lo[:, None, axis]
            valid = overlap & (gap <= 0)
            gap = np.where(valid, gap, -np.inf).max(axis=1)
            out[negative] = np.maximum(delta[negative], gap[negative])
        return out

    def tick(self, forward, strafe, jump, sprint_key, yaw_delta):
        """Advance one game tick. All arguments are length-``n`` arrays."""
        self.yaw = self.yaw + yaw_delta

        move_forward = forward.astype(float)
        move_strafe = strafe.astype(float)

        # Air control uses the sprint state from the previous tick; ground
        # speed reacts to sprint in the same tick.
        air_accel = np.where(self.prev_sprinting, SPRINT_AIR_ACCEL, AIR_ACCEL)

        wants_sprint = (move_forward >= 0.8) & sprint_key
        self.sprinting = (self.sprinting | wants_sprint) & (move_forward >= 0.8) & ~self.collided_h
        land_speed = WALK_SPEED * np.where(self.sprinting, SPRINT_MULTIPLIER, 1.0)

        self.jump_ticks = np.maximum(self.jump_ticks - 1, 0)
        self.vel[np.abs(self.vel) < MOTION_EPS] = 0.0

        yaw_rad = self.yaw * DEG
        sin_yaw, cos_yaw = np.sin(yaw_rad), np.cos(yaw_rad)

        jumping = jump & self.on_ground & (self.jump_ticks == 0)
        self.vel[jumping, 1] = JUMP_VELOCITY
        boost = jumping & self.sprinting
        self.vel[boost, 0] -= sin_yaw[boost] * SPRINT_JUMP_BOOST
        self.vel[boost, 2] += cos_yaw[boost] * SPRINT_JUMP_BOOST
        self.jump_ticks[jumping] = JUMP_COOLDOWN
        self.jump_ticks[~jump] = 0

        f = move_forward * INPUT_SCALE
        s = move_strafe * INPUT_SCALE
        accel = np.where(self.on_ground, land_speed, air_accel)
        norm_sq = f * f + s * s
        moving = norm_sq >= 1e-4
        norm = np.maximum(np.sqrt(np.where(moving, norm_sq, 1.0)), 1.0)
        scale = accel / norm
        s_scaled = np.where(moving, s * scale, 0.0)
        f_scaled = np.where(moving, f * scale, 0.0)
        self.vel[:, 0] += s_scaled * cos_yaw - f_scaled * sin_yaw
        self.vel[:, 2] += f_scaled * cos_yaw + s_scaled * sin_yaw

        drag = np.where(self.on_ground, GROUND_DRAG, AIR_DRAG)

        want = self.vel.copy()
        lo = self.pos - np.array([HALF_WIDTH, 0.0, HALF_WIDTH])
        hi = self.pos + np.array([HALF_WIDTH, PLAYER_HEIGHT, HALF_WIDTH])
        moved = np.zeros_like(want)
        for axis in (1, 0, 2):
            d = self._clip(axis, want[:, axis], lo, hi)
            moved[:, axis] = d
            lo[:, axis] += d
            hi[:, axis] += d
        self.pos += moved

        hit = moved != want
        self.collided_h = hit[:, 0] | hit[:, 2]
        self.on_ground = hit[:, 1] & (want[:, 1] < 0)
        self.vel[hit[:, 0], 0] = 0.0
        self.vel[hit[:, 2], 2] = 0.0
        self.vel[hit[:, 1], 1] = 0.0

        self.vel[:, 1] = (self.vel[:, 1] - GRAVITY) * Y_DRAG
        self.vel[:, 0] *= drag
        self.vel[:, 2] *= drag
        self.prev_sprinting = self.sprinting.copy()
