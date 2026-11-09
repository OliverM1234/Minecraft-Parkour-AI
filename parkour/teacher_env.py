"""Vectorised training environment for the state-based teacher."""
from __future__ import annotations

import time

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv

from . import courses as C
from .features import (
    ACTION_DIMS,
    DECISION_TICKS,
    FALL_MARGIN,
    OBS_SIZE,
    YAW_CHOICES,
    build_obs,
    camera_only_view,
    supported_by,
)
from .sim import Kinematics

MAX_PLATFORMS = 14
MAX_DECISIONS = 400
LANDING_REWARD = 1.0
FINISH_REWARD = 2.0
FALL_REWARD = -1.0
TIME_REWARD = -0.002
PROGRESS_WEIGHT = 0.1


class TeacherVecEnv(VecEnv):
    def __init__(self, n_envs=512, seed=0, difficulty=0.15, adaptive=True, zigzag_fraction=0.08, noise=True,
                 max_delay=0, camera_only=False, smooth=0.0, obs_noise=0.0):
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.action_space = spaces.MultiDiscrete(list(ACTION_DIMS))
        super().__init__(n_envs, self.observation_space, self.action_space)
        self.rng = np.random.default_rng(seed)
        self.kin = Kinematics(n_envs, MAX_PLATFORMS)
        self.difficulty = difficulty
        self.adaptive = adaptive
        self.zigzag_fraction = zigzag_fraction
        self.noise = noise
        self.centers = np.zeros((n_envs, MAX_PLATFORMS, 3))
        self.halves = np.zeros((n_envs, MAX_PLATFORMS, 2))
        self.counts = np.zeros(n_envs, int)
        self.cur = np.zeros(n_envs, int)
        self.steps = np.zeros(n_envs, int)
        self.ep_return = np.zeros(n_envs)
        self.disp = np.zeros((n_envs, 3))
        self.prev_action = np.zeros((n_envs, 3), dtype=np.float32)
        self.prev_action[:, 2] = 3
        self.prev_dist = np.zeros(n_envs)
        self.success_ewma = 0.0
        self.completed = 0
        self.t0 = time.time()
        self._actions = None
        self.is_zigzag = np.zeros(n_envs, bool)
        # Options that make the teacher easier to imitate from pixels: it can see the
        # world a few ticks late, sees only what a camera could reveal (no last action,
        # no absolute heading), and pays for jerky control.
        self.max_delay = int(max_delay)
        self.camera_only = camera_only
        self.smooth = float(smooth)
        self.obs_noise = float(obs_noise)
        h = self.max_delay + 1
        self.delay = np.zeros(n_envs, int)
        self.h_pos = np.zeros((n_envs, h, 3))
        self.h_disp = np.zeros((n_envs, h, 3))
        self.h_yaw = np.zeros((n_envs, h))
        self.h_ground = np.zeros((n_envs, h), bool)
        self.h_cur = np.zeros((n_envs, h), int)
        self.last_turn = np.full(n_envs, 3)
        self.last_forward = np.zeros(n_envs, int)

    # ------------------------------------------------------------------
    def _load_course(self, i, course):
        self.counts[i] = len(course)
        self.kin.box_active[i] = False
        for j, (x, y, z, wx, wz) in enumerate(course):
            self.kin.box_min[i, j] = (x, y, z)
            self.kin.box_max[i, j] = (x + wx, y + 1, z + wz)
            self.kin.box_active[i, j] = True
            self.centers[i, j] = (x + wx / 2.0, y + 1.0, z + wz / 2.0)
            self.halves[i, j] = (wx / 2.0, wz / 2.0)

    def _reset_env(self, i):
        if self.rng.random() < self.zigzag_fraction:
            course = C.ZIGZAG
            self.is_zigzag[i] = True
        else:
            course = C.random_course(self.rng, self.difficulty)
            self.is_zigzag[i] = False
        self._load_course(i, course)
        sx, sy, sz = C.spawn_point(course)
        yaw = 0.0
        if self.noise:
            hx, hz = self.halves[i, 0]
            sx += self.rng.uniform(-min(hx, 0.25), min(hx, 0.25))
            sz += self.rng.uniform(-min(hz, 0.25), min(hz, 0.25))
            yaw = self.rng.uniform(-15.0, 15.0)
        self.kin.place(i, (sx, sy, sz), yaw)
        self.cur[i] = 0
        self.steps[i] = 0
        self.ep_return[i] = 0.0
        self.disp[i] = 0.0
        self.prev_action[i] = (0, 0, 3)
        self.prev_dist[i] = self._dist(i)
        self.delay[i] = self.rng.integers(0, self.max_delay + 1)
        self.h_pos[i] = self.kin.pos[i]
        self.h_disp[i] = 0.0
        self.h_yaw[i] = self.kin.yaw[i]
        self.h_ground[i] = self.kin.on_ground[i]
        self.h_cur[i] = 0
        self.last_turn[i] = 3
        self.last_forward[i] = 0

    def _dist(self, i):
        target = min(self.cur[i] + 1, self.counts[i] - 1)
        return min(6.0, np.hypot(*(self.centers[i, target][[0, 2]] - self.kin.pos[i, [0, 2]])))

    def _dist_all(self):
        rows = np.arange(self.num_envs)
        target = np.minimum(self.cur + 1, self.counts - 1)
        delta = self.centers[rows, target][:, [0, 2]] - self.kin.pos[:, [0, 2]]
        return np.minimum(6.0, np.hypot(delta[:, 0], delta[:, 1]))

    def _push_history(self):
        if self.max_delay == 0:
            return
        for name, value in (("h_pos", self.kin.pos), ("h_disp", self.disp), ("h_yaw", self.kin.yaw),
                            ("h_ground", self.kin.on_ground), ("h_cur", self.cur)):
            arr = getattr(self, name)
            arr[:, 1:] = arr[:, :-1]
            arr[:, 0] = value

    def _obs(self):
        prev = self.prev_action
        if self.max_delay == 0:
            obs = build_obs(self.kin.pos, self.disp, self.kin.yaw, self.kin.on_ground, prev,
                            self.cur, self.centers, self.halves, self.counts)
        else:
            rows = np.arange(self.num_envs)
            d = self.delay
            obs = build_obs(self.h_pos[rows, d], self.h_disp[rows, d], self.h_yaw[rows, d], self.h_ground[rows, d], prev,
                            self.h_cur[rows, d], self.centers, self.halves, self.counts)
        if self.camera_only:
            obs = camera_only_view(obs)
        if self.obs_noise > 0:
            n = self.num_envs
            for k in range(4):
                obs[:, 6 * k] += self.rng.normal(0, self.obs_noise / 5.0, n)
                obs[:, 6 * k + 1] += self.rng.normal(0, self.obs_noise / 5.0, n)
            obs[:, 24:26] += self.rng.normal(0, self.obs_noise / 2.0, (n, 2))
        return obs

    # ------------------------------------------------------------------
    def reset(self):
        for i in range(self.num_envs):
            self._reset_env(i)
        return self._obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=int)

    def step_wait(self):
        n = self.num_envs
        act = self._actions.copy()
        if self.noise:
            flip = self.rng.random(n) < 0.03
            act[flip, 2] = self.rng.integers(0, len(YAW_CHOICES), flip.sum())
        forward = act[:, 0].astype(int)
        jump = act[:, 1].astype(bool)
        yaw_delta = YAW_CHOICES[act[:, 2]] / DECISION_TICKS
        zero = np.zeros(n, int)
        sprint = forward.astype(bool)
        rows = np.arange(n)
        reward = np.full(n, TIME_REWARD)
        done = np.zeros(n, bool)
        finished = np.zeros(n, bool)
        for _ in range(DECISION_TICKS):
            before = self.kin.pos.copy()
            self.kin.tick(forward, zero, jump, sprint, yaw_delta)
            self.disp = self.kin.pos - before
            landed = supported_by(self.kin.pos, self.centers, self.halves, self.counts) * self.kin.on_ground
            landed[~self.kin.on_ground] = -1
            improved = landed > self.cur
            if improved.any():
                reward[improved] += LANDING_REWARD * (landed[improved] - self.cur[improved])
                self.cur[improved] = landed[improved]
                self.prev_dist[improved] = self._dist_all()[improved]
            target = np.minimum(self.cur + 1, self.counts - 1)
            top = np.minimum(self.centers[rows, self.cur, 1], self.centers[rows, target, 1])
            fell = self.kin.pos[:, 1] < top - FALL_MARGIN
            reward[fell] += FALL_REWARD
            done |= fell
            end = self.cur >= self.counts - 1
            newly = end & ~done
            reward[newly] += FINISH_REWARD
            finished |= newly
            done |= newly
            self._push_history()
        dist = self._dist_all()
        reward += PROGRESS_WEIGHT * (self.prev_dist - dist)
        if self.smooth > 0:
            chosen_turn = YAW_CHOICES[self._actions[:, 2]] / 20.0
            last_turn = YAW_CHOICES[self.last_turn] / 20.0
            reward -= self.smooth * (chosen_turn ** 2 + 0.5 * np.abs(chosen_turn - last_turn))
            reward -= 0.3 * self.smooth * (self._actions[:, 0] != self.last_forward)
            self.last_turn = self._actions[:, 2].astype(int)
            self.last_forward = self._actions[:, 0].astype(int)
        self.prev_dist = dist
        self.prev_action = self._actions.astype(np.float32)
        self.steps += 1
        self.ep_return += reward
        timeout = self.steps >= MAX_DECISIONS
        all_done = done | timeout
        obs = self._obs()
        infos = [{} for _ in range(n)]
        idx = np.nonzero(all_done)[0]
        for i in idx:
            infos[i]["terminal_observation"] = obs[i].copy()
            infos[i]["TimeLimit.truncated"] = bool(timeout[i] and not done[i])
            infos[i]["episode"] = {"r": float(self.ep_return[i]), "l": int(self.steps[i]), "t": time.time() - self.t0}
            infos[i]["is_success"] = bool(finished[i])
            infos[i]["platforms"] = int(self.cur[i])
            infos[i]["zigzag"] = bool(self.is_zigzag[i])
            self.completed += 1
            self.success_ewma += 0.003 * (float(finished[i]) - self.success_ewma)
            self._reset_env(i)
        if len(idx):
            obs = self._obs()
        if self.adaptive and self.success_ewma > 0.75 and self.completed > 2000 and self.difficulty < 1.0:
            self.difficulty = min(1.0, self.difficulty + 0.05)
            self.success_ewma = 0.4
            self.completed = 0
        return obs, reward.astype(np.float32), all_done, infos

    # ------------------------------------------------------------------
    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [getattr(self, name)] * self.num_envs

    def set_attr(self, name, value, indices=None):
        setattr(self, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
