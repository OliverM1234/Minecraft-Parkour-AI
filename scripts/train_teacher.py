"""Train the state-based teacher with PPO in the fast simulator."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from parkour.teacher_env import TeacherVecEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "teacher")


class Progress(BaseCallback):
    def __init__(self, every):
        super().__init__()
        self.every = every
        self.next = every
        self.t0 = time.time()
        self.window = []
        path = os.path.join(OUT, "history.json")
        self.history = json.load(open(path)) if os.path.exists(path) else []
        self.best = -1.0

    def _on_training_start(self):
        self.next = self.num_timesteps + self.every

    def _on_step(self):
        for info in self.locals["infos"]:
            if "is_success" in info:
                self.window.append((info["is_success"], info["platforms"], info["zigzag"]))
        if self.num_timesteps >= self.next and len(self.window) > 100:
            env = self.training_env
            arr = np.array(self.window[-5000:], dtype=float)
            zig = arr[arr[:, 2] > 0]
            row = dict(
                steps=int(self.num_timesteps),
                seconds=round(time.time() - self.t0, 1),
                difficulty=round(float(env.get_attr("difficulty")[0]), 2),
                success=round(float(arr[:, 0].mean()), 3),
                zigzag_success=round(float(zig[:, 0].mean()), 3) if len(zig) else None,
                platforms=round(float(arr[:, 1].mean()), 2),
            )
            self.history.append(row)
            print(json.dumps(row), flush=True)
            json.dump(self.history, open(os.path.join(OUT, "history.json"), "w"), indent=1)
            self.model.save(os.path.join(OUT, "teacher_latest"))
            if row["success"] >= self.best and row["difficulty"] >= 1.0:
                self.best = row["success"]
                self.model.save(os.path.join(OUT, "teacher_best"))
            self.window.clear()
            self.next += self.every
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--resume", default=None)
    p.add_argument("--difficulty", type=float, default=0.15)
    p.add_argument("--out", default=None)
    p.add_argument("--max-delay", type=int, default=0, help="random observation delay per episode, in ticks")
    p.add_argument("--camera-only", action="store_true")
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--obs-noise", type=float, default=0.0)
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = a.out
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    env = VecMonitor(TeacherVecEnv(a.envs, seed=a.seed, difficulty=a.difficulty, max_delay=a.max_delay,
                                   camera_only=a.camera_only, smooth=a.smooth, obs_noise=a.obs_noise))
    if a.resume:
        model = PPO.load(a.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=8192,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            device="cpu",
            seed=a.seed,
            verbose=0,
        )
    t = time.time()
    model.learn(a.steps, callback=Progress(1_000_000), reset_num_timesteps=not a.resume)
    model.save(os.path.join(OUT, "teacher_final"))
    print("done", round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    main()
