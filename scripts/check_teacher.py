"""Play a teacher in the real game on random courses and describe how it moves."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402

from parkour import courses as C  # noqa: E402
from parkour.features import YAW_CHOICES, camera_only_view  # noqa: E402
from parkour.real_env import BLOCKS, RealGame  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--teacher", default="runs/teacher2/teacher_final.zip")
p.add_argument("--camera-only", action="store_true")
p.add_argument("--courses", type=int, default=60)
p.add_argument("--zigzag", type=int, default=10)
p.add_argument("--seed", type=int, default=31337)
a = p.parse_args()
torch.set_num_threads(2)
model = PPO.load(a.teacher, device="cpu")
game = RealGame(image_size=128)
game.hard_reset()
rng = np.random.default_rng(a.seed)
turns, forwards, jumps = [], [], []
entropy = []


def play(course, block):
    game.build(course, block)
    game.respawn()
    info = {}
    n = 0
    for n in range(1, 401):
        obs = game.teacher_obs()
        if a.camera_only:
            obs = camera_only_view(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs))
            probs = [d.probs[0].numpy() for d in dist.distribution]
        action = [int(np.argmax(q)) for q in probs]
        entropy.append([float(-(q * np.log(q + 1e-9)).sum()) for q in probs])
        turns.append(action[2])
        forwards.append(action[0])
        jumps.append(action[1])
        done, info = game.step(action)
        if done:
            break
    return bool(info.get("finished")), int(info.get("platform", 0)), n


rand = [play(C.random_course(rng, 1.0), str(rng.choice(BLOCKS))) for _ in range(a.courses)]
zig = [play(C.ZIGZAG, "quartz_block") for _ in range(a.zigzag)]
game.close()
print("random finished", sum(r[0] for r in rand), "/", len(rand), "mean decisions", np.mean([r[2] for r in rand]))
print("zigzag finished", sum(r[0] for r in zig), "/", len(zig))
t = np.array(turns)
print("turn histogram", dict(zip(YAW_CHOICES.tolist(), np.round(np.bincount(t, minlength=7) / len(t), 3).tolist())))
print("mean absolute turn per decision", np.abs(YAW_CHOICES[t]).mean(), "direction changes per decision", np.mean(np.sign(YAW_CHOICES[t][1:]) * np.sign(YAW_CHOICES[t][:-1]) < 0))
print("forward share", np.mean(forwards), "jump share", np.mean(jumps))
print("mean entropy forward/jump/turn", np.round(np.mean(entropy, axis=0), 3))
