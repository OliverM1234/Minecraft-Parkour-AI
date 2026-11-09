"""Record one attempt of the ORIGINAL agent (original checkpoint, env and 64x64
observation) so the video can show the starting point honestly."""
import os, pickle, sys
os.environ.pop("MINEDOJO_HIDE_GUI", None)
os.environ["MINEDOJO_HEADLESS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2, gym, numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_parkour import (ActionRepeatWrapper, GrayscaleResizeObservation, ParkourActionWrapper,
                           ParkourRewardWrapper, make_base_env)

class Capture(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self.frames = []
    def step(self, a):
        o, r, d, i = self.env.step(a); self.frames.append(o["rgb"].transpose(1, 2, 0).copy()); return o, r, d, i
    def reset(self, **k):
        o = self.env.reset(**k); self.frames.append(o["rgb"].transpose(1, 2, 0).copy()); return o

env = make_base_env(image_size=(64, 64))
env = ParkourActionWrapper(env); env = ParkourRewardWrapper(env)
cap = Capture(env)
env = ActionRepeatWrapper(cap, repeat=4); env = GrayscaleResizeObservation(env, 64); env = TimeLimit(env, 200)
venv = DummyVecEnv([lambda: env])
model = PPO.load("checkpoints/parkour_ppo_latest.zip", env=venv, device="cpu")
obs = venv.reset(); done = False; plat = 0
n = 0
while not done:
    a, _ = model.predict(obs, deterministic=True)
    obs, r, dones, infos = venv.step(a)
    done = bool(dones[0]); plat = max(plat, int(infos[0].get("platform_idx", 0))); n += 1
venv.close()
out = {"frames": cap.frames, "platform": plat, "decisions": n}
os.makedirs("runs/demo_before", exist_ok=True)
pickle.dump(out, open("runs/demo_before/before.pkl", "wb"))
cv2.imwrite("media/old_cnn_view.png", cv2.resize(cap.frames[1][:, :, ::-1], (512, 512), interpolation=cv2.INTER_NEAREST))
print("frames", len(cap.frames), "platform", plat, "decisions", n, flush=True)
