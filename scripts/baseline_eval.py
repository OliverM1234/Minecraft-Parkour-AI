"""Measure the ORIGINAL project's trained agent on the ORIGINAL course.

Uses the original env (train_parkour.make_env), checkpoint and observation
pipeline, so this is a like-for-like "before" number.
"""
import json, os, sys, time
os.environ.pop("MINEDOJO_HIDE_GUI", None)
os.environ["MINEDOJO_HEADLESS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_parkour import make_env

CKPT = "checkpoints/parkour_ppo_latest.zip"
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
env = DummyVecEnv([make_env])
model = PPO.load(CKPT, env=env, device="cpu")
rows = []
t0 = time.time()
for ep in range(EPISODES):
    obs = env.reset(); done = False; steps = 0; ret = 0.0; plat = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = env.step(action)
        done = bool(dones[0]); steps += 1; ret += float(r[0])
        plat = max(plat, int(infos[0].get("platform_idx", 0)))
    rows.append(dict(platforms=plat, decisions=steps, reward=round(ret, 2)))
    print(ep, rows[-1], flush=True)
env.close()
out = dict(checkpoint=CKPT, episodes=EPISODES, platforms_mean=float(np.mean([r["platforms"] for r in rows])),
           platforms_max=int(max(r["platforms"] for r in rows)), finished=sum(r["platforms"] >= 7 for r in rows), seconds=round(time.time() - t0, 1), rows=rows)
json.dump(out, open("runs/baseline.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"}, flush=True)
