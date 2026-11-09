"""Measure how many ticks the camera image trails the game state reported on the same step."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from parkour import courses as C
from parkour.real_env import RealGame

game = RealGame(image_size=128)
game.hard_reset()
game.build(C.ZIGZAG, "gold_block")
game.respawn()

def frame():
    return game.frame().astype(np.float32)

def settle(n=12):
    for _ in range(n):
        game._noop()

def closest(frames, a, b):
    """For each frame: position between reference a (0) and reference b (1)."""
    out = []
    d = (b - a).ravel()
    for f in frames:
        out.append(float(np.dot((f - a).ravel(), d) / (np.dot(d, d) + 1e-6)))
    return out

results = {"yaw": [], "teleport": [], "walk": []}
for trial in range(8):
    settle()
    y0 = game.state()[1]
    F0 = frame()
    frames, yaws = [], []
    delta = 12.0 if trial % 2 == 0 else -12.0
    a = game.env.action_space.no_op()
    a["camera"] = np.array([0.0, delta], dtype=np.float32)
    game.obs, _, _, _ = game.env.step(a)
    frames.append(frame()); yaws.append(game.state()[1])
    for _ in range(7):
        game._noop(); frames.append(frame()); yaws.append(game.state()[1])
    settle()
    F1 = frame()
    alpha = closest(frames, F0, F1)
    state_tick = next(i for i, v in enumerate(yaws) if abs(v - y0) > 1e-3)
    img_tick = next((i for i, v in enumerate(alpha) if v > 0.5), None)
    print("yaw", trial, "state changes at", state_tick, "image passes halfway at", img_tick, [round(x, 2) for x in alpha[:5]], flush=True)
    results["yaw"].append([state_tick, img_tick])

for trial in range(6):
    game.respawn()
    settle()
    F0 = frame()
    p0 = game.state()[0].copy()
    frames, ps = [], []
    sx, sy, sz = C.spawn_point(game.course)
    game._chat(f"/tp @p {sx + 0.3:.3f} {sy:.3f} {sz + 0.2:.3f} {0.0:.3f} 40.000")
    frames.append(frame()); ps.append(game.state()[0].copy())
    for _ in range(7):
        game._noop(); frames.append(frame()); ps.append(game.state()[0].copy())
    settle()
    F1 = frame()
    alpha = closest(frames, F0, F1)
    state_tick = next(i for i, v in enumerate(ps) if np.abs(v - p0).max() > 1e-3)
    img_tick = next((i for i, v in enumerate(alpha) if v > 0.5), None)
    print("teleport", trial, "state changes at", state_tick, "image passes halfway at", img_tick, [round(x, 2) for x in alpha[:6]], flush=True)
    results["teleport"].append([state_tick, img_tick])
game.close()
json.dump(results, open("runs/frame_lag.json", "w"))
yaw = np.array(results["yaw"], float); tp = np.array(results["teleport"], float)
print("frames behind state: yaw", np.mean(yaw[:, 1] - yaw[:, 0]), "teleport", np.mean(tp[:, 1] - tp[:, 0]))
