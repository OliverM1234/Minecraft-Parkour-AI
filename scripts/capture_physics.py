"""Record per-tick player traces from the real game for scripted key sequences.

The traces are compared against the fast simulator by validate_sim.py. Saves to
runs/physics/physics_real.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parkour.real_env import PITCH, RealGame  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "physics")
os.makedirs(OUT, exist_ok=True)
BEDROCK = [[-100, 0, -100], [100, 1, 100]]
TOP = 5.0

game = RealGame(image_size=64)
game.hard_reset()
env = game.env
experiments = {}


def key(forward=0, sprint=0, jump=0, left=0, right=0, yaw=0.0):
    a = env.action_space.no_op()
    a["forward"] = forward
    a["sprint"] = sprint
    a["jump"] = jump
    a["left"] = left
    a["right"] = right
    a["camera"] = np.array([0.0, yaw], dtype=np.float32)
    return a


def state():
    pos, yaw, _ = game.state()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(yaw)]


def run(name, script, boxes, start=(0.5, TOP, 0.5, 0.0)):
    game._chat(f"/tp @p {start[0]} {start[1]} {start[2]} {start[3]} {PITCH}")
    for _ in range(6):
        game._noop()
    trace = [state()]
    for step in script:
        game.obs, _, _, _ = env.step(key(**step))
        trace.append(state())
    experiments[name] = {"actions": script, "trace": trace, "boxes": boxes}
    print(name, "final", [round(v, 3) for v in trace[-1]], flush=True)


def fill(x1, y1, z1, x2, y2, z2, block):
    game._chat(f"/fill {x1} {y1} {z1} {x2} {y2} {z2} {block}")


# Wide pad: free movement ----------------------------------------------------
fill(-30, 4, -20, 30, 4, 80, "quartz_block")
game._noop(20)
pad = [[-30, 4, -20], [31, 5, 81]]
sets = {
    "walk40": [dict(forward=1)] * 40,
    "sprint40": [dict(forward=1, sprint=1)] * 40,
    "sprint_jump60": [dict(forward=1, sprint=1, jump=1)] * 60,
    "walk_jump60": [dict(forward=1, jump=1)] * 60,
    "sprint_jump_release": [dict(forward=1, sprint=1)] * 10 + [dict(forward=1, sprint=1, jump=1)] + [dict(forward=1, sprint=1)] * 6 + [dict()] * 25,
    "stand_jump30": [dict(jump=1)] * 30,
    "diag_sprint_jump60": [dict(forward=1, sprint=1, jump=1, left=1)] * 60,
    "strafe40": [dict(left=1)] * 40,
    "turn_sprint60": [dict(forward=1, sprint=1, yaw=4.0)] * 60,
    "turn_sprint_jump60": [dict(forward=1, sprint=1, jump=1, yaw=-3.0)] * 60,
    "stop_after_sprint": [dict(forward=1, sprint=1)] * 20 + [dict()] * 30,
}
for name, script in sets.items():
    run(name, script, [pad, BEDROCK])

# Single-block platforms: edges, falls and real landings ---------------------
fill(-30, 4, -20, 30, 4, 80, "air")
game._chat("/setblock 0 4 0 quartz_block")
game._chat("/setblock 0 4 3 quartz_block")
game._chat("/setblock 0 4 7 quartz_block")
game._noop(20)
blocks = [[[0, 4, 0], [1, 5, 1]], [[0, 4, 3], [1, 5, 4]], [[0, 4, 7], [1, 5, 8]], BEDROCK]
run("edge_walk_off", [dict(forward=1, sprint=1)] * 12 + [dict()] * 28, blocks)
run("hop_landing", [dict(forward=1, sprint=1)] * 2 + [dict(forward=1, sprint=1, jump=1)] * 2 + [dict(forward=1, sprint=1)] * 40, blocks)
run("hop_then_chain", [dict(forward=1, sprint=1)] * 2 + [dict(forward=1, sprint=1, jump=1)] * 60, blocks)
run("edge_corner_stand", [dict()] * 20, blocks, start=(0.72, TOP, 0.72, 0.0))
run("corner_hop", [dict(forward=1, sprint=1, jump=1)] * 40, blocks, start=(0.2, TOP, 0.6, 12.0))

game.close()
json.dump(experiments, open(os.path.join(OUT, "physics_real.json"), "w"))
print("saved", len(experiments), "experiments", flush=True)
