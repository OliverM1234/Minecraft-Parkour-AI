"""Replay key sequences recorded in the real game through the fast simulator.

Usage: validate_sim.py [runs/physics/physics_real.json] [out.json]

Prints the per-tick position error for each recorded experiment.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parkour.sim import Kinematics  # noqa: E402

EPISODE_FLOOR = 4.4  # platform top (5.0) minus the fall margin used in training


def replay(exp):
    trace = np.array(exp["trace"])
    boxes = np.array(exp["boxes"], float)
    k = Kinematics(1, len(boxes))
    for j, (lo, hi) in enumerate(boxes):
        k.box_min[0, j] = lo
        k.box_max[0, j] = hi
        k.box_active[0, j] = True
    k.place(0, trace[0, :3], trace[0, 3])
    out = [np.r_[k.pos[0], k.yaw[0]]]
    for a in exp["actions"]:
        forward = np.array([a.get("forward", 0)])
        strafe = np.array([a.get("left", 0) - a.get("right", 0)])
        jump = np.array([bool(a.get("jump", 0))])
        sprint = np.array([bool(a.get("sprint", 0))])
        yaw = np.array([float(a.get("yaw", 0.0))])
        k.tick(forward, strafe, jump, sprint, yaw)
        out.append(np.r_[k.pos[0], k.yaw[0]])
    return trace, np.array(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/physics/physics_real.json"
    data = json.load(open(path))
    report = {}
    print(f"{'experiment':24s} {'ticks':>5s} {'max err (blocks)':>17s} {'mean err':>10s}")
    for name, exp in data.items():
        trace, sim = replay(exp)
        err = np.linalg.norm(trace[:, :3] - sim[:, :3], axis=1)
        # A run is over once the player drops below platform level, so only
        # compare up to that point (the floor far below is not modelled).
        below = np.nonzero(trace[:, 1] < EPISODE_FLOOR)[0]
        end = int(below[0]) + 1 if len(below) else len(trace)
        err = err[:end]
        report[name] = {"ticks": len(err) - 1, "max_error": float(err.max()), "mean_error": float(err.mean())}
        print(f"{name:24s} {len(err) - 1:5d} {err.max():17.5f} {err.mean():10.5f}")
    worst = max(v["max_error"] for v in report.values())
    print(f"worst case: {worst:.5f} blocks across {sum(v['ticks'] for v in report.values())} ticks")
    if len(sys.argv) > 2:
        json.dump(report, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
