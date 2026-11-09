"""Final evaluation in the real game on courses never used in training.

Runs the vision-only student (and optionally the teacher, for reference) on
held-out procedural courses and on the original course from the first version
of the project. Writes a JSON report with per-episode results.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402

from parkour import courses as C  # noqa: E402
from parkour.features import camera_only_view  # noqa: E402
from parkour.real_env import BLOCKS, RealGame, preprocess_rgb  # noqa: E402
from parkour.student import FrameStack, StudentNet, act  # noqa: E402

from scripts_common import HOLDOUT_SEED  # noqa: E402


def run_student(game, net, course, block, offset=(0.0, 0.0), max_decisions=400):
    game.build(course, block)
    game.respawn(offset=offset)
    stack = FrameStack()
    stack.reset(preprocess_rgb(game.frame()))
    info = {"finished": False, "fell": False, "platform": 0}
    n = 0
    for n in range(1, max_decisions + 1):
        action, _ = act(net, stack.array(), deterministic=True)
        done, info = game.step(action)
        stack.push(preprocess_rgb(game.frame()))
        if done:
            break
    return dict(finished=bool(info.get("finished", False)), platform=int(info.get("platform", 0)), decisions=n,
                game_seconds=n * 2 / 20.0)


def run_teacher(game, model, course, block, offset=(0.0, 0.0), max_decisions=400, camera_only=False):
    game.build(course, block)
    game.respawn(offset=offset)
    info = {"finished": False, "fell": False, "platform": 0}
    n = 0
    for n in range(1, max_decisions + 1):
        obs = game.teacher_obs()
        action, _ = model.predict(camera_only_view(obs) if camera_only else obs, deterministic=True)
        done, info = game.step(action[0])
        if done:
            break
    return dict(finished=bool(info.get("finished", False)), platform=int(info.get("platform", 0)), decisions=n,
                game_seconds=n * 2 / 20.0)


def summarise(rows, length=None):
    finished = np.array([r["finished"] for r in rows], float)
    out = dict(episodes=len(rows), finished=int(finished.sum()), success_rate=round(float(finished.mean()), 4))
    if length is not None:
        out["mean_platforms"] = round(float(np.mean([r["platform"] for r in rows])), 3)
    p = finished.mean()
    out["ci95"] = round(float(1.96 * np.sqrt(max(p * (1 - p), 1e-9) / len(rows))), 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="runs/student/student_latest.pt")
    p.add_argument("--teacher", default="runs/teacher/teacher_final.zip")
    p.add_argument("--courses", type=int, default=100)
    p.add_argument("--zigzag", type=int, default=50)
    p.add_argument("--image", type=int, default=128)
    p.add_argument("--out", default="runs/evaluation.json")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--camera-only-teacher", action="store_true", help="teacher was trained on camera-visible inputs only")
    p.add_argument("--seed", type=int, default=HOLDOUT_SEED, help="course seed; use a validation seed when choosing between checkpoints")
    a = p.parse_args()

    torch.set_num_threads(2)
    net = StudentNet()
    net.load_state_dict(torch.load(a.student)["net"], strict=False)
    net.eval()
    model = None if a.skip_teacher else PPO.load(a.teacher, device="cpu")
    game = RealGame(image_size=a.image)
    game.hard_reset()

    rng = np.random.default_rng(a.seed)
    report = dict(seed=a.seed, image=a.image, student=a.student, details={})
    courses = []
    for _ in range(a.courses):
        courses.append((C.random_course(rng, 1.0), str(rng.choice(BLOCKS))))

    t0 = time.time()
    student_rows = []
    for i, (course, block) in enumerate(courses):
        r = run_student(game, net, course, block)
        r["length"] = len(course)
        student_rows.append(r)
        print("student", i, len(course), r, flush=True)
    report["student_unseen"] = summarise(student_rows, 1)
    report["details"]["student_unseen"] = student_rows

    zig = []
    for i in range(a.zigzag):
        offset = (float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.2, 0.2))) if i else (0.0, 0.0)
        zig.append(run_student(game, net, C.ZIGZAG, "quartz_block", offset=offset))
        print("zigzag", i, zig[-1], flush=True)
    report["student_zigzag"] = summarise(zig, 1)
    report["details"]["student_zigzag"] = zig

    if model is not None:
        teacher_rows = []
        for i, (course, block) in enumerate(courses):
            r = run_teacher(game, model, course, block, camera_only=a.camera_only_teacher)
            r["length"] = len(course)
            teacher_rows.append(r)
            print("teacher", i, len(course), r, flush=True)
        report["teacher_unseen"] = summarise(teacher_rows, 1)
        report["details"]["teacher_unseen"] = teacher_rows

    report["seconds"] = round(time.time() - t0, 1)
    game.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print({k: v for k, v in report.items() if k != "details"}, flush=True)


if __name__ == "__main__":
    main()
