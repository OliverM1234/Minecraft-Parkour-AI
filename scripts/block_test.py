"""Does the student cope with block textures it never trained on?

Runs the same courses once per block type, so differences between block types are
not differences in course difficulty. Uses its own seed, separate from the
validation and held-out courses. The result is reported, not used to choose a model.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parkour import courses as C  # noqa: E402
from parkour.real_env import BLOCKS, RealGame  # noqa: E402
from parkour.student import StudentNet  # noqa: E402

from evaluate import run_student, summarise  # noqa: E402

SEEN = ["quartz_block"]
UNSEEN = ["brick_block", "purpur_block", "planks", "obsidian"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--courses", type=int, default=40)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--out", default="runs/block_test.json")
    a = p.parse_args()
    assert not set(UNSEEN) & set(BLOCKS), "an 'unseen' block was used in training"

    torch.set_num_threads(2)
    net = StudentNet()
    net.load_state_dict(torch.load(a.student)["net"], strict=False)
    net.eval()
    game = RealGame(image_size=128)
    game.hard_reset()

    rng = np.random.default_rng(a.seed)
    courses = [C.random_course(rng, 1.0) for _ in range(a.courses)]
    report = dict(seed=a.seed, student=a.student, seen=SEEN, unseen=UNSEEN, results={}, details={})
    for block in SEEN + UNSEEN:
        rows = [run_student(game, net, course, block) for course in courses]
        report["results"][block] = summarise(rows, 1)
        report["details"][block] = rows
        print(block, report["results"][block], flush=True)
    game.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
