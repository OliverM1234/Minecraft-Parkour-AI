"""Parkour course layouts, shared by the simulator and the real-game client.

A course is a list of platforms in block coordinates. Each platform is
``(x, y, z, width_x, width_z)``: the blocks ``x .. x+width_x-1`` by
``z .. z+width_z-1`` at height ``y``. Platform 0 is always the spawn.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Platform = Tuple[int, int, int, int, int]
BASE_Y = 4

# The course from the original project (same blocks, same order).
ZIGZAG: List[Platform] = [
    (0, BASE_Y, 0, 1, 1),
    (0, BASE_Y, 2, 1, 1),
    (-1, BASE_Y, 5, 1, 1),
    (-2, BASE_Y, 9, 3, 1),
    (0, BASE_Y, 13, 1, 1),
    (1, BASE_Y, 17, 1, 1),
    (0, BASE_Y, 21, 3, 1),
    (0, BASE_Y, 25, 1, 1),
]


def spawn_point(course: List[Platform]) -> Tuple[float, float, float]:
    x, y, z, wx, wz = course[0]
    return (x + wx / 2.0, float(y + 1), z + wz / 2.0)


def _too_close(a: Platform, b: Platform, min_gap: int = 1) -> bool:
    ax0, ay, az0, awx, awz = a
    bx0, by, bz0, bwx, bwz = b
    gap_x = max(bx0 - (ax0 + awx), ax0 - (bx0 + bwx))
    gap_z = max(bz0 - (az0 + awz), az0 - (bz0 + bwz))
    return max(gap_x, gap_z) < min_gap and abs(ay - by) < 3


def random_course(
    rng: np.random.Generator,
    difficulty: float = 1.0,
    length: int | None = None,
    vertical: bool = True,
    vertical_p: float = 0.25,
) -> List[Platform]:
    """Sample a course whose hops stay inside what the movement model allows.

    ``difficulty`` in [0, 1] widens platforms at 0 and stretches hops at 1.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    n = int(length if length is not None else rng.integers(6, 13))
    wide_p = 0.75 - 0.6 * difficulty
    course: List[Platform] = [(-1, BASE_Y, -1, 3, 3) if rng.random() < 0.5 else (0, BASE_Y, 0, 1, 1)]
    heading = 0.0
    prev_turn = 0.0
    tries = 0
    while len(course) < n and tries < 400:
        tries += 1
        px, py, pz, pwx, pwz = course[-1]
        cx, cz = px + pwx / 2.0, pz + pwz / 2.0
        max_turn = 25.0 + 45.0 * difficulty
        turn = float(rng.uniform(-max_turn, max_turn))
        new_heading = float(np.clip(heading + turn, -80.0, 80.0))
        first = len(course) == 1
        if first:
            max_dist = 2.6 if pwx == 1 else 3.6
        else:
            straight = abs(turn) < 25.0
            max_dist = 3.0 + (1.3 if straight else 0.6) * difficulty
        min_dist = 2.0
        dist = float(rng.uniform(min_dist, max(min_dist + 0.1, max_dist)))
        wx = wz = 1
        if rng.random() < wide_p:
            wx, wz = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        y = py
        if vertical and not first and rng.random() < vertical_p * difficulty:
            step = int(rng.choice([-1, 1]))
            if step == 1 and dist > 3.0:
                dist = float(rng.uniform(2.0, 3.0))
            y = int(np.clip(py + step, BASE_Y - 2, BASE_Y + 4))
        rad = math.radians(new_heading)
        nx = cx - math.sin(rad) * dist - wx / 2.0
        nz = cz + math.cos(rad) * dist - wz / 2.0
        candidate: Platform = (int(round(nx)), y, int(round(nz)), wx, wz)
        if any(_too_close(candidate, p) for p in course):
            continue
        # A jump needs a real gap: reject anything that can be walked across.
        if not _too_close(candidate, course[-1], min_gap=1) and _hop_distance(course[-1], candidate) <= 4.6:
            course.append(candidate)
            heading = new_heading
            prev_turn = turn
    return course


def _hop_distance(a: Platform, b: Platform) -> float:
    ax, _, az, awx, awz = a
    bx, _, bz, bwx, bwz = b
    return math.hypot((bx + bwx / 2.0) - (ax + awx / 2.0), (bz + bwz / 2.0) - (az + awz / 2.0))


def course_commands(course: List[Platform], block: str) -> List[str]:
    """Chat commands that place a course in a running game."""
    out = []
    for x, y, z, wx, wz in course:
        if wx == 1 and wz == 1:
            out.append(f"/setblock {x} {y} {z} {block}")
        else:
            out.append(f"/fill {x} {y} {z} {x + wx - 1} {y} {z + wz - 1} {block}")
    return out
