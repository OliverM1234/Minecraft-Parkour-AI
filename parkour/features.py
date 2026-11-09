"""Teacher observations and the action definition, shared by sim and real game."""
from __future__ import annotations

import numpy as np

DECISION_TICKS = 2
YAW_CHOICES = np.array([-20.0, -8.0, -3.0, 0.0, 3.0, 8.0, 20.0])
ACTION_DIMS = (2, 2, len(YAW_CHOICES))  # forward, jump, turn
LOOKAHEAD = 4
OBS_SIZE = 6 * LOOKAHEAD + 3 + 1 + 3 + 2
FALL_MARGIN = 0.6
DEG = np.pi / 180.0


def course_arrays(courses, max_platforms):
    """Turn a list of courses into padded numpy arrays."""
    n = len(courses)
    centers = np.zeros((n, max_platforms, 3))
    halves = np.zeros((n, max_platforms, 2))
    counts = np.zeros(n, int)
    for i, course in enumerate(courses):
        counts[i] = len(course)
        for j, (x, y, z, wx, wz) in enumerate(course):
            centers[i, j] = (x + wx / 2.0, y + 1.0, z + wz / 2.0)
            halves[i, j] = (wx / 2.0, wz / 2.0)
    return centers, halves, counts


def build_obs(pos, disp, yaw, on_ground, prev_action, cur, centers, halves, counts):
    """Teacher observation: where the next platforms are, in the body frame."""
    n = len(pos)
    rad = yaw * DEG
    sin_yaw, cos_yaw = np.sin(rad), np.cos(rad)
    obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
    rows = np.arange(n)
    col = 0
    last = counts - 1
    for k in range(LOOKAHEAD):
        index = np.minimum(cur + k, last)
        exists = (cur + k) <= last
        rel = centers[rows, index] - pos
        forward = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 2]
        left = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 2]
        obs[:, col] = forward / 5.0
        obs[:, col + 1] = left / 5.0
        obs[:, col + 2] = rel[:, 1] / 2.0
        obs[:, col + 3] = halves[rows, index, 0]
        obs[:, col + 4] = halves[rows, index, 1]
        obs[:, col + 5] = exists
        col += 6
    obs[:, col] = (-sin_yaw * disp[:, 0] + cos_yaw * disp[:, 2]) / 0.4
    obs[:, col + 1] = (cos_yaw * disp[:, 0] + sin_yaw * disp[:, 2]) / 0.4
    obs[:, col + 2] = disp[:, 1] / 0.5
    col += 3
    obs[:, col] = on_ground
    col += 1
    obs[:, col] = prev_action[:, 0]
    obs[:, col + 1] = prev_action[:, 1]
    obs[:, col + 2] = YAW_CHOICES[prev_action[:, 2].astype(int)] / 20.0
    col += 3
    obs[:, col] = sin_yaw
    obs[:, col + 1] = cos_yaw
    return obs


def camera_only_view(obs):
    """Blank the teacher inputs a camera cannot reveal (last action, absolute heading)."""
    obs = obs.copy()
    obs[:, 28:] = 0.0
    return obs


def supported_by(pos, centers, halves, counts, player_half=0.3, tol=1e-6):
    """Index of the platform each player stands on, or -1 if airborne."""
    n = len(pos)
    result = -np.ones(n, int)
    dx = np.abs(centers[..., 0] - pos[:, None, 0])
    dz = np.abs(centers[..., 2] - pos[:, None, 2])
    dy = np.abs(centers[..., 1] - pos[:, None, 1])
    valid = np.arange(centers.shape[1])[None, :] < counts[:, None]
    hits = valid & (dy < 1e-3) & (dx < halves[..., 0] + player_half - tol) & (dz < halves[..., 1] + player_half - tol)
    any_hit = hits.any(axis=1)
    # Prefer the furthest platform along the course if several overlap.
    result[any_hit] = (hits.shape[1] - 1 - np.argmax(hits[any_hit, ::-1], axis=1))
    return result
