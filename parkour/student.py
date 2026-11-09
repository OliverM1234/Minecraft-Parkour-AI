"""The vision-only student: a small CNN that maps camera frames to actions."""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ACTION_DIMS

IMAGE = 64
STACK = 4
CHANNELS = 3
AUX_DIM = 28  # teacher features recoverable from pixels (platforms, velocity, grounded)


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS * STACK, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(256, d) for d in ACTION_DIMS)
        self.aux = nn.Linear(256, AUX_DIM)

    def forward(self, x):
        z = self.body(self.conv(x))
        return [h(z) for h in self.heads], self.aux(z)


def to_batch(stacks: np.ndarray) -> torch.Tensor:
    """(B, STACK, C, H, W) uint8 -> (B, STACK*C, H, W) float in [0, 1]."""
    b, s, c, h, w = stacks.shape
    return torch.from_numpy(stacks.reshape(b, s * c, h, w).copy()).float() / 255.0


def act(net, stack, deterministic=True):
    with torch.no_grad():
        logits, _ = net(to_batch(stack[None]))
    probs = [F.softmax(l, dim=-1)[0].numpy() for l in logits]
    if deterministic:
        return [int(p.argmax()) for p in probs], probs
    return [int(np.random.choice(len(p), p=p / p.sum())) for p in probs], probs


class FrameStack:
    def __init__(self):
        self.frames = deque(maxlen=STACK)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(STACK):
            self.frames.append(frame)

    def push(self, frame):
        self.frames.append(frame)

    def array(self):
        return np.stack(self.frames)


class Store:
    """Disk-backed dataset of (frame, teacher action probabilities)."""

    def __init__(self, path, capacity=500_000):
        os.makedirs(path, exist_ok=True)
        self.path, self.capacity = path, capacity
        self.meta_path = os.path.join(path, "meta.json")
        resume = os.path.exists(self.meta_path)
        mode = "r+" if resume else "w+"
        aux_path = os.path.join(path, "aux.dat")
        aux_mode = "r+" if os.path.exists(aux_path) else "w+"
        meta = json.load(open(self.meta_path)) if resume else {"n": 0}
        self.frames = np.memmap(os.path.join(path, "frames.dat"), np.uint8, mode, shape=(capacity, CHANNELS, IMAGE, IMAGE))
        self.labels = np.memmap(os.path.join(path, "labels.dat"), np.float32, mode, shape=(capacity, sum(ACTION_DIMS)))
        self.start = np.memmap(os.path.join(path, "start.dat"), np.int64, mode, shape=(capacity,))
        self.aux = np.memmap(aux_path, np.float32, aux_mode, shape=(capacity, AUX_DIM))
        self.n = meta["n"]
        self.aux_start = meta.get("aux_start", self.n if aux_mode == "w+" else 0)

    def add(self, frame, label, episode_start_index, aux):
        if self.n >= self.capacity:
            return
        self.frames[self.n] = frame
        self.labels[self.n] = label
        self.start[self.n] = episode_start_index
        self.aux[self.n] = aux
        self.n += 1

    def flush(self):
        self.frames.flush()
        self.labels.flush()
        self.start.flush()
        self.aux.flush()
        json.dump({"n": self.n, "aux_start": int(self.aux_start)}, open(self.meta_path, "w"))

    def stack(self, i):
        idx = np.maximum(i - np.arange(STACK - 1, -1, -1), self.start[i])
        return self.frames[idx]

    def batch(self, rng, size):
        chosen = rng.integers(0, self.n, size)
        stacks = np.stack([self.stack(i) for i in chosen])
        return stacks, np.array(self.labels[chosen]), np.array(self.aux[chosen]), chosen >= self.aux_start


TURN_START = ACTION_DIMS[0] + ACTION_DIMS[1]  # where the turn probabilities begin in a label
AUX_LEFT = [1, 7, 13, 19, 25]  # aux features that change sign when the view is mirrored


def mirror_batch(stacks, labels, aux, rng, fraction=0.5):
    """Flip part of a batch left to right, with matching labels.

    The game's movement is left-right symmetric, so a mirrored frame paired with
    the mirrored action is another valid example, at no data-collection cost.
    """
    stacks, labels, aux = stacks.copy(), labels.copy(), aux.copy()
    flip = rng.random(len(stacks)) < fraction
    stacks[flip] = stacks[flip][..., ::-1]
    labels[flip, TURN_START:] = labels[flip, TURN_START:][:, ::-1]
    aux[np.ix_(flip, AUX_LEFT)] *= -1.0
    return stacks, labels, aux


def split_labels(labels: torch.Tensor):
    out, k = [], 0
    for d in ACTION_DIMS:
        out.append(labels[:, k: k + d])
        k += d
    return out


def distill_loss(logits, labels, head_weights=(1.0, 2.0, 1.0), ground=None):
    """Cross-entropy between the student and the teacher's action probabilities.

    If ``ground`` (1 = on the ground) is given, the jump head concentrates on the
    frames where jumping actually does something.
    """
    total = 0.0
    for i, (lg, target, w) in enumerate(zip(logits, split_labels(labels), head_weights)):
        per_sample = -(target * F.log_softmax(lg, dim=-1)).sum(dim=-1)
        if i == 1 and ground is not None:
            per_sample = per_sample * (0.2 + 1.3 * ground)
        total = total + w * per_sample.mean()
    return total


def _aux_weights() -> torch.Tensor:
    """Accuracy matters most for the two nearest platforms and for motion."""
    w = torch.full((AUX_DIM,), 0.5)
    w[[0, 1, 2, 6, 7, 8]] = 3.0  # current and next platform: forward, left, height
    w[[12, 13, 14]] = 1.0  # the platform after that
    w[[24, 25, 26]] = 3.0  # velocity
    w[27] = 1.0  # on the ground
    return w / w.mean()


AUX_WEIGHTS = _aux_weights()


def weighted_aux_loss(pred, target):
    return (AUX_WEIGHTS * (pred - target) ** 2).mean()
