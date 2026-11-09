"""Copy a student dataset, replacing its action labels with a new teacher's choices.

Only frames recorded with teacher features can be relabelled. Used when the teacher
is retrained, so the frames already collected in the real game are not wasted.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402

from parkour.features import OBS_SIZE  # noqa: E402
from parkour.student import AUX_DIM, Store  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--src", default="runs/student/data")
p.add_argument("--dst", default="runs/student2/data")
p.add_argument("--teacher", default="runs/teacher2/teacher_final.zip")
a = p.parse_args()

torch.set_num_threads(2)
src = Store(a.src)
os.makedirs(a.dst, exist_ok=True)
if os.path.exists(os.path.join(a.dst, "meta.json")):
    raise SystemExit("destination already has data")
dst = Store(a.dst)
model = PPO.load(a.teacher, device="cpu")
first, last = src.aux_start, src.n
print("relabelling frames", first, "to", last, flush=True)
aux = np.array(src.aux[first:last])
obs = np.zeros((len(aux), OBS_SIZE), dtype=np.float32)
obs[:, :AUX_DIM] = aux
labels = []
for i in range(0, len(obs), 4096):
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(obs[i:i + 4096]))
        labels.append(np.concatenate([d.probs.numpy() for d in dist.distribution], axis=1))
labels = np.concatenate(labels)
dst.frames[:last - first] = src.frames[first:last]
dst.labels[:last - first] = labels
dst.start[:last - first] = np.asarray(src.start[first:last]) - first
dst.aux[:last - first] = aux
dst.n = last - first
dst.aux_start = 0
dst.flush()
print("wrote", dst.n, "samples; mean forward", labels[:, 1].mean(), "mean jump", labels[:, 3].mean(), flush=True)
