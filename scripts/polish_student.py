"""Final supervised polish of the student on the full aggregated dataset,
with a decaying learning rate. Writes student_polished.pt."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from parkour.student import Store, StudentNet, distill_loss, mirror_batch, to_batch, weighted_aux_loss

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/student")
p.add_argument("--steps", type=int, default=15000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--threads", type=int, default=6)
p.add_argument("--mirror", action="store_true")
p.add_argument("--weighted-loss", action="store_true")
p.add_argument("--aux-coef", type=float, default=0.5)
a = p.parse_args()

torch.set_num_threads(a.threads)
store = Store(os.path.join(a.dir, "data"))
print("samples", store.n, flush=True)
net = StudentNet()
saved = torch.load(os.path.join(a.dir, "student_latest.pt"))
net.load_state_dict(saved["net"], strict=False)
opt = torch.optim.Adam(net.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=2e-5)
rng = np.random.default_rng(1)
t = time.time()
losses = []
for i in range(a.steps):
    stacks, labels, aux, valid = store.batch(rng, a.batch)
    if a.mirror:
        stacks, labels, aux = mirror_batch(stacks, labels, aux, rng)
    logits, pred = net(to_batch(stacks))
    ground = torch.from_numpy(aux[:, 27]) if a.weighted_loss else None
    loss = distill_loss(logits, torch.from_numpy(labels), ground=ground)
    if valid.any():
        m = torch.from_numpy(valid)
        if a.weighted_loss:
            loss = loss + a.aux_coef * weighted_aux_loss(pred[m], torch.from_numpy(aux)[m])
        else:
            loss = loss + a.aux_coef * torch.nn.functional.mse_loss(pred[m], torch.from_numpy(aux)[m])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    sched.step()
    losses.append(loss.item())
    if (i + 1) % 1000 == 0:
        print(i + 1, round(float(np.mean(losses[-1000:])), 4), round(time.time() - t), flush=True)
        torch.save({"net": net.state_dict(), "steps": i + 1}, os.path.join(a.dir, "student_polished.pt"))
print("done", flush=True)
