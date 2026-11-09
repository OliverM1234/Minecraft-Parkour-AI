"""Train the vision-only student on real game frames with DAgger.

The teacher sees exact state and supplies the correct action for every frame.
Each round mixes teacher and student control, so the student also gets labels
for the awkward states its own mistakes lead to. Runs unattended and survives
game crashes: the dataset and network are written to disk as it goes.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402

from parkour import courses as C  # noqa: E402
from parkour.features import ACTION_DIMS, camera_only_view  # noqa: E402
from parkour.real_env import BLOCKS, RealGame, preprocess_rgb  # noqa: E402
from parkour.student import AUX_DIM, FrameStack, Store, StudentNet, act, distill_loss, mirror_batch, to_batch, weighted_aux_loss  # noqa: E402


def teacher_probs(model, obs):
    t = torch.as_tensor(obs).float()
    with torch.no_grad():
        dist = model.policy.get_distribution(t)
        probs = [d.probs[0].numpy() for d in dist.distribution]
    return probs


def episode(game, model, net, store, rng, beta, course, deterministic_student, max_decisions=350, camera_only=False):
    offset = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25)))
    game.respawn(offset=offset, yaw=float(rng.uniform(-10, 10)))
    stack = FrameStack()
    stack.reset(preprocess_rgb(game.frame()))
    first = store.n if store is not None else 0
    info = {"finished": False, "fell": False, "platform": 0}
    decisions = 0
    for _ in range(max_decisions):
        obs = game.teacher_obs()
        probs = teacher_probs(model, camera_only_view(obs) if camera_only else obs)
        if store is not None:
            store.add(stack.frames[-1], np.concatenate(probs), first, obs[0, :AUX_DIM])
        if beta > 0 and rng.random() < beta:
            action = [int(np.argmax(p)) if rng.random() < 0.7 else int(rng.choice(len(p), p=p / p.sum())) for p in probs]
        else:
            action, _ = act(net, stack.array(), deterministic=deterministic_student)
        done, info = game.step(action)
        decisions += 1
        stack.push(preprocess_rgb(game.frame()))
        if done:
            break
    info["decisions"] = decisions
    return info


def perception_error(net, store, first, last, sample=2000, seed=0):
    """How well the student reads the scene on frames it has not trained on yet."""
    if last - first < 64:
        return {}
    rng = np.random.default_rng(seed)
    chosen = rng.integers(first, last, min(sample, last - first))
    preds, target = [], []
    for i in range(0, len(chosen), 256):
        idx = chosen[i:i + 256]
        with torch.no_grad():
            _, p = net(to_batch(np.stack([store.stack(j) for j in idx])))
        preds.append(p.numpy())
        target.append(np.array(store.aux[idx]))
    e = np.concatenate(preds) - np.concatenate(target)
    scale = dict(next_forward=(6, 5.0), next_left=(7, 5.0), next_height=(8, 2.0), speed_forward=(24, 8.0), speed_left=(25, 8.0))
    out = {k: round(float(np.sqrt(np.mean(e[:, i] ** 2)) * c), 3) for k, (i, c) in scale.items()}
    return out


def train(net, opt, store, rng, steps, batch=128, mirror=False, aux_coef=0.5, weighted=False):
    if store.n < batch:
        return float("nan")
    losses = []
    for _ in range(steps):
        stacks, labels, aux, valid = store.batch(rng, batch)
        if mirror:
            stacks, labels, aux = mirror_batch(stacks, labels, aux, rng)
        logits, pred = net(to_batch(stacks))
        ground = torch.from_numpy(aux[:, 27]) if weighted else None
        loss = distill_loss(logits, torch.from_numpy(labels), ground=ground)
        if valid.any():
            mask = torch.from_numpy(valid)
            if weighted:
                loss = loss + aux_coef * weighted_aux_loss(pred[mask], torch.from_numpy(aux)[mask])
            else:
                loss = loss + aux_coef * torch.nn.functional.mse_loss(pred[mask], torch.from_numpy(aux)[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="runs/teacher/teacher_final.zip")
    p.add_argument("--out", default="runs/student")
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--episodes-per-round", type=int, default=40)
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--image", type=int, default=128)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--mirror", action="store_true", help="also train on left-right mirrored frames")
    p.add_argument("--camera-only-teacher", action="store_true", help="teacher was trained on camera-visible inputs only")
    p.add_argument("--beta-start", type=float, default=None, help="start teacher-control share here and decay 0.85 per round")
    p.add_argument("--warm-start", default=None, help="student checkpoint to start from when none exists in --out")
    p.add_argument("--steps-factor", type=float, default=3.0, help="training steps per round, as passes over the dataset")
    p.add_argument("--beta-decay", type=float, default=0.85)
    p.add_argument("--aux-coef", type=float, default=0.5, help="weight of the scene-reading task")
    p.add_argument("--weighted-loss", action="store_true", help="emphasise nearby geometry, and jumping decisions on the ground")
    p.add_argument("--vertical-mix", type=float, default=0.0, help="share of courses built with extra height changes")
    p.add_argument("--pretrain-steps", type=int, default=0, help="train on existing data before collecting more")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    model = PPO.load(a.teacher, device="cpu")
    net = StudentNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    start_round = 0
    ckpt = os.path.join(a.out, "student_latest.pt")
    if os.path.exists(ckpt):
        saved = torch.load(ckpt)
        missing = net.load_state_dict(saved["net"], strict=False).missing_keys
        if not missing:
            opt.load_state_dict(saved["opt"])
        start_round = saved["round"] + 1
        print("resumed from round", start_round, flush=True)
    elif a.warm_start:
        net.load_state_dict(torch.load(a.warm_start)["net"], strict=False)
        print("warm start from", a.warm_start, flush=True)
    store = Store(os.path.join(a.out, "data"))
    log_path = os.path.join(a.out, "log.jsonl")
    game = None
    t_end = time.time() + a.hours * 3600.0
    r = start_round
    best = -1.0
    if a.pretrain_steps and start_round == 0 and store.n:
        torch.set_num_threads(a.threads)
        t1 = time.time()
        loss = train(net, opt, store, rng, a.pretrain_steps, mirror=a.mirror, aux_coef=a.aux_coef, weighted=a.weighted_loss)
        print("pretrain loss", round(loss, 4), "seconds", round(time.time() - t1), flush=True)
    while time.time() < t_end:
        if a.beta_start is not None:
            beta = a.beta_start * a.beta_decay ** r
        elif r < 4:
            beta = 1.0
        else:
            beta = 0.8 * 0.85 ** (r - 4)
        if beta < 0.05:
            beta = 0.0
        stats = []
        try:
            torch.set_num_threads(1)  # inference only; leave the cores to the game
            if game is None:
                game = RealGame(image_size=a.image)
                game.hard_reset()
            t0 = time.time()
            first_new = store.n
            course, block, used = None, None, 0
            for e in range(a.episodes_per_round):
                if course is None or used >= 4:
                    extra_height = rng.random() < a.vertical_mix
                    course = C.ZIGZAG if rng.random() < 0.15 else C.random_course(rng, 1.0, vertical_p=0.6 if extra_height else 0.25)
                    block = str(rng.choice(BLOCKS))
                    game.build(course, block)
                    used = 0
                used += 1
                stats.append(episode(game, model, net, store, rng, beta, course, deterministic_student=e % 2 == 0, camera_only=a.camera_only_teacher))
            store.flush()
            collect_time = time.time() - t0
        except Exception:
            traceback.print_exc()
            print("game failure; restarting client", flush=True)
            try:
                if game is not None:
                    game.close()
            except Exception:
                pass
            game = None
            os.system("pkill -f MalmoMod >/dev/null 2>&1; pkill -f GradleStart > /dev/null 2>&1")
            time.sleep(5)
            continue
        t1 = time.time()
        torch.set_num_threads(a.threads)  # the game is idle during training
        steps = int(np.clip(a.steps_factor * store.n / 128, 500, a.train_steps))
        seen = perception_error(net, store, first_new, store.n)
        loss = train(net, opt, store, rng, steps, mirror=a.mirror, aux_coef=a.aux_coef, weighted=a.weighted_loss)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "round": r}, ckpt)
        fin = np.mean([s["finished"] for s in stats])
        greedy = np.mean([s["finished"] for s in stats[::2]])
        plat = np.mean([s["platform"] for s in stats])
        row = dict(round=r, beta=round(beta, 3), n=store.n, finished=round(float(fin), 3), greedy_finished=round(float(greedy), 3),
                   platforms=round(float(plat), 2), loss=round(loss, 4), steps=steps, collect_s=round(collect_time, 1),
                   train_s=round(time.time() - t1, 1), wall=round(time.time(), 1), perception=seen)
        if beta == 0.0 and greedy >= best:
            best = float(greedy)
            torch.save({"net": net.state_dict(), "round": r, "greedy_finished": best}, os.path.join(a.out, "student_best.pt"))
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        r += 1
    if game is not None:
        game.close()


if __name__ == "__main__":
    main()
