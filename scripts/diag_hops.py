"""Per-hop failure analysis in the real game, and a sweep over action decoding.

Plays the same courses with each decoding setting, records every decision, and
reports where hops fail: which block, which hop shape, and what went wrong at
take-off. Uses its own seed, separate from the validation and held-out seeds.
"""
import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO  # noqa: E402

from parkour import courses as C  # noqa: E402
from parkour.features import YAW_CHOICES  # noqa: E402
from parkour.real_env import BLOCKS, RealGame, preprocess_rgb  # noqa: E402
from parkour.student import FrameStack, StudentNet, to_batch  # noqa: E402


def teacher_probs(model, obs):
    t = torch.as_tensor(obs).float()
    with torch.no_grad():
        dist = model.policy.get_distribution(t)
        return [d.probs[0].numpy() for d in dist.distribution]


def student_probs(net, stack):
    with torch.no_grad():
        logits, _ = net(to_batch(stack[None]))
    return [torch.softmax(l, dim=-1)[0].numpy() for l in logits]


def decode(probs, jump_threshold=0.5, forward_threshold=0.5, turn_mode="argmax"):
    forward = int(probs[0][1] >= forward_threshold)
    jump = int(probs[1][1] >= jump_threshold)
    if turn_mode == "argmax":
        turn = int(np.argmax(probs[2]))
    else:
        value = float((probs[2] * YAW_CHOICES).sum())
        turn = int(np.argmin(np.abs(YAW_CHOICES - value)))
    return [forward, jump, turn]


def play(game, net, model, course, block, cfg, use_teacher=False, max_decisions=400):
    game.build(course, block)
    game.respawn()
    stack = FrameStack()
    stack.reset(preprocess_rgb(game.frame()))
    rec = []
    info = {"finished": False, "fell": False, "platform": 0}
    for _ in range(max_decisions):
        obs = game.teacher_obs()
        tp = teacher_probs(model, obs)
        sp = student_probs(net, stack.array()) if net is not None else tp
        if use_teacher:
            action = [int(np.argmax(p)) for p in tp]
        else:
            action = decode(sp, *cfg)
        pos, yaw, _ = game.state()
        rec.append(dict(pos=pos.copy(), yaw=yaw, disp=game.disp[0].copy(), ground=bool(obs[0, 27] > 0.5), cur=game.cur,
                        action=action, sp=[p.copy() for p in sp], tp=[p.copy() for p in tp]))
        done, info = game.step(action)
        stack.push(preprocess_rgb(game.frame()))
        if done:
            break
    final, _, _ = game.state()
    return dict(course=course, block=block, rec=rec, finished=bool(info.get("finished")), platform=int(info.get("platform", 0)), final=final.copy())


def hop_rows(ep):
    course, rec = ep["course"], ep["rec"]
    rows = []
    indices = sorted(set(r["cur"] for r in rec))
    for i in indices:
        if i >= len(course) - 1:
            continue
        seg = [k for k, r in enumerate(rec) if r["cur"] == i]
        ground = [k for k in seg if rec[k]["ground"]]
        if not ground:
            continue
        take = ground[-1]
        r = rec[take]
        px, py, pz, pwx, pwz = course[i]
        tx, ty, tz, twx, twz = course[i + 1]
        cx0, cz0 = px + pwx / 2.0, pz + pwz / 2.0
        cx1, cz1 = tx + twx / 2.0, tz + twz / 2.0
        L = math.hypot(cx1 - cx0, cz1 - cz0)
        ux, uz = (cx1 - cx0) / L, (cz1 - cz0) / L
        prev_heading = 0.0
        if i > 0:
            qx, _, qz, qwx, qwz = course[i - 1]
            prev_heading = math.degrees(math.atan2(-(cx0 - (qx + qwx / 2.0)), cz0 - (qz + qwz / 2.0)))
        heading = math.degrees(math.atan2(-ux, uz))
        turn = (heading - prev_heading + 180) % 360 - 180
        # Horizontal gap still to clear at take-off, measured to the nearest point of the target.
        dx_gap = max(tx - r["pos"][0], r["pos"][0] - (tx + twx), 0.0)
        dz_gap = max(tz - r["pos"][2], r["pos"][2] - (tz + twz), 0.0)
        gap = math.hypot(dx_gap, dz_gap)
        bearing = math.degrees(math.atan2(-(cx1 - r["pos"][0]), cz1 - r["pos"][2]))
        yaw_error = (bearing - r["yaw"] + 180) % 360 - 180
        speed = math.hypot(r["disp"][0], r["disp"][2]) * 20.0
        landed = i + 1 in indices or (ep["finished"] and i == len(course) - 2)
        if i + 1 in indices:
            k = next(k for k, q in enumerate(rec) if q["cur"] == i + 1)
            end = rec[k]["pos"]
        else:
            end = ep["final"]
        rel = end[0] - cx1, end[2] - cz1
        along = rel[0] * ux + rel[1] * uz
        side = rel[0] * uz - rel[1] * ux
        jumped = r["action"][1] == 1 or any(rec[k]["action"][1] == 1 for k in ground[-2:])
        rows.append(dict(i=i, block=ep["block"], L=L, dy=ty - py, target_w=max(twx, twz), target_narrow=(twx == 1 and twz == 1), source_w=max(pwx, pwz), turn=turn,
                         gap=gap, yaw_error=yaw_error, speed=speed, landed=landed, along=along, side=side, jumped=jumped,
                         t_jump=float(r["tp"][1][1]), s_jump=float(r["sp"][1][1]), first=i == 0))
    return rows


def agreement(ep):
    rows = defaultdict(list)
    for r in ep["rec"]:
        key = "ground" if r["ground"] else "air"
        for name, s, t in zip(("forward", "jump", "turn"), r["sp"], r["tp"]):
            rows[f"{key}_{name}"].append(int(np.argmax(s) == np.argmax(t)))
    return rows


def bucket(rows, key, edges):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            out[f"{lo:g}..{hi:g}"] = (len(sel), round(float(np.mean([r["landed"] for r in sel])), 3))
    return out


def report(name, episodes):
    rows = [r for ep in episodes for r in hop_rows(ep)]
    finished = [e for e in episodes if e["finished"]]
    print(f"=== {name}: finished {len(finished)}/{len(episodes)}; hops {len(rows)}; per-hop success {np.mean([r['landed'] for r in rows]):.3f}", flush=True)
    print(" by hop length", bucket(rows, "L", [0, 2.3, 2.8, 3.3, 3.8, 4.3, 5]))
    print(" by dy", bucket(rows, "dy", [-2, -0.5, 0.5, 2]))
    print(" by abs turn", bucket([dict(r, turn=abs(r["turn"])) for r in rows], "turn", [0, 15, 30, 45, 60, 90, 180]))
    print(" by required gap at take-off", bucket(rows, "gap", [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6]))
    print(" first hop", bucket([dict(r, first=float(r["first"])) for r in rows], "first", [0, 0.5, 1.5]))
    print(" narrow target", bucket([dict(r, nt=float(r["target_narrow"])) for r in rows], "nt", [0, 0.5, 1.5]))
    per_block = defaultdict(list)
    for r in rows:
        per_block[r["block"]].append(r["landed"])
    print(" by block", {k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(per_block.items())})
    fails = [r for r in rows if not r["landed"]]
    if fails:
        kinds = Counter()
        for r in fails:
            if not r["jumped"]:
                kinds["walked off edge"] += 1
            elif r["along"] < -0.8 and abs(r["side"]) < 1.2:
                kinds["jumped, fell short"] += 1
            elif r["along"] > 0.8 and abs(r["side"]) < 1.2:
                kinds["jumped, went long"] += 1
            else:
                kinds["jumped, missed sideways"] += 1
        print(" failure kinds", dict(kinds))
        print(" failure side offset: mean %.2f, frac negative %.2f" % (np.mean([r["side"] for r in fails]), np.mean([r["side"] < 0 for r in fails])))
        print(" failure yaw error at take-off: mean %.1f, abs mean %.1f" % (np.mean([r["yaw_error"] for r in fails]), np.mean([abs(r["yaw_error"]) for r in fails])))
        print(" failure teacher P(jump) at take-off: %.2f; student P(jump): %.2f" % (np.mean([r["t_jump"] for r in fails]), np.mean([r["s_jump"] for r in fails])))
    ok = [r for r in rows if r["landed"]]
    if ok:
        print(" landed: side offset mean %.2f sd %.2f; along mean %.2f sd %.2f; yaw error mean %.1f abs %.1f; gap at take-off mean %.2f; speed %.2f" % (
            np.mean([r["side"] for r in ok]), np.std([r["side"] for r in ok]), np.mean([r["along"] for r in ok]), np.std([r["along"] for r in ok]),
            np.mean([r["yaw_error"] for r in ok]), np.mean([abs(r["yaw_error"]) for r in ok]), np.mean([r["gap"] for r in ok]), np.mean([r["speed"] for r in ok])))
    agg = defaultdict(list)
    for ep in episodes:
        for k, v in agreement(ep).items():
            agg[k].extend(v)
    print(" student argmax equals teacher argmax", {k: round(float(np.mean(v)), 3) for k, v in sorted(agg.items())}, flush=True)
    return dict(name=name, finished=len(finished), episodes=len(episodes), per_hop=float(np.mean([r["landed"] for r in rows])), mean_platforms=float(np.mean([e["platform"] for e in episodes])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="runs/student/student_latest.pt")
    p.add_argument("--teacher", default="runs/teacher/teacher_final.zip")
    p.add_argument("--courses", type=int, default=50)
    p.add_argument("--seed", type=int, default=20250)
    p.add_argument("--sweep", default="0.5,0.5,argmax", help="semicolon-separated jump_threshold,forward_threshold,turn_mode")
    p.add_argument("--with-teacher", action="store_true")
    p.add_argument("--tag", default="diag")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    net = StudentNet()
    net.load_state_dict(torch.load(a.student)["net"], strict=False)
    net.eval()
    model = PPO.load(a.teacher, device="cpu")
    game = RealGame(image_size=128)
    game.hard_reset()
    rng = np.random.default_rng(a.seed)
    courses = [(C.random_course(rng, 1.0), str(rng.choice(BLOCKS))) for _ in range(a.courses)]
    os.makedirs("runs/diag", exist_ok=True)
    summary = []
    if a.with_teacher:
        eps = [play(game, net, model, c, b, None, use_teacher=True) for c, b in courses]
        summary.append(report("teacher (greedy, exact state)", eps))
    for spec in a.sweep.split(";"):
        jt, ft, mode = spec.split(",")
        cfg = (float(jt), float(ft), mode)
        eps = [play(game, net, model, c, b, cfg) for c, b in courses]
        summary.append(report(f"student jump>={jt} forward>={ft} turn={mode}", eps))
        pickle.dump(eps, open(f"runs/diag/{a.tag}_{jt}_{ft}_{mode}.pkl", "wb"))
    game.close()
    json.dump(summary, open(f"runs/diag/{a.tag}_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
