"""Record demo runs in the real game and compose the presentation video.

    demo.py record  --policy student --which zigzag,0,1,2 --out runs/demo
    demo.py replay  --out runs/demo            (same actions, third-person camera)
    demo.py compose --out runs/demo --video media/demo.mp4
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (15, 18, 24)
PANEL = (24, 29, 38)
TEXT = (232, 236, 241)
MUTED = (150, 160, 175)
ACCENT = (86, 190, 255)
GOOD = (96, 214, 128)
BAD = (240, 104, 104)


def jpeg(frame_rgb):
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def unjpeg(data):
    import cv2
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]


def build_list(which):
    from parkour import courses as C
    from parkour.real_env import BLOCKS
    from scripts_common import HOLDOUT_SEED
    rng = np.random.default_rng(HOLDOUT_SEED)
    pool = [(C.random_course(rng, 1.0), str(rng.choice(BLOCKS))) for _ in range(300)]
    runs = []
    for w in which.split(","):
        if w == "zigzag":
            runs.append(("zigzag", C.ZIGZAG, "quartz_block"))
        else:
            course, block = pool[int(w)]
            runs.append((f"course{int(w):03d}", course, block))
    return runs


def record(a):
    import torch
    from parkour.real_env import RealGame, preprocess_rgb
    from parkour.student import FrameStack, StudentNet, act
    from stable_baselines3 import PPO

    torch.set_num_threads(2)
    net = None
    teacher = None
    if a.policy == "student":
        net = StudentNet()
        net.load_state_dict(torch.load(a.student)["net"], strict=False)
        net.eval()
    else:
        teacher = PPO.load(a.teacher, device="cpu")
    game = RealGame(image_size=a.image)
    game.hard_reset()
    os.makedirs(a.out, exist_ok=True)
    for name, course, block in build_list(a.which):
        game.build(course, block)
        game.respawn()
        frames, track, decisions = [], [], []

        def cap(g):
            pos, yaw, _ = g.state()
            frames.append(jpeg(g.frame()))
            track.append([float(pos[0]), float(pos[1]), float(pos[2]), float(yaw), int(g.cur)])

        cap(game)
        stack = FrameStack()
        stack.reset(preprocess_rgb(game.frame()))
        info = {}
        for _ in range(400):
            if net is not None:
                action, probs = act(net, stack.array(), deterministic=True)
                probs = [p.tolist() for p in probs]
            else:
                action, _ = teacher.predict(game.teacher_obs(), deterministic=True)
                action = [int(x) for x in action[0]]
                probs = [[1.0 if j == action[i] else 0.0 for j in range(d)] for i, d in enumerate((2, 2, 7))]
            decisions.append(dict(action=action, probs=probs, stack=stack.array().copy()))
            done, info = game.step(action, capture=cap)
            stack.push(preprocess_rgb(game.frame()))
            if done:
                break
        result = dict(name=name, course=course, block=block, frames=frames, track=track, decisions=decisions,
                      finished=bool(info.get("finished")), platform=int(info.get("platform", 0)), policy=a.policy)
        pickle.dump(result, open(os.path.join(a.out, f"{name}.pkl"), "wb"))
        print(name, "finished" if result["finished"] else "fell", "platform", result["platform"], "/", len(course) - 1,
              "ticks", len(track), flush=True)
    game.close()


def replay(a):
    os.environ["MINEDOJO_THIRD_PERSON"] = "1"
    from parkour.real_env import RealGame

    game = RealGame(image_size=a.image)
    game.hard_reset()
    for path in sorted(glob.glob(os.path.join(a.out, "*.pkl"))):
        run = pickle.load(open(path, "rb"))
        game.build(run["course"], run["block"])
        game.respawn()
        frames, track = [], []

        def cap(g):
            pos, yaw, _ = g.state()
            frames.append(jpeg(g.frame()))
            track.append([float(pos[0]), float(pos[1]), float(pos[2]), float(yaw), int(g.cur)])

        cap(game)
        for d in run["decisions"]:
            done, info = game.step(d["action"], capture=cap)
            if done:
                break
        a_track = np.array(track)
        b_track = np.array(run["track"])
        k = min(len(a_track), len(b_track))
        error = float(np.abs(a_track[:k, :3] - b_track[:k, :3]).max())
        run["third_person"] = frames
        run["third_person_error"] = error
        pickle.dump(run, open(path, "wb"))
        print(run["name"], "third-person frames", len(frames), "max trace error", round(error, 5), flush=True)
    game.close()


class Canvas:
    def __init__(self, size):
        from PIL import Image, ImageDraw
        self.Image = Image
        self.im = Image.new("RGB", size, BG)
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, size=20, color=TEXT, bold=False):
        from PIL import ImageFont
        self.d.text(xy, s, font=ImageFont.truetype(FONT_B if bold else FONT, size), fill=color)

    def rect(self, box, fill=PANEL, outline=None):
        self.d.rectangle(box, fill=fill, outline=outline)

    def paste(self, arr, xy):
        self.im.paste(self.Image.fromarray(arr), xy)

    def array(self):
        return np.array(self.im)


def minimap(run, tick, box, canvas):
    x0, y0, x1, y1 = box
    canvas.rect(box, PANEL)
    course = run["course"]
    xs = [p[0] for p in course] + [p[0] + p[3] for p in course]
    zs = [p[2] for p in course] + [p[2] + p[4] for p in course]
    span = max(max(xs) - min(xs), max(zs) - min(zs), 6) + 2.0
    scale = min((x1 - x0 - 20) / span, (y1 - y0 - 20) / span)
    cx, cz = (max(xs) + min(xs)) / 2.0, (max(zs) + min(zs)) / 2.0
    midx, midy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def to_px(x, z):
        return midx - (x - cx) * scale, midy + (cz - z) * scale

    current = run["track"][min(tick, len(run["track"]) - 1)][4]
    for i, (x, y, z, wx, wz) in enumerate(course):
        ax, ay = to_px(x, z)
        bx, by = to_px(x + wx, z + wz)
        color = GOOD if i <= current else (110, 122, 140)
        canvas.d.rectangle((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)), fill=color)
    pts = [to_px(t[0], t[2]) for t in run["track"][: tick + 1]]
    if len(pts) > 1:
        canvas.d.line(pts, fill=ACCENT, width=2)
    px, py = pts[-1]
    canvas.d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 220, 120))


def bars(canvas, x, y, width, probs, labels, chosen):
    n = len(probs)
    bw = width / n
    h = 46
    for i, p in enumerate(probs):
        bx = x + i * bw
        canvas.rect((bx + 3, y, bx + bw - 3, y + h), (36, 43, 54))
        filled = h * float(p)
        color = ACCENT if i == chosen else (86, 104, 128)
        canvas.rect((bx + 3, y + h - filled, bx + bw - 3, y + h), color)
        canvas.text((bx + 4, y + h + 3), labels[i], 12, MUTED)


def compose_frame(run, tick, third_person=True):
    import cv2
    H = 720
    total = len(run["track"])
    t = min(tick, total - 1)
    frames_main = run.get("third_person") if third_person and run.get("third_person") else run["frames"]
    main = unjpeg(frames_main[min(t, len(frames_main) - 1)])
    main = cv2.resize(main, (H, H), interpolation=cv2.INTER_AREA)
    pov = unjpeg(run["frames"][t])
    c = Canvas((1280, H))
    c.paste(main, (0, 0))
    x = 736
    w = 1280 - x - 16
    c.text((x, 14), "Parkour agent · plays from pixels only", 22, TEXT, bold=True)
    c.text((x, 44), f"{run['name']}  ·  {run['policy']} policy  ·  real game, 20 ticks/s", 14, MUTED)

    pov_size = 260
    c.text((x, 72), "What the agent sees", 15, MUTED)
    c.paste(cv2.resize(pov, (pov_size, pov_size), interpolation=cv2.INTER_AREA), (x, 94))

    d = run["decisions"][min(t // 2, len(run["decisions"]) - 1)]
    px = x + pov_size + 14
    c.text((px, 72), "CNN input (64 x 64)", 15, MUTED)
    thumb = 122
    stack = d["stack"]
    for i in range(4):
        f = stack[i].transpose(1, 2, 0)
        f = cv2.resize(f, (thumb, thumb), interpolation=cv2.INTER_NEAREST)
        c.paste(f, (px + (i % 2) * (thumb + 6), 94 + (i // 2) * (thumb + 6)))

    y = 368
    c.text((x, y), "Decisions", 15, MUTED)
    fw, jp, turn = d["probs"]
    bars(c, x, y + 22, 96, fw, ["no", "go"], d["action"][0])
    c.text((x + 28, y + 86), "forward", 13, TEXT)
    bars(c, x + 110, y + 22, 96, jp, ["no", "yes"], d["action"][1])
    c.text((x + 146, y + 86), "jump", 13, TEXT)
    bars(c, x + 220, y + 22, w - 220, turn, ["-20", "-8", "-3", "0", "3", "8", "20"], d["action"][2])
    c.text((x + 220 + (w - 220) / 2 - 42, y + 86), "turn (degrees)", 13, TEXT)

    my = 478
    minimap(run, t, (x, my, x + 300, H - 18), c)
    c.text((x + 6, my + 4), "course (top-down)", 12, MUTED)
    cur = run["track"][t][4]
    goal = len(run["course"]) - 1
    sx = x + 316
    c.text((sx, my), "Platform", 15, MUTED)
    c.text((sx, my + 20), f"{cur} / {goal}", 40, TEXT, bold=True)
    c.text((sx, my + 80), "Time", 15, MUTED)
    c.text((sx, my + 100), f"{t / 20.0:.1f} s", 34, TEXT, bold=True)
    if t == total - 1:
        status = ("FINISHED", GOOD) if run["finished"] else ("FELL", BAD)
        c.text((sx, my + 156), status[0], 34, status[1], bold=True)
    return c.array()


def title_card(lines, size=(1280, 720)):
    c = Canvas(size)
    y = 250 - 30 * (len(lines) - 2)
    for i, (text, sz, color, bold) in enumerate(lines):
        from PIL import ImageFont
        font = ImageFont.truetype(FONT_B if bold else FONT, sz)
        w = c.d.textlength(text, font=font)
        c.d.text(((size[0] - w) / 2, y), text, font=font, fill=color)
        y += int(sz * 1.5)
    return c.array()


def compose(a):
    import imageio_ffmpeg
    runs = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(a.out, "*.pkl")))]
    order = [r for r in runs if r["name"] == "zigzag"] + [r for r in runs if r["name"] != "zigzag"]
    os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
    w = imageio_ffmpeg.write_frames(a.video, (1280, 720), fps=20, codec="libx264", quality=None,
                                    bitrate="6M", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                    output_params=["-preset", "medium", "-movflags", "+faststart"])
    w.send(None)
    metrics = json.load(open(a.metrics)) if a.metrics and os.path.exists(a.metrics) else {}

    def hold(frame, seconds):
        for _ in range(int(20 * seconds)):
            w.send(frame.tobytes())

    intro = [
        ("A parkour agent that plays Minecraft from pixels", 40, TEXT, True),
        ("A CNN looks at the game's camera image and chooses the keys to press.", 22, MUTED, False),
        ("No coordinates, no game data: just what a player sees.", 22, MUTED, False),
    ]
    hold(title_card(intro), 4)
    if metrics:
        lines = [("Measured in the real game", 34, TEXT, True)]
        for k in metrics.get("title_lines", []):
            lines.append((k, 24, ACCENT, False))
        lines.append(("Sample runs from the evaluation follow.", 20, MUTED, False))
        hold(title_card(lines), 4)
    before_path = "runs/demo_before/before.pkl"
    if os.path.exists(before_path):
        import cv2
        before = pickle.load(open(before_path, "rb"))
        base = json.load(open("runs/baseline.json")) if os.path.exists("runs/baseline.json") else None
        hold(title_card([("The starting point", 40, TEXT, True),
                         ("The first version of the agent, with its original 64 x 64 view of the game.", 22, MUTED, False)]), 2.5)
        panel_lines = [
            ("First version", 26, TEXT, True),
            ("Trained for about 5,000 steps", 18, MUTED, False),
            ("Most of its view was the game's HUD", 18, MUTED, False),
        ]
        if base:
            panel_lines.append((f"Fell on its second jump: {base['episodes'] - base['finished']} of {base['episodes']} attempts", 18, BAD, False))
        for i in range(3 * len(before["frames"])):
            f = before["frames"][min(i // 3, len(before["frames"]) - 1)]
            c = Canvas((1280, 720))
            c.paste(cv2.resize(f, (720, 720), interpolation=cv2.INTER_NEAREST), (0, 0))
            y = 60
            for text, sz, color, bold in panel_lines:
                c.text((760, y), text, sz, color, bold=bold)
                y += int(sz * 2)
            c.text((760, y + 30), "(what the first CNN saw, enlarged)", 16, MUTED)
            w.send(c.array().tobytes())
        hold(title_card([("Rebuilt: same game, same course", 40, TEXT, True)]), 2)
    for r in order:
        name = "The original course" if r["name"] == "zigzag" else "A course it has never seen"
        hold(title_card([(name, 40, TEXT, True),
                         (f"{len(r['course'])} platforms, one attempt, played live by the CNN.", 22, MUTED, False)]), 2)
        total = len(r["track"])
        for t in range(total):
            w.send(compose_frame(r, t, third_person=a.third_person).tobytes())
        hold(compose_frame(r, total - 1, third_person=a.third_person), 1.5)
    outro = [("Trained on a laptop, without a GPU, in under a day", 36, TEXT, True),
             ("Teacher trained in a fast physics model · student CNN trained on real game frames", 22, MUTED, False)]
    hold(title_card(outro), 4)
    w.close()
    print("wrote", a.video, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["record", "replay", "compose"])
    p.add_argument("--policy", default="student")
    p.add_argument("--student", default="runs/student/student_latest.pt")
    p.add_argument("--teacher", default="runs/teacher/teacher_final.zip")
    p.add_argument("--which", default="zigzag,0,1,2")
    p.add_argument("--image", type=int, default=720)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--video", default="media/demo.mp4")
    p.add_argument("--metrics", default="runs/video_metrics.json")
    p.add_argument("--third-person", action="store_true")
    args = p.parse_args()
    if args.mode == "record":
        record(args)
    elif args.mode == "replay":
        replay(args)
    else:
        compose(args)
