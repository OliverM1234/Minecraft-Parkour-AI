"""Side-by-side video of what each network gets to see, from the same recorded run.

Left: the game camera, as a person sees it. Middle: the first version's input,
re-created from the same frames (grayscale, 64 x 64, one new frame every 4 ticks).
The shrink averages pixels, which matches how the game drew its 64 x 64 frames:
Minecraft blurs textures when drawing small (compare media/old_cnn_view.png). The HUD is hidden in both,
so the comparison is only about colour, frame rate and frame history. Right: the
rebuilt student's real input, as stored at each decision (4 colour frames,
64 x 64, a new decision every 2 ticks).

Usage: view_compare.py [--runs runs/demo_v2] [--which zigzag,course000] [--out media/view_compare.mp4]
"""
import argparse
import os
import pickle

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (14, 21, 25)
TEXT = (242, 245, 246)
MUTED = (184, 196, 203)
ACCENT = (79, 179, 246)
OLD = (235, 104, 52)
W, H = 1280, 720
PANEL = 384
TOP = 128
XS = [32, 448, 864]
OLD_TICKS, NEW_TICKS = 4, 2


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def unjpeg(data):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def old_view(frame_rgb):
    """The first version's input: a 64 x 64 grayscale frame."""
    small = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[:, :, None], 3, axis=2)


def big(img64, size=PANEL):
    return cv2.resize(img64, (size, size), interpolation=cv2.INTER_NEAREST)


def frame_at(run, tick, label):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((32, 28), label, font=font(26, True), fill=TEXT)
    d.text((W - 32, 34), f"{tick / 20:.1f} s", font=font(20), fill=MUTED, anchor="ra")

    full = unjpeg(run["frames"][tick])
    old_tick = (tick // OLD_TICKS) * OLD_TICKS
    old = old_view(unjpeg(run["frames"][old_tick]))
    stack = run["decisions"][min(tick // NEW_TICKS, len(run["decisions"]) - 1)]["stack"]
    new = [f.transpose(1, 2, 0) for f in stack]

    titles = [("The game, as a person sees it", MUTED), ("First version's input", OLD), ("Rebuilt agent's input", ACCENT)]
    for x, (t, c) in zip(XS, titles):
        d.text((x, TOP - 40), t, font=font(22, True), fill=c)
    canvas.paste(Image.fromarray(cv2.resize(full, (PANEL, PANEL), interpolation=cv2.INTER_AREA)), (XS[0], TOP))
    canvas.paste(Image.fromarray(big(old)), (XS[1], TOP))
    canvas.paste(Image.fromarray(big(new[-1])), (XS[2], TOP))

    y = TOP + PANEL + 20
    d.text((XS[0], y), "Full size, 20 frames a second", font=font(19), fill=MUTED)
    d.text((XS[1], y), "Grayscale · 5 new frames a second", font=font(19), fill=MUTED)
    d.text((XS[2], y), "Colour · 10 new frames a second", font=font(19), fill=MUTED)

    y += 44
    thumb = 84
    d.text((XS[1], y), "1 frame per decision", font=font(19, True), fill=TEXT)
    d.text((XS[2], y), "4 frames per decision", font=font(19, True), fill=TEXT)
    y += 34
    canvas.paste(Image.fromarray(big(old, thumb)), (XS[1], y))
    for i, f in enumerate(new):
        canvas.paste(Image.fromarray(big(f, thumb)), (XS[2] + i * (thumb + 16), y))
    d.text((XS[0], y), "Both inputs are 64 × 64, with the\nstatus bars hidden in both, so only\nthe extra data differs.", font=font(19), fill=MUTED, spacing=8)
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/demo_v2")
    p.add_argument("--which", default="zigzag,course000")
    p.add_argument("--out", default="media/view_compare.mp4")
    a = p.parse_args()
    labels = {"zigzag": "The original course"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    w = imageio_ffmpeg.write_frames(a.out, (W, H), fps=20, codec="libx264", quality=None,
                                    output_params=["-crf", "20", "-movflags", "+faststart"])
    w.send(None)
    for name in a.which.split(","):
        run = pickle.load(open(os.path.join(a.runs, f"{name}.pkl"), "rb"))
        label = labels.get(name, "A course it never trained on")
        n = len(run["frames"])
        for t in range(n):
            w.send(frame_at(run, t, label).tobytes())
        last = frame_at(run, n - 1, label).tobytes()
        for _ in range(20):
            w.send(last)
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
