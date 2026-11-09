"""Generate the result charts used in the README (media/fig_*.png) and, with --deck,
the slide versions without titles and with larger text (media/deck_fig_*.png)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})
os.makedirs("media", exist_ok=True)
DECK = "--deck" in sys.argv
if DECK:
    plt.rcParams.update({"font.size": 17, "axes.titlesize": 17})


def title(ax, main, sub=None):
    if DECK:
        return
    ax.set_title(main, loc="left", fontsize=17, fontweight="bold", color=INK, pad=26 if sub else 12)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=12.5)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"media/{'deck_' if DECK else ''}{name}", dpi=180)
    plt.close(fig)
    print("wrote", name)


def physics():
    path = "runs/physics/physics_real.json"
    if not os.path.exists(path):
        return
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "scripts")
    from validate_sim import replay
    exp = json.load(open(path))["sprint_jump60"]
    real, sim = replay(exp)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(sim[:, 2], sim[:, 1], color=BLUE, linewidth=2, label="Simulator", zorder=2)
    ax.scatter(real[:, 2], real[:, 1], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1.6, label="Real game", zorder=3)
    ax.set_xlabel("Distance travelled (blocks)")
    ax.set_ylabel("Height (blocks)")
    ax.set_ylim(4.85, 6.75)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    v = json.load(open("runs/physics/validation.json"))
    worst = max(x["max_error"] for x in v.values())
    ticks = sum(x["ticks"] for x in v.values())
    title(ax, "The simulator moves exactly like the real game",
          f"Largest position difference across {len(v)} recorded tests ({ticks} ticks): {worst:.4f} blocks")
    save(fig, "fig_sim_vs_real.png")


def teacher():
    series = [("runs/teacher/history.json", "First teacher: exact state", AXIS, 2.0),
              ("runs/teacher2/history.json", "Final teacher: camera-visible inputs, delayed and noisy", BLUE, 2.4)]
    series = [s for s in series if os.path.exists(s[0])]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for path, name, color, width in series:
        h = json.load(open(path))
        steps = np.array([r["steps"] for r in h]) / 1e6
        success = np.array([r["success"] for r in h]) * 100
        ax.plot(steps, success, color=color, linewidth=width, label=name)
        ax.annotate(f"{success[-1]:.0f}%", (steps[-1], success[-1]), textcoords="offset points", xytext=(6, -4), fontsize=plt.rcParams["font.size"], color=INK)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 56)
    ax.set_xlabel("Training steps (millions, in the simulator)")
    ax.set_ylabel("Courses finished (%)")
    ax.legend(frameon=False, loc="lower right")
    title(ax, "Both teachers learn the courses in under an hour on a laptop", "Random courses finished while training in the simulator, with noise switched on")
    save(fig, "fig_teacher.png")


def student():
    if not (os.path.exists("runs/evaluation_v1.json") and os.path.exists("runs/evaluation_v2.json")):
        return
    first = json.load(open("runs/evaluation_v1.json"))
    final = json.load(open("runs/evaluation_v2.json"))
    rows = [("First student", first["student_unseen"], AXIS),
            ("Final student", final["student_unseen"], BLUE),
            ("Final teacher (exact positions)", final["teacher_unseen"], AXIS)]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    values = [r[1]["finished"] for r in rows]
    ax.barh([r[0] for r in rows], values, color=[r[2] for r in rows], height=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Courses finished, out of 100")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 1.5, i, f"{v}", va="center", color=INK, fontsize=plt.rcParams["font.size"])
    title(ax, "From 40 to 65 of 100 unseen courses", "Same courses, one attempt each, in the real game")
    save(fig, "fig_student.png")


def before_after():
    if not (os.path.exists("runs/baseline.json") and os.path.exists("runs/evaluation.json")):
        return
    base = json.load(open("runs/baseline.json"))
    ev = json.load(open("runs/evaluation.json"))
    new = ev["student_zigzag"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    labels = ["Before: first version", "After: rebuilt agent"]
    values = [base["platforms_mean"], new["mean_platforms"]]
    colors = [AXIS, BLUE]
    ax.barh(labels, values, color=colors, height=0.5)
    ax.set_xlim(0, 7.8)
    ax.set_xlabel("Platforms reached on the original course (out of 7, average)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.text(values[0] + 0.1, 0, f"{values[0]:.1f}   ({base['finished']} of {base['episodes']} attempts finished)", va="center", color=INK, fontsize=13)
    ax.text(values[1] - 0.1, 1, f"{values[1]:.1f}   ({new['finished']} of {new['episodes']} finished)", va="center", ha="right", color="white", fontsize=13)
    title(ax, "Same course, same game: before and after")
    save(fig, "fig_before_after.png")


if __name__ == "__main__":
    physics()
    teacher()
    student()
    before_after()
