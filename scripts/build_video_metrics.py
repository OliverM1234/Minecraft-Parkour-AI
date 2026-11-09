"""Write the measured numbers shown on the demo video's title card."""
import json
import sys

ev = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/evaluation.json"))
base = json.load(open("runs/baseline.json"))
z = ev["student_zigzag"]
c = ev["student_unseen"]
lines = [
    f"Original course: finished {z['finished']} of {z['episodes']} attempts (before this rebuild: {base['finished']} of {base['episodes']})",
    f"Courses it had never seen: finished {c['finished']} of {c['episodes']}",
]
json.dump({"title_lines": lines}, open("runs/video_metrics.json", "w"), indent=1)
print(lines)
