#!/bin/bash
# Pick a checkpoint on validation courses, run the held-out test once, build the video.
# Usage: scripts/final_stage.sh <student_dir> <tag> [teacher.zip] [--camera-only-teacher]
cd "$(dirname "$0")/.."
PY=/home/oliver-m/minedojo_env/bin/python
export MINEDOJO_HEADLESS=1 MINEDOJO_HIDE_GUI=1
DIR=${1:-runs/student}
TAG=${2:-v1}
TEACHER=${3:-runs/teacher/teacher_final.zip}
EXTRA=${4:-}
LOG=runs/logs/overnight.log
mkdir -p runs/logs
stamp() { echo "[$(date)] [$TAG] $1" >> "$LOG"; }
for name in latest polished; do
  [ -f "$DIR/student_$name.pt" ] || continue
  $PY -u scripts/evaluate.py --student "$DIR/student_$name.pt" --skip-teacher --courses 60 --zigzag 20 --seed 555111 --out runs/validation_${TAG}_$name.json >> runs/logs/validate_$TAG.log 2>&1
done
CKPT=$($PY - "$TAG" "$DIR" <<'PYEOF'
import json, sys
tag, d = sys.argv[1:3]
scores = {}
for name in ("latest", "polished"):
    try:
        r = json.load(open(f"runs/validation_{tag}_{name}.json"))
        scores[name] = r["student_unseen"]["success_rate"] + 0.2 * r["student_zigzag"]["success_rate"]
    except Exception:
        scores[name] = -1
print(scores, file=sys.stderr)
best = max(scores, key=scores.get)
print(f"{d}/student_{best}.pt")
PYEOF
)
echo "$CKPT" > runs/chosen_student_$TAG.txt
stamp "chosen checkpoint: $CKPT; running held-out test"
$PY -u scripts/evaluate.py --student "$CKPT" --teacher "$TEACHER" $EXTRA --courses 100 --zigzag 50 --out runs/evaluation_$TAG.json >> runs/logs/evaluate_$TAG.log 2>&1
stamp "held-out test done; building demo"
$PY scripts/build_video_metrics.py runs/evaluation_$TAG.json >> runs/logs/evaluate_$TAG.log 2>&1
rm -rf runs/demo_$TAG
$PY -u scripts/demo.py record --policy student --student "$CKPT" --which zigzag,0,1,2,3,4 --image 720 --out runs/demo_$TAG >> runs/logs/demo_$TAG.log 2>&1
$PY -u scripts/demo.py replay --image 720 --out runs/demo_$TAG >> runs/logs/demo_$TAG.log 2>&1
$PY -u scripts/demo.py compose --out runs/demo_$TAG --video media/demo_$TAG.mp4 --third-person >> runs/logs/demo_$TAG.log 2>&1
stamp "demo video done"
