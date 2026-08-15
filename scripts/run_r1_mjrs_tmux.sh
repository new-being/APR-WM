#!/usr/bin/env bash
# Run R1-MJ/RS ordered gates inside tmux-friendly logging.
set -euo pipefail
cd /root/PARSE-WM
mkdir -p runs/r1_mj0 runs/r1_rs0 logs
PY=".venv-robosuite/bin/python"
# Do NOT force MUJOCO_GL=osmesa/egl here: missing native GL libs make the
# public mujoco import raise AttributeError. Physics-only import handles it.
unset MUJOCO_GL || true
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

log "start R1-MJ/RS runner"
if [[ ! -f runs/r1_mj0/closure/summary.json ]]; then
  log "MJ0 missing; running r1-mj0"
  "$PY" -m aprwm_v0 r1-mj0 --output runs/r1_mj0/closure
else
  log "MJ0 summary present; skip unless mj0_passed is false"
fi

mj0_ok="$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path("runs/r1_mj0/closure/summary.json")
print("1" if p.is_file() and json.loads(p.read_text()).get("mj0_passed") else "0")
PY
)"

if [[ "$mj0_ok" != "1" ]]; then
  log "ERROR: MJ0 did not pass; refuse RS0"
  exit 2
fi

log "MJ0 passed; launching RS0 formal 9-cell (duration=10s)"
"$PY" -m aprwm_v0 r1-rs0 \
  --output runs/r1_rs0/c0 \
  --mj0-summary runs/r1_mj0/closure/summary.json \
  --duration 10.0

log "RS0 finished"
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("runs/r1_rs0/c0/summary.json")
d = json.loads(p.read_text())
print(json.dumps({
    "rs0_passed": d.get("rs0_passed"),
    "all_cells_individually_close": d.get("all_cells_individually_close"),
    "cell_count": d.get("cell_count"),
    "packages": d.get("packages"),
}, indent=2, ensure_ascii=False))
PY
log "done"
