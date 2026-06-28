#!/usr/bin/env bash
# Reproducible AdamW-vs-Gefen optimizer benchmark launcher.
#
# Runs (optional per-optimizer LR sweep ->) 2000-step finals for every
# optimizer x model -> plots -> aggregated table. Everything is passed via
# CLI flags; there are NO hardcoded paths. Each (model, optimizer) job is
# pinned to one GPU from --gpus, round-robin, so within a model all optimizers
# share a GPU type for clean tok/s comparison.
#
# See README.md for the full regime and a concrete example.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"

PY=""
OUT=""
GPUS=""
STEPS=2000
SEQ=2048
ARCH="8.6"
SEED=0
DATASET="tatsu-lab/alpaca"
EVAL_EVERY=50
LR_SWEEP=0
SWEEP_STEPS=175
MUON_ADJUST="match_rms_adamw"
OPTS_DEFAULT="adamw_bf16 adamw8bit adamw4bit gefen_fused gefen_muon"
OPTS="$OPTS_DEFAULT"
NO_MUON=0

# tag -> model path, and the tag order, collected from CLI
declare -a TAGS=()
declare -a PATHS=()

# fair default LRs (each optimizer's own optimum); overridden by --lr-sweep
fair_lr() {
  case "$1" in
    adamw_bf16|adamw8bit|adamw4bit) echo "5e-5" ;;
    gefen_fused|gefen_nonfused)     echo "2e-5" ;;
    gefen_muon)                     echo "5e-5" ;;
    *) echo "5e-5" ;;
  esac
}

# LR grid swept per optimizer when --lr-sweep is given
sweep_grid() {
  case "$1" in
    adamw_bf16|adamw8bit|adamw4bit) echo "1e-5 2e-5 5e-5 1e-4" ;;
    gefen_fused|gefen_nonfused)     echo "5e-6 1e-5 2e-5 5e-5" ;;
    gefen_muon)                     echo "1e-5 2e-5 5e-5 1e-4" ;;
    *) echo "1e-5 2e-5 5e-5" ;;
  esac
}

usage() {
  cat <<EOF
Usage: run.sh --venv <python> --out <dir> --gpus "2,5,7" [options] \\
              ( --model <tag>=<path> [--model ...] | --model-0p6b <path> --model-1p7b <path> )

Required:
  --venv <python>        Python interpreter of the env where the Gefen fork is
                         installed (pip install -e . on THIS fork) plus
                         bitsandbytes, torchao, matplotlib, datasets, transformers.
  --out <dir>            Output dir for results.jsonl, logs/, table, charts.
  --gpus "a,b,c"         Comma-separated CUDA device indices to use (round-robin).
  At least one model:
    --model <tag>=<path>   Repeatable. tag groups results (e.g. qwen1p7b=/models/Qwen3-1.7B).
    --model-0p6b <path>    Convenience for --model qwen0p6b=<path>.
    --model-1p7b <path>    Convenience for --model qwen1p7b=<path>.

Options:
  --steps N              Final-run steps (default $STEPS).
  --seq N                Sequence / pack-block length (default $SEQ).
  --arch X.Y             TORCH_CUDA_ARCH_LIST, e.g. 8.6 Ampere, 12.0 Blackwell (default $ARCH).
  --seed N               Random seed / data order (default $SEED).
  --dataset NAME         HF Alpaca-format dataset id (default $DATASET).
  --opts "a b c"         Optimizers to run (default: "$OPTS_DEFAULT").
                         Available: adamw_bf16 adamw8bit adamw4bit gefen_fused gefen_nonfused gefen_muon
  --no-muon              Drop gefen_muon from the run (it is in the default set but is
                         the slowest optimizer; use this to skip the Newton-Schulz cost).
  --muon-adjust MODE     gefen_muon adjust_lr_fn (default $MUON_ADJUST).
  --eval-every N         Intermediate eval-logging cadence (default $EVAL_EVERY).
  --lr-sweep             Run a short per-optimizer LR sweep first and use each
                         optimizer's best LR for the finals (default: documented fair LRs).
  --sweep-steps N        Steps per LR-sweep cell (default $SWEEP_STEPS).
  -h, --help             This help.

Example:
  ./run.sh --venv /path/to/venv/bin/python --out ./out \\
           --model-0p6b /models/Qwen3-0.6B --model-1p7b /models/Qwen3-1.7B \\
           --gpus "2,5,7" --steps 2000 --arch 8.6
EOF
}

[ $# -eq 0 ] && { usage; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)        PY="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --seq)         SEQ="$2"; shift 2 ;;
    --arch)        ARCH="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --dataset)     DATASET="$2"; shift 2 ;;
    --opts)        OPTS="$2"; shift 2 ;;
    --no-muon)     NO_MUON=1; shift ;;
    --muon-adjust) MUON_ADJUST="$2"; shift 2 ;;
    --eval-every)  EVAL_EVERY="$2"; shift 2 ;;
    --lr-sweep)    LR_SWEEP=1; shift ;;
    --sweep-steps) SWEEP_STEPS="$2"; shift 2 ;;
    --model)       TAGS+=("${2%%=*}"); PATHS+=("${2#*=}"); shift 2 ;;
    --model-0p6b)  TAGS+=("qwen0p6b"); PATHS+=("$2"); shift 2 ;;
    --model-1p7b)  TAGS+=("qwen1p7b"); PATHS+=("$2"); shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage; exit 1 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
[ -n "$PY" ]   || die "--venv <python> is required"
[ -x "$PY" ] || [ -f "$PY" ] || die "--venv python not found/executable: $PY"
[ -n "$OUT" ]  || die "--out <dir> is required"
[ -n "$GPUS" ] || die "--gpus \"a,b,c\" is required"
[ "${#TAGS[@]}" -gt 0 ] || die "at least one --model / --model-0p6b / --model-1p7b is required"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[ "${#GPU_ARR[@]}" -gt 0 ] || die "no GPUs parsed from --gpus"

# --no-muon strips gefen_muon whether it came from the default set or --opts
if [ "$NO_MUON" -eq 1 ]; then
  OPTS="$(echo "$OPTS" | tr ' ' '\n' | grep -vx 'gefen_muon' | tr '\n' ' ')"
fi

mkdir -p "$OUT" "$OUT/logs"
RESULTS="$OUT/results.jsonl"
SWEEP="$OUT/results_lrsweep.jsonl"
LOGS="$OUT/logs"
CELL="$HERE/sweep_cell.py"

# --- GPU pool: dispatch jobs round-robin, one per GPU, wait when full ---
declare -A GPU_BUSY=()
next_gpu() { # echoes a free gpu id, blocking until one frees
  while true; do
    for g in "${GPU_ARR[@]}"; do
      if [ -z "${GPU_BUSY[$g]:-}" ]; then echo "$g"; return; fi
    done
    wait -n 2>/dev/null || true
    for g in "${!GPU_BUSY[@]}"; do
      kill -0 "${GPU_BUSY[$g]}" 2>/dev/null || unset "GPU_BUSY[$g]"
    done
  done
}

run_cell() { # gpu tag model opt lr steps out logfile
  local gpu="$1" tag="$2" model="$3" opt="$4" lr="$5" steps="$6" out="$7" log="$8"
  echo "[start $(date +%T)] $tag $opt lr=$lr gpu=$gpu steps=$steps -> $log"
  "$PY" "$CELL" --opt "$opt" --model "$model" --lr "$lr" --seed "$SEED" \
    --steps "$steps" --seq "$SEQ" --gpu "$gpu" --arch "$ARCH" --tag "$tag" \
    --dataset "$DATASET" --eval-every "$EVAL_EVERY" --muon-adjust "$MUON_ADJUST" \
    --out "$out" > "$log" 2>&1
  echo "[done  $(date +%T)] $tag $opt rc=$?"
}

dispatch() { # tag model opt lr steps out logfile  -> backgrounds on a free gpu
  local g; g="$(next_gpu)"
  GPU_BUSY[$g]=PENDING
  run_cell "$g" "$@" &
  GPU_BUSY[$g]=$!
}

# ---- optional LR sweep ----
declare -A BEST_LR=()
if [ "$LR_SWEEP" -eq 1 ]; then
  echo "=== LR SWEEP (${SWEEP_STEPS} steps/cell) ==="
  : > "$SWEEP"
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"; model="${PATHS[$i]}"
    for opt in $OPTS; do
      for lr in $(sweep_grid "$opt"); do
        dispatch "$tag" "$model" "$opt" "$lr" "$SWEEP_STEPS" "$SWEEP" \
          "$LOGS/sweep_${tag}_${opt}_${lr}.log"
      done
    done
  done
  wait
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    for opt in $OPTS; do
      BEST_LR["${tag}_${opt}"]="$("$PY" "$HERE/pick_lr.py" --results "$SWEEP" --tag "$tag" --opt "$opt")"
      echo "  best LR  $tag $opt -> ${BEST_LR[${tag}_${opt}]}"
    done
  done
fi

# ---- 2000-step finals ----
echo "=== FINALS (${STEPS} steps) ==="
: > "$RESULTS"
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"; model="${PATHS[$i]}"
  for opt in $OPTS; do
    if [ "$LR_SWEEP" -eq 1 ]; then lr="${BEST_LR[${tag}_${opt}]}"; else lr="$(fair_lr "$opt")"; fi
    dispatch "$tag" "$model" "$opt" "$lr" "$STEPS" "$RESULTS" "$LOGS/${tag}_${opt}.log"
  done
done
wait
echo "=== ALL FINALS DONE $(date +%T) ==="

# ---- table + charts ----
"$PY" "$HERE/build_table.py" --results "$RESULTS" --out-dir "$OUT" \
  --label qwen0p6b=Qwen3-0.6B --label qwen1p7b=Qwen3-1.7B
"$PY" "$HERE/plot_quads.py"  --csv "$OUT/comparison_table.csv" --out-dir "$OUT"
"$PY" "$HERE/plot_curves.py" --csv "$OUT/comparison_table.csv" --logs "$LOGS" --out-dir "$OUT"

echo
echo "Outputs in $OUT :"
echo "  comparison_table.csv / comparison_table.md"
echo "  quad_<model>.png, loss_curve_<model>.png, tput_vram_<model>.png"
echo "DONE $(date +%T)"
