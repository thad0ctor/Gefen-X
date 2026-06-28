#!/usr/bin/env bash
# Reproducible AdamW-vs-Gefen optimizer benchmark launcher.
#
# Runs (optional per-optimizer LR sweep ->) finals for every optimizer x model
# -> plots -> aggregated table. Settings come from CLI flags and/or a YAML
# config; there are NO hardcoded paths.
#
#   precedence:  CLI flag  >  config.yaml  >  built-in default
#
# A config is loaded from --config <path>, else ./config.yaml if it exists
# (see config.example.yaml). Each (model, optimizer) job is pinned to one GPU
# from the GPU list, round-robin, so within a model all optimizers share a GPU
# type for clean tok/s comparison. GPU indices are PCI-bus-ordered
# (CUDA_DEVICE_ORDER=PCI_BUS_ID); list them with --list-gpus.
#
# See README.md for the full regime and a concrete example.
set -u

# PCI-bus order so the indices in --gpus / config match `--list-gpus` and the
# physical GPUs the cells pin (children inherit this).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- built-in defaults (overridden by config.yaml, then by CLI) ----
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
CONFIG=""
DRY=0

# tag -> model path (parallel arrays) and tag -> display label
declare -a TAGS=()
declare -a PATHS=()
declare -A LABELS=([qwen0p6b]=Qwen3-0.6B [qwen1p7b]=Qwen3-1.7B)
# keys explicitly set on the CLI (so config.yaml does not override them)
declare -A CLI_SET=()

die() { echo "ERROR: $*" >&2; exit 1; }

list_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
  echo "PCI-ordered GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) — use the index column in --gpus / gpus:"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total \
    --format=csv,noheader | awk -F', ' '{printf "  %s  %-28s  %s  used %s / %s\n",$1,$2,$3,$4,$5}'
}

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
Usage: run.sh [--config config.yaml] [options]
       run.sh --venv <python> --out <dir> --gpus "1,2,4" \\
              ( --model <tag>=<path> [--model ...] | --model-0p6b <path> --model-1p7b <path> )

Settings come from CLI flags and/or a YAML config (see config.example.yaml).
Precedence: CLI flag > config.yaml > built-in default. ./config.yaml is loaded
automatically when present; --config <path> points elsewhere. With a complete
config, 'run.sh' needs no flags at all.

Required (here or in the config):
  --venv <python>        Python of the env with the Gefen fork (pip install -e .
                         on THIS fork) + bitsandbytes, torchao, matplotlib,
                         datasets, transformers.
  --out <dir>            Output dir for results.jsonl, logs/, table, charts.
  --gpus "a,b,c"         PCI-ordered CUDA device indices to use (round-robin).
  At least one model:
    --model <tag>=<path>   Repeatable. tag groups results (e.g. qwen1p7b=/models/Qwen3-1.7B).
    --model-0p6b <path>    Convenience for --model qwen0p6b=<path>.
    --model-1p7b <path>    Convenience for --model qwen1p7b=<path>.

Options:
  --config <path>        YAML config file (default: ./config.yaml if it exists).
  --list-gpus            Print the PCI-ordered GPU list and exit.
  --dry-run, --check     Resolve config + CLI, print the effective settings, and exit
                         without launching any runs.
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

Example (all from config):
  cp config.example.yaml config.yaml && \$EDITOR config.yaml
  ./run.sh

Example (pure CLI):
  ./run.sh --venv /path/to/venv/bin/python --out ./out \\
           --model-0p6b /models/Qwen3-0.6B --model-1p7b /models/Qwen3-1.7B \\
           --gpus "2,5,7" --steps 2000 --arch 8.6
EOF
}

# Nothing to do with no args and no config to fall back on.
[ $# -eq 0 ] && [ ! -f "$HERE/config.yaml" ] && { usage; exit 1; }

# ---- parse CLI (records which keys were explicitly set) ----
while [ $# -gt 0 ]; do
  case "$1" in
    --config)      CONFIG="$2"; shift 2 ;;
    --list-gpus)   list_gpus; exit 0 ;;
    --dry-run|--check) DRY=1; shift ;;
    --venv)        PY="$2"; CLI_SET[venv]=1; shift 2 ;;
    --out)         OUT="$2"; CLI_SET[out]=1; shift 2 ;;
    --gpus)        GPUS="$2"; CLI_SET[gpus]=1; shift 2 ;;
    --steps)       STEPS="$2"; CLI_SET[steps]=1; shift 2 ;;
    --seq)         SEQ="$2"; CLI_SET[seq]=1; shift 2 ;;
    --arch)        ARCH="$2"; CLI_SET[arch]=1; shift 2 ;;
    --seed)        SEED="$2"; CLI_SET[seed]=1; shift 2 ;;
    --dataset)     DATASET="$2"; CLI_SET[dataset]=1; shift 2 ;;
    --opts)        OPTS="$2"; CLI_SET[opts]=1; shift 2 ;;
    --no-muon)     NO_MUON=1; CLI_SET[no_muon]=1; shift ;;
    --muon-adjust) MUON_ADJUST="$2"; CLI_SET[muon_adjust]=1; shift 2 ;;
    --eval-every)  EVAL_EVERY="$2"; CLI_SET[eval_every]=1; shift 2 ;;
    --lr-sweep)    LR_SWEEP=1; CLI_SET[lr_sweep]=1; shift ;;
    --sweep-steps) SWEEP_STEPS="$2"; CLI_SET[sweep_steps]=1; shift 2 ;;
    --model)       TAGS+=("${2%%=*}"); PATHS+=("${2#*=}"); CLI_SET[models]=1; shift 2 ;;
    --model-0p6b)  TAGS+=("qwen0p6b"); PATHS+=("$2"); CLI_SET[models]=1; shift 2 ;;
    --model-1p7b)  TAGS+=("qwen1p7b"); PATHS+=("$2"); CLI_SET[models]=1; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage; exit 1 ;;
  esac
done

# ---- load YAML config for anything the CLI did not set ----
[ -z "$CONFIG" ] && [ -f "$HERE/config.yaml" ] && CONFIG="$HERE/config.yaml"
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || die "config file not found: $CONFIG"
  # a bare python3 can read the config (stdlib-only parser), even to discover venv
  CFG_READER=""
  for c in python3 python "$PY"; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { CFG_READER="$c"; break; }
  done
  [ -n "$CFG_READER" ] || die "need python3 (or python) on PATH to read $CONFIG"

  declare -a YTAGS=() YPATHS=() YLABELS=()
  set_if_unset() { # cli_key varname value
    [ -n "${CLI_SET[$1]:-}" ] && return 0
    printf -v "$2" '%s' "$3"
  }
  while IFS=$'\t' read -r key v1 v2 v3; do
    case "$key" in
      MODEL)       YTAGS+=("$v1"); YPATHS+=("$v2"); YLABELS+=("$v3") ;;
      venv)        set_if_unset venv PY "$v1" ;;
      out)         set_if_unset out OUT "$v1" ;;
      gpus)        set_if_unset gpus GPUS "$v1" ;;
      steps)       set_if_unset steps STEPS "$v1" ;;
      seq)         set_if_unset seq SEQ "$v1" ;;
      arch)        set_if_unset arch ARCH "$v1" ;;
      seed)        set_if_unset seed SEED "$v1" ;;
      dataset)     set_if_unset dataset DATASET "$v1" ;;
      opts)        set_if_unset opts OPTS "$v1" ;;
      muon_adjust) set_if_unset muon_adjust MUON_ADJUST "$v1" ;;
      eval_every)  set_if_unset eval_every EVAL_EVERY "$v1" ;;
      sweep_steps) set_if_unset sweep_steps SWEEP_STEPS "$v1" ;;
      no_muon)     [ -n "${CLI_SET[no_muon]:-}" ] || { [ "$v1" = "true" ] && NO_MUON=1 || NO_MUON=0; } ;;
      lr_sweep)    [ -n "${CLI_SET[lr_sweep]:-}" ] || { [ "$v1" = "true" ] && LR_SWEEP=1 || LR_SWEEP=0; } ;;
      ""|\#*)      : ;;
      *)           echo "WARN: ignoring unknown config key '$key'" >&2 ;;
    esac
  done < <("$CFG_READER" "$HERE/load_config.py" "$CONFIG")

  # models from config only if the CLI gave none
  if [ -z "${CLI_SET[models]:-}" ] && [ "${#YTAGS[@]}" -gt 0 ]; then
    for i in "${!YTAGS[@]}"; do
      TAGS+=("${YTAGS[$i]}"); PATHS+=("${YPATHS[$i]}")
      [ -n "${YLABELS[$i]}" ] && LABELS["${YTAGS[$i]}"]="${YLABELS[$i]}"
    done
  fi
fi

# ---- validate ----
[ -n "$PY" ]   || die "no python given (--venv or config 'venv:')"
[ -x "$PY" ] || [ -f "$PY" ] || die "venv python not found/executable: $PY"
[ -n "$OUT" ]  || die "no output dir (--out or config 'out:')"
[ -n "$GPUS" ] || die "no GPUs (--gpus or config 'gpus:'); see --list-gpus"
[ "${#TAGS[@]}" -gt 0 ] || die "no models (--model... or config 'models:')"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[ "${#GPU_ARR[@]}" -gt 0 ] || die "no GPUs parsed from '$GPUS'"

# Tables/plots compare optimizers within a model side by side, which is only
# apples-to-apples on one GPU type; round-robin pins by index, not model, so warn
# (don't fail) if the chosen pool mixes GPU models. Indices match PCI_BUS_ID order.
if command -v nvidia-smi >/dev/null 2>&1; then
  declare -A GPU_NAME=()
  while IFS=',' read -r _idx _name; do
    GPU_NAME["${_idx// /}"]="${_name# }"
  done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
  declare -a SEL_NAMES=()
  for g in "${GPU_ARR[@]}"; do SEL_NAMES+=("${GPU_NAME[$g]:-unknown}"); done
  if [ "$(printf '%s\n' "${SEL_NAMES[@]}" | sort -u | wc -l)" -gt 1 ]; then
    echo "WARN: mixed GPU types in pool (${SEL_NAMES[*]}); within-model tok/s comparisons aren't apples-to-apples" >&2
  fi
fi

# --no-muon strips gefen_muon whether it came from the default set, config or --opts
if [ "$NO_MUON" -eq 1 ]; then
  OPTS="$(echo "$OPTS" | tr ' ' '\n' | grep -vx 'gefen_muon' | tr '\n' ' ')"
fi

echo "config:  ${CONFIG:-<none>}"
echo "venv:    $PY"
echo "out:     $OUT"
echo "gpus:    ${GPU_ARR[*]}  (PCI order)"
echo "models:  ${TAGS[*]}"
echo "opts:    $OPTS"
echo "steps:   $STEPS   lr_sweep: $LR_SWEEP"
[ "$DRY" -eq 1 ] && { echo "(dry run: config resolved OK; not launching)"; exit 0; }

mkdir -p "$OUT" "$OUT/logs"
RESULTS="$OUT/results.jsonl"
SWEEP="$OUT/results_lrsweep.jsonl"
LOGS="$OUT/logs"
CELL="$HERE/sweep_cell.py"
# Background cells append a line here on failure; the pool's `wait -n` reaps pids
# before the phase `wait`, so per-pid status is unrecoverable — collect via file.
FAILLOG="$OUT/failures.log"

abort_on_failures() { # phase-name
  [ -s "$FAILLOG" ] || return 0
  echo "ERROR: $1: $(wc -l < "$FAILLOG") cell(s) failed:" >&2
  cat "$FAILLOG" >&2
  exit 1
}

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
  local rc=$?
  echo "[done  $(date +%T)] $tag $opt rc=$rc"
  [ "$rc" -ne 0 ] && echo "$tag $opt lr=$lr gpu=$gpu rc=$rc (see $log)" >> "$FAILLOG"
  return "$rc"
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
  : > "$SWEEP"; : > "$FAILLOG"
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
  abort_on_failures "LR sweep"
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    for opt in $OPTS; do
      BEST_LR["${tag}_${opt}"]="$("$PY" "$HERE/pick_lr.py" --results "$SWEEP" --tag "$tag" --opt "$opt")"
      echo "  best LR  $tag $opt -> ${BEST_LR[${tag}_${opt}]}"
    done
  done
fi

# ---- finals ----
echo "=== FINALS (${STEPS} steps) ==="
: > "$RESULTS"; : > "$FAILLOG"
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"; model="${PATHS[$i]}"
  for opt in $OPTS; do
    if [ "$LR_SWEEP" -eq 1 ]; then lr="${BEST_LR[${tag}_${opt}]}"; else lr="$(fair_lr "$opt")"; fi
    dispatch "$tag" "$model" "$opt" "$lr" "$STEPS" "$RESULTS" "$LOGS/${tag}_${opt}.log"
  done
done
wait
abort_on_failures "finals"
rm -f "$FAILLOG"
echo "=== ALL FINALS DONE $(date +%T) ==="

# ---- table + charts ----
LABEL_ARGS=()
for i in "${!TAGS[@]}"; do
  t="${TAGS[$i]}"
  [ -n "${LABELS[$t]:-}" ] && LABEL_ARGS+=(--label "$t=${LABELS[$t]}")
done
"$PY" "$HERE/build_table.py" --results "$RESULTS" --out-dir "$OUT" ${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"}
"$PY" "$HERE/plot_quads.py"  --csv "$OUT/comparison_table.csv" --out-dir "$OUT"
"$PY" "$HERE/plot_curves.py" --csv "$OUT/comparison_table.csv" --logs "$LOGS" --out-dir "$OUT"

echo
echo "Outputs in $OUT :"
echo "  comparison_table.csv / comparison_table.md"
echo "  quad_<model>.png, loss_curve_<model>.png, tput_vram_<model>.png"
echo "DONE $(date +%T)"
