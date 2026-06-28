#!/usr/bin/env bash
# Reproducible FSDP2 sharding-sweep launcher for GefenMuon.
#
# For every world_size x sharded_mode, launches fsdp2_convergence.py under
# `torchrun --nproc_per_node=<world>` pinned to the first <world> GPUs of the
# pool, appends the cell's RESULT json to <out>/results_fsdp2.jsonl, saves a
# per-cell log under <out>/logs/, then renders the two charts. Settings come
# from CLI flags and/or a YAML config; there are NO hardcoded paths.
#
#   precedence:  CLI flag  >  config.yaml  >  built-in default
#
# A config is loaded from --config <path>, else ./config.yaml if it exists
# (see config.example.yaml). GPU indices are PCI-bus-ordered
# (CUDA_DEVICE_ORDER=PCI_BUS_ID); list them with --list-gpus.
#
# See README.md for the full regime and a concrete example.
set -u

# PCI-bus order so the indices in --gpus / config match `--list-gpus` and the
# physical GPUs torchrun pins (children inherit this).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Unbuffered so the per-cell step logs stream live into the log files.
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- built-in defaults (overridden by config.yaml, then by CLI) ----
PY=""
OUT=""
GPUS=""
MODEL=""
TAG="qwen0p6b"
WORLD_SIZES="2 4"
MODES="exact approx"
STEPS=2000
SEQ=2048
ARCH="8.6"
SEED=0
LR="5e-5"
DATASET="tatsu-lab/alpaca"
EVAL_EVERY=50
MAX_EVAL=32
MUON_ADJUST="match_rms_adamw"
CUDA_HOME_CFG=""        # optional; exported only if set (else inherit env)
CONFIG=""
DRY=0

declare -A CLI_SET=()

die() { echo "ERROR: $*" >&2; exit 1; }

list_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
  echo "PCI-ordered GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID) — use the index column in --gpus / gpus:"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total \
    --format=csv,noheader | awk -F', ' '{printf "  %s  %-28s  %s  used %s / %s\n",$1,$2,$3,$4,$5}'
}

usage() {
  cat <<EOF
Usage: run.sh [--config config.yaml] [options]
       run.sh --venv <python> --out <dir> --gpus "1,4,5,7" --model <path> \\
              [--world-sizes "2 4"] [--modes "exact approx"]

Settings come from CLI flags and/or a YAML config (see config.example.yaml).
Precedence: CLI flag > config.yaml > built-in default. ./config.yaml is loaded
automatically when present; --config <path> points elsewhere.

Required (here or in the config):
  --venv <python>        Python of the env with the Gefen fork installed
                         (pip install -e . on THIS fork) + datasets,
                         transformers, matplotlib. torch must have FSDP2
                         (fully_shard) — torch >= 2.5.
  --out <dir>            Output dir for results_fsdp2.jsonl, logs/, charts.
  --gpus "a,b,c"         PCI-ordered CUDA device indices. Each world_size W
                         cell pins the FIRST W of these; the pool must have at
                         least max(world_sizes) GPUs.
  --model <path>         Model to fine-tune (e.g. /models/Qwen3-0.6B).

Options:
  --config <path>        YAML config file (default: ./config.yaml if it exists).
  --list-gpus            Print the PCI-ordered GPU list and exit.
  --dry-run, --check     Resolve config + CLI, print effective settings, exit.
  --tag NAME             Result tag / chart label key (default $TAG).
  --world-sizes "2 4"    Shard counts to sweep (default "$WORLD_SIZES").
  --modes "exact approx" sharded_mode values to sweep (default "$MODES").
  --steps N              Steps per cell (default $STEPS).
  --seq N                Sequence / pack-block length (default $SEQ).
  --lr X                 Constant LR (default $LR).
  --arch X.Y             TORCH_CUDA_ARCH_LIST (8.6 Ampere, 12.0 Blackwell; default $ARCH).
  --seed N               Random seed / data order (default $SEED).
  --dataset NAME         HF Alpaca-format dataset id (default $DATASET).
  --eval-every N         Intermediate eval-logging cadence (default $EVAL_EVERY).
  --max-eval N           Held-out eval examples (default $MAX_EVAL).
  --muon-adjust MODE     GefenMuonHybrid adjust_lr_fn (default $MUON_ADJUST).
  --cuda-home PATH       Export CUDA_HOME for the JIT kernel build (else inherit env).
  -h, --help             This help.

Example (all from config):
  cp config.example.yaml config.yaml && \$EDITOR config.yaml
  ./run.sh

Example (pure CLI):
  ./run.sh --venv /path/to/venv/bin/python --out ./out \\
           --model /models/Qwen3-0.6B --gpus "1,4,5,7" \\
           --world-sizes "2 4" --steps 2000 --arch 8.6
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
    --model)       MODEL="$2"; CLI_SET[model]=1; shift 2 ;;
    --tag)         TAG="$2"; CLI_SET[tag]=1; shift 2 ;;
    --world-sizes) WORLD_SIZES="$2"; CLI_SET[world_sizes]=1; shift 2 ;;
    --modes)       MODES="$2"; CLI_SET[modes]=1; shift 2 ;;
    --steps)       STEPS="$2"; CLI_SET[steps]=1; shift 2 ;;
    --seq)         SEQ="$2"; CLI_SET[seq]=1; shift 2 ;;
    --lr)          LR="$2"; CLI_SET[lr]=1; shift 2 ;;
    --arch)        ARCH="$2"; CLI_SET[arch]=1; shift 2 ;;
    --seed)        SEED="$2"; CLI_SET[seed]=1; shift 2 ;;
    --dataset)     DATASET="$2"; CLI_SET[dataset]=1; shift 2 ;;
    --eval-every)  EVAL_EVERY="$2"; CLI_SET[eval_every]=1; shift 2 ;;
    --max-eval)    MAX_EVAL="$2"; CLI_SET[max_eval]=1; shift 2 ;;
    --muon-adjust) MUON_ADJUST="$2"; CLI_SET[muon_adjust]=1; shift 2 ;;
    --cuda-home)   CUDA_HOME_CFG="$2"; CLI_SET[cuda_home]=1; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; usage; exit 1 ;;
  esac
done

# ---- load YAML config for anything the CLI did not set ----
[ -z "$CONFIG" ] && [ -f "$HERE/config.yaml" ] && CONFIG="$HERE/config.yaml"
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || die "config file not found: $CONFIG"
  CFG_READER=""
  for c in python3 python "$PY"; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { CFG_READER="$c"; break; }
  done
  [ -n "$CFG_READER" ] || die "need python3 (or python) on PATH to read $CONFIG"

  set_if_unset() { # cli_key varname value
    [ -n "${CLI_SET[$1]:-}" ] && return 0
    printf -v "$2" '%s' "$3"
  }
  while IFS=$'\t' read -r key v1 _v2 _v3; do
    case "$key" in
      venv)        set_if_unset venv PY "$v1" ;;
      out)         set_if_unset out OUT "$v1" ;;
      gpus)        set_if_unset gpus GPUS "$v1" ;;
      model)       set_if_unset model MODEL "$v1" ;;
      tag)         set_if_unset tag TAG "$v1" ;;
      world_sizes) set_if_unset world_sizes WORLD_SIZES "$v1" ;;
      modes)       set_if_unset modes MODES "$v1" ;;
      steps)       set_if_unset steps STEPS "$v1" ;;
      seq)         set_if_unset seq SEQ "$v1" ;;
      lr)          set_if_unset lr LR "$v1" ;;
      arch)        set_if_unset arch ARCH "$v1" ;;
      seed)        set_if_unset seed SEED "$v1" ;;
      dataset)     set_if_unset dataset DATASET "$v1" ;;
      eval_every)  set_if_unset eval_every EVAL_EVERY "$v1" ;;
      max_eval)    set_if_unset max_eval MAX_EVAL "$v1" ;;
      muon_adjust) set_if_unset muon_adjust MUON_ADJUST "$v1" ;;
      cuda_home)   set_if_unset cuda_home CUDA_HOME_CFG "$v1" ;;
      ""|\#*)      : ;;
      MODEL)       : ;;  # ignore optimizer-sweep-style models: list if present
      *)           echo "WARN: ignoring unknown config key '$key'" >&2 ;;
    esac
  done < <("$CFG_READER" "$HERE/load_config.py" "$CONFIG")
fi

# ---- validate ----
[ -n "$PY" ]    || die "no python given (--venv or config 'venv:')"
[ -x "$PY" ] || die "venv python not found/executable: $PY"
[ -n "$OUT" ]   || die "no output dir (--out or config 'out:')"
[ -n "$GPUS" ]  || die "no GPUs (--gpus or config 'gpus:'); see --list-gpus"
[ -n "$MODEL" ] || die "no model (--model or config 'model:')"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[ "${#GPU_ARR[@]}" -gt 0 ] || die "no GPUs parsed from '$GPUS'"

# Each world_size W cell pins the first W GPUs; the pool must be big enough.
MAXW=0
for w in $WORLD_SIZES; do
  case "$w" in (*[!0-9]*|"") die "world_sizes must be positive ints, got '$w'";; esac
  [ "$w" -gt "$MAXW" ] && MAXW="$w"
done
[ "${#GPU_ARR[@]}" -ge "$MAXW" ] \
  || die "need >= $MAXW GPUs for world_sizes '$WORLD_SIZES' but pool has ${#GPU_ARR[@]} (${GPUS})"

for mode in $MODES; do
  case "$mode" in exact|approx) ;; *) die "modes must be exact/approx, got '$mode'";; esac
done

# Warn (don't fail) if the chosen pool mixes GPU types — step-time comparisons
# across cells are only apples-to-apples on one GPU type.
if command -v nvidia-smi >/dev/null 2>&1; then
  declare -A GPU_NAME=()
  while IFS=',' read -r _idx _name; do
    GPU_NAME["${_idx// /}"]="${_name# }"
  done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
  declare -a SEL_NAMES=()
  for g in "${GPU_ARR[@]:0:$MAXW}"; do SEL_NAMES+=("${GPU_NAME[$g]:-unknown}"); done
  if [ "$(printf '%s\n' "${SEL_NAMES[@]}" | sort -u | wc -l)" -gt 1 ]; then
    echo "WARN: mixed GPU types in the first $MAXW of the pool (${SEL_NAMES[*]}); step-time comparisons aren't apples-to-apples" >&2
  fi
fi

echo "config:      ${CONFIG:-<none>}"
echo "venv:        $PY"
echo "out:         $OUT"
echo "gpus:        ${GPU_ARR[*]}  (PCI order)"
echo "model/tag:   $MODEL  [$TAG]"
echo "world_sizes: $WORLD_SIZES"
echo "modes:       $MODES"
echo "steps:       $STEPS   seq: $SEQ   lr: $LR   seed: $SEED"
[ "$DRY" -eq 1 ] && { echo "(dry run: config resolved OK; not launching)"; exit 0; }

mkdir -p "$OUT" "$OUT/logs"
RESULTS="$OUT/results_fsdp2.jsonl"
LOGS="$OUT/logs"
CELL="$HERE/fsdp2_convergence.py"
: > "$RESULTS"

[ -n "$CUDA_HOME_CFG" ] && export CUDA_HOME="$CUDA_HOME_CFG"
export TORCH_CUDA_ARCH_LIST="$ARCH"

FAILED=0
PORT=$(( 29500 + (RANDOM % 1000) ))
for world in $WORLD_SIZES; do
  # first <world> GPUs of the pool, comma-joined for CUDA_VISIBLE_DEVICES
  sel=$(IFS=,; echo "${GPU_ARR[*]:0:$world}")
  for mode in $MODES; do
    log="$LOGS/${TAG}_w${world}_${mode}.log"
    PORT=$(( PORT + 1 ))
    echo "[start $(date +%T)] world=$world mode=$mode gpus=$sel steps=$STEPS -> $log"
    CUDA_VISIBLE_DEVICES="$sel" \
    "$PY" -m torch.distributed.run --nproc_per_node="$world" \
      --master-port="$PORT" "$CELL" \
        --model "$MODEL" --lr "$LR" --seed "$SEED" --steps "$STEPS" \
        --seq "$SEQ" --sharded-mode "$mode" --tag "$TAG" \
        --dataset "$DATASET" --eval-every "$EVAL_EVERY" --max-eval "$MAX_EVAL" \
        --muon-adjust "$MUON_ADJUST" --out "$RESULTS" > "$log" 2>&1
    rc=$?
    echo "[done  $(date +%T)] world=$world mode=$mode rc=$rc"
    [ "$rc" -ne 0 ] && { echo "  FAILED (see $log)" >&2; FAILED=1; }
  done
done

[ "$FAILED" -eq 0 ] || die "one or more cells failed; see $LOGS/*.log"
echo "=== ALL CELLS DONE $(date +%T) ==="

# ---- charts ----
"$PY" "$HERE/plot_sharding.py" --results "$RESULTS" --logs "$LOGS" \
  --out-dir "$OUT" --tag "$TAG"

echo
echo "Outputs in $OUT :"
echo "  results_fsdp2.jsonl"
echo "  muon_shard_loss.png, muon_shard_perf.png"
echo "  logs/${TAG}_w<world>_<mode>.log"
echo "DONE $(date +%T)"
