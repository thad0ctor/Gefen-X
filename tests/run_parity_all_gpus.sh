#!/usr/bin/env bash
# Run the fused-update parity suite on every visible GPU in parallel. Each GPU
# gets its own process (so the JIT kernel build targets that GPU's arch) and its
# own TORCH_EXTENSIONS_DIR (so the per-arch builds don't race on the cache).
#
# Usage:
#   PYTHON=/path/to/python bash tests/run_parity_all_gpus.sh
# Honors a caller-provided CUDA_VISIBLE_DEVICES (indices, or UUID/MIG ids); if
# unset, sweeps 0..device_count-1. CUDA_HOME must point at an nvcc matching
# torch.version.cuda, as for any Gefen build.
set -uo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEST="$HERE/test_fused_update_v2_parity.py"

# Device list: honor the caller's CUDA_VISIBLE_DEVICES (a subset / reordered /
# UUID list) if set; otherwise enumerate every device. Fail fast if discovery
# errors or finds nothing, so a CPU-only / misconfigured host can't look green.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a DEVICES <<<"$CUDA_VISIBLE_DEVICES"
else
  NGPU="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)" || NGPU=""
  if ! [[ "$NGPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU discovery failed (could not read torch.cuda.device_count())." >&2
    exit 1
  fi
  DEVICES=()
  for ((i = 0; i < NGPU; i++)); do DEVICES+=("$i"); done
fi
N=${#DEVICES[@]}
if [ "$N" -eq 0 ]; then
  echo "ERROR: no CUDA devices to sweep." >&2
  exit 1
fi

TMP="$(mktemp -d)"
echo "running fused-update parity on $N GPU(s): ${DEVICES[*]}"
pids=()
for ((k = 0; k < N; k++)); do
  CUDA_VISIBLE_DEVICES="${DEVICES[$k]}" TORCH_EXTENSIONS_DIR="$TMP/ext_$k" \
    "$PY" "$TEST" >"$TMP/gpu_$k.log" 2>&1 &
  pids+=("$!")
done

rc=0
for ((k = 0; k < N; k++)); do
  if wait "${pids[$k]}"; then status=PASS; else status=FAIL; rc=1; fi
  name="$(grep -m1 '=== device' "$TMP/gpu_$k.log" | sed 's/=== //; s/ ===//')"
  echo "GPU ${DEVICES[$k]} [$status] ${name:-unknown}"
  grep -E '^\[[0-9]\]|OVERALL' "$TMP/gpu_$k.log" | sed 's/^/    /'
done

echo ""
if [ "$rc" -eq 0 ]; then
  echo "ALL GPUS: PASS"
  rm -rf "$TMP"  # only clean up on success; keep logs/artifacts for debugging on failure
else
  echo "ALL GPUS: FAIL (logs kept in $TMP)"
fi
exit "$rc"
