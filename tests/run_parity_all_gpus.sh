#!/usr/bin/env bash
# Run the fused-update parity suite on every visible GPU in parallel. Each GPU
# gets its own process (so the JIT kernel build targets that GPU's arch) and its
# own TORCH_EXTENSIONS_DIR (so the per-arch builds don't race on the cache).
#
# Usage:
#   PYTHON=/path/to/python bash tests/run_parity_all_gpus.sh
# (CUDA_HOME must point at an nvcc matching torch.version.cuda, as for any build.)
set -uo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEST="$HERE/test_fused_update_v2_parity.py"
NGPU="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
TMP="$(mktemp -d)"

echo "running fused-update parity on $NGPU GPU(s) in parallel..."
pids=()
for ((i = 0; i < NGPU; i++)); do
  CUDA_VISIBLE_DEVICES="$i" TORCH_EXTENSIONS_DIR="$TMP/ext_$i" \
    "$PY" "$TEST" >"$TMP/gpu_$i.log" 2>&1 &
  pids+=("$!")
done

rc=0
for ((i = 0; i < NGPU; i++)); do
  if wait "${pids[$i]}"; then status=PASS; else status=FAIL; rc=1; fi
  name="$(grep -m1 '=== device' "$TMP/gpu_$i.log" | sed 's/=== //; s/ ===//')"
  echo "GPU $i [$status] ${name:-unknown}"
  grep -E '^\[[0-9]\]|OVERALL' "$TMP/gpu_$i.log" | sed 's/^/    /'
done

echo ""
if [ "$rc" -eq 0 ]; then echo "ALL GPUS: PASS"; else echo "ALL GPUS: FAIL (logs in $TMP)"; fi
exit "$rc"
