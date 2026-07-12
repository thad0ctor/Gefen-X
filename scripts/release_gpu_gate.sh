#!/usr/bin/env bash
# Local two-GPU CUDA/JIT/distributed release gate for gefen-x.
#
# release.yml runs the CPU and Transformers Trainer wheel gates on hosted
# runners; this script is the GPU half of the release gate and runs on a local
# CUDA machine. Approving the `testpypi` environment in release.yml is the
# release manager's attestation that this script passed against that run's
# exact `dist` artifact. It mirrors the former gpu_release_tests workflow job:
# same runner preflight, same wheel install, same mandatory test list, and
# zero skipped tests allowed.
#
# Usage:
#   scripts/release_gpu_gate.sh v0.4.0              gate the tag's release-run
#                                                   artifact (downloads `dist`
#                                                   via `gh run download`)
#   scripts/release_gpu_gate.sh v0.4.0 --wheel PATH gate a local wheel instead
#
# Options:
#   --fresh         rebuild the cached gate venv from scratch
#   --no-tag-check  skip verifying that HEAD is the tagged commit (the test
#                   suite must otherwise come from the tag being gated)
#
# Environment:
#   GEFEN_GATE_PYTHON     base interpreter for the venv (default: python3). It
#                         must already provide a CUDA-enabled torch build; the
#                         venv is created with --system-site-packages so torch
#                         is inherited while the wheel, pytest, and accelerate
#                         installs stay inside the venv.
#   GEFEN_GATE_CACHE      venv cache directory
#                         (default: ~/.cache/gefen-x/release-gate)
#   GEFEN_GATE_REPO       GitHub repo whose release run holds the artifact
#                         (default: thad0ctor/Gefen-X)
#   GEFEN_GATE_EXTRA_SITE optional site-packages directory grafted into the
#                         gate venv via a .pth file, searched after the venv's
#                         own packages. Use it when the CUDA torch build that
#                         matches the local nvcc lives in another virtualenv
#                         rather than in the base interpreter (a venv base
#                         python cannot be inherited with
#                         --system-site-packages). Same-minor Python required.
#   CUDA_VISIBLE_DEVICES  choose GPUs; the gate uses the first two listed.

set -euo pipefail

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

TAG=""
WHEEL=""
FRESH=0
TAG_CHECK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --no-tag-check) TAG_CHECK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    v*) TAG="$1"; shift ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -z "$TAG" ]; then
  echo "error: a release tag (vX.Y.Z[...]) is required" >&2
  usage >&2
  exit 2
fi
EXPECTED_VERSION="${TAG#v}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$TAG_CHECK" -eq 1 ]; then
  TAG_COMMIT="$(git rev-parse --verify --quiet "refs/tags/${TAG}^{commit}" || true)"
  if [ -z "$TAG_COMMIT" ]; then
    echo "error: tag $TAG not found locally; fetch it or pass --no-tag-check" >&2
    exit 1
  fi
  if [ "$(git rev-parse HEAD)" != "$TAG_COMMIT" ]; then
    echo "error: HEAD is not $TAG — the gate must run the tagged test suite." >&2
    echo "  git checkout $TAG   (or pass --no-tag-check if you know better)" >&2
    exit 1
  fi
fi

WORK="$(mktemp -d)"
JIT_ROOT="$(mktemp -d)"
trap 'rm -rf "$WORK" "$JIT_ROOT"' EXIT

if [ -z "$WHEEL" ]; then
  command -v gh >/dev/null || { echo "error: gh CLI required to download the run artifact" >&2; exit 1; }
  # Tags publish from the fork, so don't let gh resolve `origin` (upstream).
  # The default matches the trusted-publisher registration in release.yml.
  GATE_REPO="${GEFEN_GATE_REPO:-thad0ctor/Gefen-X}"
  RUN_ID="$(gh run list -R "$GATE_REPO" --workflow release.yml --branch "$TAG" \
    --limit 1 --json databaseId --jq '.[0].databaseId')"
  if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
    echo "error: no release.yml run found for tag $TAG on $GATE_REPO" >&2
    exit 1
  fi
  echo "downloading dist artifact from $GATE_REPO release run $RUN_ID"
  gh run download "$RUN_ID" -R "$GATE_REPO" --name dist --dir "$WORK/dist"
  WHEEL="$(ls "$WORK"/dist/*.whl)"
fi
[ -f "$WHEEL" ] || { echo "error: wheel not found: $WHEEL" >&2; exit 1; }
echo "gating wheel: $WHEEL"
sha256sum "$WHEEL"

BASE_PYTHON="${GEFEN_GATE_PYTHON:-python3}"
CACHE_DIR="${GEFEN_GATE_CACHE:-$HOME/.cache/gefen-x/release-gate}"
VENV="$CACHE_DIR/venv"
if [ "$FRESH" -eq 1 ]; then
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "creating gate venv at $VENV (base: $BASE_PYTHON)"
  mkdir -p "$CACHE_DIR"
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
PYTHON="$VENV/bin/python"
SITE_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ -n "${GEFEN_GATE_EXTRA_SITE:-}" ]; then
  [ -d "$GEFEN_GATE_EXTRA_SITE" ] || { echo "error: GEFEN_GATE_EXTRA_SITE is not a directory: $GEFEN_GATE_EXTRA_SITE" >&2; exit 1; }
  printf '%s\n' "$GEFEN_GATE_EXTRA_SITE" > "$SITE_DIR/gefen_gate_extra.pth"
  echo "grafted extra site dir: $GEFEN_GATE_EXTRA_SITE"
else
  rm -f "$SITE_DIR/gefen_gate_extra.pth"
fi

"$PYTHON" - <<'PY'
import re
import shutil
import subprocess

import torch

assert torch.version.cuda is not None, "release gate requires a CUDA-enabled PyTorch build"
assert torch.cuda.is_available(), "release gate requires CUDA"
assert torch.cuda.device_count() >= 2, "release gate requires at least two visible GPUs"
assert torch.distributed.is_available(), "torch.distributed is unavailable"
assert torch.distributed.is_nccl_available(), "release gate requires NCCL"
names = [torch.cuda.get_device_name(index) for index in range(2)]
capabilities = [torch.cuda.get_device_capability(index) for index in range(2)]
assert names[0] == names[1], (
    "replica-exact gate requires identical GPU models",
    names,
)
assert capabilities[0] == capabilities[1], (
    "replica-exact gate requires homogeneous GPUs",
    capabilities,
)
nvcc = shutil.which("nvcc")
assert nvcc is not None, "release gate requires nvcc on PATH"
nvcc_result = subprocess.run(
    [nvcc, "--version"], check=True, capture_output=True, text=True
)
match = re.search(r"release\s+([0-9]+)\.", nvcc_result.stdout)
assert match is not None, "could not parse nvcc CUDA version"
assert int(match.group(1)) == int(torch.version.cuda.split(".")[0]), (
    "nvcc and PyTorch CUDA major versions differ",
    nvcc_result.stdout,
    torch.version.cuda,
)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPUs:", names)
print("capabilities:", capabilities)
print(nvcc_result.stdout)
PY

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  RELEASE_GPUS="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | cut -d, -f1,2)"
else
  RELEASE_GPUS=0,1
fi
echo "mandatory release tests will use CUDA_VISIBLE_DEVICES=$RELEASE_GPUS"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip uninstall -y gefen-x || true
"$PYTHON" -m pip install "$WHEEL" pytest 'accelerate==1.14.0'
EXPECTED_VERSION="$EXPECTED_VERSION" REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path

import gefen

version = metadata.version("gefen-x")
assert version == os.environ["EXPECTED_VERSION"], (version, os.environ["EXPECTED_VERSION"])
module_path = Path(gefen.__file__).resolve()
repo_src = (Path(os.environ["REPO_ROOT"]) / "src").resolve()
assert repo_src not in module_path.parents, module_path
print("installed wheel:", module_path, version)
PY
"$VENV/bin/ninja" --version 2>/dev/null || ninja --version

JUNIT="$WORK/release-gpu.xml"
CUDA_VISIBLE_DEVICES="$RELEASE_GPUS" \
GEFEN_KERNEL_BUILD_ROOT="$JIT_ROOT" \
GEFEN_VERBOSE_BUILD=1 \
"$PYTHON" -m pytest -q -ra --junitxml="$JUNIT" \
  tests/test_amp_grad_scaler.py \
  tests/test_capturable.py \
  tests/test_capturable_fsdp2.py \
  tests/test_deterministic_mode.py \
  tests/test_factored_v_ema_parity.py \
  tests/test_fused_fsdp2_noncontig.py \
  tests/test_fused_full_update_parity.py \
  tests/test_fused_update_v2_full_parity.py \
  tests/test_gefen_fsdp2_checkpoint.py \
  tests/test_muon_distributed_checkpoint_safety.py \
  tests/test_muon_grad_presence.py \
  tests/test_step_preflight_atomicity.py \
  tests/test_muon_fsdp2_parity.py::test_muon_fsdp2_fused_cuda_parity \
  tests/test_muon_fsdp2_parity.py::test_muon_fsdp2_fused_multirank_parity \
  tests/test_muon_fsdp2_parity.py::test_muon_fsdp2_distributed_parity \
  tests/test_muon_fsdp2_parity.py::test_muon_fsdp2_distributed_fluctuating_grad_set \
  tests/test_muon_fsdp2_parity.py::test_muon_fsdp2_distributed_checkpoint_restore

JUNIT="$JUNIT" "$PYTHON" - <<'PY'
import os
import xml.etree.ElementTree as ET

root = ET.parse(os.environ["JUNIT"]).getroot()
cases = root.findall(".//testcase")
assert cases, "release gate collected no tests"
skipped = [
    "{}::{}".format(case.attrib.get("classname", ""), case.attrib.get("name", ""))
    for case in cases
    if case.find("skipped") is not None
]
assert not skipped, "mandatory release tests skipped:\n{}".format("\n".join(skipped))
print("mandatory GPU release tests:", len(cases), "passed with zero skips")
PY

echo
echo "GPU release gate PASSED for $TAG"
echo "wheel: $(basename "$WHEEL")  sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"
echo "You may now approve the testpypi environment for this run."
