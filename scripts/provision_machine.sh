#!/usr/bin/env bash
# Provision a fresh machine for blood-analyzer development from a clean clone.
# Usage: bash scripts/provision_machine.sh [/path/to/repo]
#   MODE=minimal  installs only the lightweight venv (torch CPU, Pillow, numpy, pytest, ruff, mypy).
#   MODE=full     installs everything (default): torch CPU first so pip skips the CUDA build,
#                 then the package with ml+dev extras, which also provides the `bloodfilm` CLI.
set -euo pipefail

REPO="${1:-/opt/blood}"
MODE="${MODE:-full}"
VENV="$REPO/.venv"

python3 -m venv "$VENV" --upgrade-deps
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

if [ "$MODE" = "full" ]; then
  pip install -e "$REPO[ml,dev]"
else
  pip install -e "$REPO" --no-deps
  pip install Pillow numpy pytest ruff mypy
fi

cd "$REPO"
bloodfilm assets download mll23 --report-output outputs/reports/mll23_download.json
bloodfilm assets verify mll23 --report-output outputs/reports/mll23_verify.json
bloodfilm dataset audit mll23 --report-output outputs/reports/mll23_audit.json
bloodfilm dataset build-manifest mll23 \
  --output data/manifests/mll23_manifest.csv \
  --invalid-output data/manifests/mll23_invalid.csv \
  --report-output outputs/reports/mll23_manifest.json

# Expected to fail until DinoBloom-B weights are placed manually; the error states the next step.
bloodfilm smoke --config configs/base.yaml || true
