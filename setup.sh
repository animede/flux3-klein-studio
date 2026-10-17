#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model_choice="${1:-flux3}"
if [[ "$model_choice" != "flux3" && "$model_choice" != "none" ]]; then
  echo "Usage: ./setup.sh [flux3|none] [--acknowledge-flux3-license]" >&2
  exit 2
fi
# Remove development-machine links if this folder was copied to another PC.
for path in .venv models/flux-3-image-klein models/flux-3-image-klein-int8; do
  if [[ -L "$path" && ! -e "$path" ]]; then
    rm "$path"
  fi
done
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
if [[ "$model_choice" != "none" ]]; then
  .venv/bin/python download_models.py "$model_choice" "${@:2}"
fi
