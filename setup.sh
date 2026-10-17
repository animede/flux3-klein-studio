#!/usr/bin/env bash
# Copyright 2026 YUZUKI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
