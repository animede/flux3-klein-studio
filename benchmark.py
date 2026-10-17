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

"""Run one local generation and print measured time and PyTorch VRAM."""
import json
import os
import threading
from pathlib import Path

from engine import Engine
from PIL import Image


def progress(status, step, message):
    print(f"[{status}:{step}] {message}", flush=True)


root = Path(__file__).resolve().parent
reference_path = root / "outputs" / "benchmark-512.png"
reference = Image.open(reference_path).convert("RGB") if reference_path.is_file() else None
small_reference = reference.resize((384, 384), Image.Resampling.LANCZOS) if reference is not None else None
cases = [
    {
        "name": "t2i-768",
        "prompt": "A cinematic landscape of a quiet lakeside observatory at sunset, detailed illustration.",
        "reference": None,
        "width": 768,
        "height": 768,
        "seed": 43,
    },
    {
        "name": "i2i-512",
        "prompt": (
            "Create a new image of the same red panda character from the reference, "
            "painting at an artist studio. Preserve the face, teal scarf, and illustration style."
        ),
        "reference": reference,
        "width": 512,
        "height": 512,
        "seed": 44,
    },
    {
        "name": "ref8-512",
        "prompt": (
            "Create a new image of the same red panda character from the references, "
            "walking through a flower garden. Preserve the face, teal scarf, and illustration style."
        ),
        "reference": [small_reference] * 8 if small_reference is not None else None,
        "width": 512,
        "height": 512,
        "seed": 45,
    },
]

engine = Engine()
model_key = os.getenv("BENCH_MODEL", "flux3")
for case in cases:
    selected = os.getenv("BENCH_CASE")
    if selected and case["name"] != selected:
        continue
    if case["reference"] is None and case["name"].startswith("i2i"):
        continue
    image, metrics = engine.generate(
        model_key=model_key, prompt=case["prompt"], reference=case["reference"],
        width=case["width"], height=case["height"], seed=case["seed"],
        progress=progress, cancel=threading.Event(),
    )
    output = root / "outputs" / f"benchmark-{case['name']}.png"
    output.parent.mkdir(exist_ok=True)
    image.save(output)
    print(json.dumps({"case": case["name"], "output": str(output), **metrics}, ensure_ascii=False, indent=2))
