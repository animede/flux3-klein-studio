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

from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = {
    "flux3": {
        "family": "flux3",
        "name": "FLUX 3 Image Klein",
        "description": "最新世代・蒸留済み推論モデル",
        "license": "FLUX Kommunity License v1.0",
        "model_dir": ROOT / "models" / "flux-3-image-klein",
        "int8_dir": ROOT / "models" / "flux-3-image-klein-int8",
        "base_repo": "black-forest-labs/flux-3-image-klein",
        "base_revision": "337337621b8526e120f2275d250242f4a6bedd75",
        "int8_repo": "black-forest-labs/flux-3-image-klein-comfy",
        "int8_revision": "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0",
        "int8_file": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
        "outpaint_lora_repo": "ausboss/FLUX-3-Image-Klein-Outpaint-LoRA",
        "outpaint_lora_revision": "d5c120af172d7c3018fdb1b02adf1194db23eca5",
        "outpaint_lora_file": "flux-3-image-klein-outpaint.safetensors",
        "quantization": "Official INT8 ConvRot",
        "text_quantization": "BF16 (CPU)",
        "steps": 16,
        "guidance": None,
        "reference_pixel_budget": 8 * 768**2,
        "download_patterns": [
            "modular_model_index.json",
            "transformer/config.json",
            "text_encoder/*",
            "vae/*",
            "README.md",
            "LICENSE.md",
            "NOTICE.md",
        ],
    },
}

DEFAULT_MODEL = "flux3"
