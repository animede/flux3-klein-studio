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
        "text_quantization": "BF16 (CPU memory, GPU sequential offload)",
        "steps": 16,
        "guidance": None,
        # Timestep shift (alpha): BFL changed the default from 6.93 to 16 on 2026-10-17
        # (reference code black-forest-labs/flux-3-image-klein 62e90b92 and the ComfyUI workflows).
        "shift": 16.0,
        # References: up to 1 MP each (BFL's reference code), 2 MP in total so 8 references still fit 8 GiB.
        "reference_max_pixels": 1024 * 1024,
        "reference_total_pixels": 2 * 1024 * 1024,
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

# Official prompt upsampler (flux-3-image-klein-pup, Qwen3.5-9B, Apache-2.0) for local use. Optional;
# `download_models.py flux3 --pup int8|gguf|all` fetches them. Both need a large GPU (see upsampler.py).
PUP_CONFIG_DIR = ROOT / "models" / "flux-3-image-klein-pup"
PUP_MODELS = {
    # Config, tokenizer and processor files only; the INT8 weights come from the Comfy repository.
    "config": {
        "repo": "black-forest-labs/flux-3-image-klein-pup",
        "revision": "3f3edb52b2c55de0d49feab28598b68f7f8e4830",
        "dir": PUP_CONFIG_DIR,
        "patterns": ["*.json", "*.jinja", "README.md"],
    },
    # Official INT8 ConvRot build from the Comfy distribution (same pinned revision as the DiT).
    "int8": {
        "repo": "black-forest-labs/flux-3-image-klein-comfy",
        "revision": MODELS["flux3"]["int8_revision"],
        "file": "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
        "dir": MODELS["flux3"]["int8_dir"],
    },
    # Community GGUF (Q4_K_M + vision mmproj) for llama.cpp.
    "gguf": {
        "repo": "rekuen/flux-3-image-klein-pup-GGUF",
        "revision": "4dc05841418c455df85470d3befae8d270d880d4",
        "files": ["flux-3-image-klein-pup-Q4_K_M.gguf", "mmproj-flux-3-image-klein-pup-F16.gguf"],
        "dir": ROOT / "models" / "flux-3-image-klein-pup-gguf",
    },
}
PUP_INT8_PATH = PUP_MODELS["int8"]["dir"] / PUP_MODELS["int8"]["file"]
PUP_GGUF_PATH = PUP_MODELS["gguf"]["dir"] / PUP_MODELS["gguf"]["files"][0]
PUP_MMPROJ_PATH = PUP_MODELS["gguf"]["dir"] / PUP_MODELS["gguf"]["files"][1]
