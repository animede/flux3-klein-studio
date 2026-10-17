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

"""Download the FLUX 3 model files at pinned revisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download, snapshot_download

from model_config import MODELS, PUP_MODELS


def download_model(config):
    snapshot_download(
        config["base_repo"], revision=config["base_revision"], local_dir=config["model_dir"],
        allow_patterns=config["download_patterns"], max_workers=3,
    )
    result = hf_hub_download(
        config["int8_repo"], config["int8_file"], revision=config["int8_revision"],
        local_dir=config["int8_dir"],
    )
    hf_hub_download(
        config["outpaint_lora_repo"], config["outpaint_lora_file"],
        revision=config["outpaint_lora_revision"], local_dir=config["int8_dir"],
    )
    return result


def download_pup(kind):
    """Official prompt upsampler for local use: `int8` (BFL's INT8 ConvRot) and/or `gguf` (Q4_K_M + mmproj)."""
    if kind in ("int8", "all"):
        cfg = PUP_MODELS["config"]
        snapshot_download(cfg["repo"], revision=cfg["revision"], local_dir=cfg["dir"], allow_patterns=cfg["patterns"])
        item = PUP_MODELS["int8"]
        hf_hub_download(item["repo"], item["file"], revision=item["revision"], local_dir=item["dir"])
    if kind in ("gguf", "all"):
        item = PUP_MODELS["gguf"]
        for name in item["files"]:
            hf_hub_download(item["repo"], name, revision=item["revision"], local_dir=item["dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download pinned FLUX 3 Image Klein model files."
    )
    parser.add_argument("model", nargs="?", default="flux3", choices=[*MODELS])
    parser.add_argument(
        "--acknowledge-flux3-license",
        action="store_true",
        help=(
            "Confirm that you reviewed and accepted the FLUX Kommunity License v1.0. "
            "Required for FLUX 3 downloads."
        ),
    )
    parser.add_argument(
        "--pup", choices=["int8", "gguf", "all"],
        help=(
            "Also download the official prompt upsampler for local use: int8 (BFL's INT8 ConvRot, 9.8 GB, "
            "about 12 GiB of free VRAM) and/or gguf (community Q4_K_M + mmproj, 6.5 GB, about 8 GiB)."
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_flux3_license:
        parser.error(
            "FLUX 3 Image Klein is released under the FLUX Kommunity License v1.0. "
            "Review https://huggingface.co/black-forest-labs/flux-3-image-klein "
            "and THIRD_PARTY_NOTICES.md, accept the official Hugging Face access "
            "terms, then rerun with --acknowledge-flux3-license."
        )
    with ThreadPoolExecutor(max_workers=1) as pool:
        print(pool.submit(download_model, MODELS[args.model]).result(), flush=True)
    print(f"Model download complete: {args.model}", flush=True)
    if args.pup:
        download_pup(args.pup)
        print(f"Prompt upsampler download complete: {args.pup}", flush=True)
