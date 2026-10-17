"""Download the FLUX 3 model files at pinned revisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download, snapshot_download

from model_config import MODELS


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
