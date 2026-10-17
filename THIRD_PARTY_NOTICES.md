# Third-party models and licenses

This repository contains application source code. It does **not** distribute model
weights. The `models/` directory and generated images are excluded from Git.

The repository's [Apache License 2.0](LICENSE) applies only to the application code
authored for FLUX 3 Image Klein Reference Studio (Copyright 2026 YUZUKI; see [NOTICE](NOTICE)). It does not grant rights to model weights,
model names, third-party software, or generated content. The controlling license is
always the license published by each provider.

## FLUX 3 Image Klein

- Inference model: [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- Training/research checkpoint, not used by this application: [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)
- License: [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)
- Usage rules: [Black Forest Labs Usage Policy](https://bfl.ai/legal/usage-policy)
- INT8 (ConvRot) checkpoint used by this application: [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) (same license)

Official notice (`NOTICE.md` of the model repository), reproduced verbatim:

> This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

The FLUX 3 weights are not covered by this repository's Apache-2.0 license. The
FLUX Kommunity License permits non-commercial use and contains a commercial-output
exception for qualifying users with less than US$5 million in gross annualized
revenue, including affiliates, subject to its complete terms. It is not blanket
permission for commercial fine-tuning, model distribution, or every commercial
use. The license also contains use restrictions and requirements for derivatives
and redistribution.

The downloader therefore requires `--acknowledge-flux3-license`. This records only
the operator's acknowledgement; it does not replace acceptance of the gated model
terms on Hugging Face and does not modify the license. The application downloads
the distilled inference checkpoint only. It does not download the Base checkpoint
or the prompt upsampler model.

## FLUX 3 Image Klein Outpaint LoRA (optional)

- Model: [ausboss/FLUX-3-Image-Klein-Outpaint-LoRA](https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA) (community-made, pinned revision)
- License: FLUX Kommunity License (a Derivative of FLUX 3 Image Klein); the FLUX 3 terms above apply.

## FLUX 3 Image Klein prompt upsampler system prompts

- Source: [black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
- License: Apache License 2.0
- For LLMs other than the official upsampler, the application downloads `src/flux3/pup_prompts_external_vlm.json`
  from [black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein) (Apache License 2.0,
  pinned commit) on first use.
- By default the application downloads only `system_prompts.json` from this repository and
  sends those prompts to the configured LLM. The bf16 upsampler weights are not downloaded.

## Official prompt upsampler for local use (optional)

`download_models.py flux3 --acknowledge-flux3-license --pup int8|gguf|all` downloads quantized
builds of the official upsampler. They are not part of this repository.

- `--pup int8`: `text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors` from
  [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
  (pinned revision), plus the configuration, tokenizer and processor files of
  `flux-3-image-klein-pup`. The underlying model is released under the Apache License 2.0; the
  repository that distributes this file is covered by the FLUX Kommunity License v1.0, and you
  must accept its gated terms to download it.
- `--pup gguf`: `flux-3-image-klein-pup-Q4_K_M.gguf` and `mmproj-flux-3-image-klein-pup-F16.gguf`
  from [rekuen/flux-3-image-klein-pup-GGUF](https://huggingface.co/rekuen/flux-3-image-klein-pup-GGUF)
  (community-made, Apache License 2.0, pinned revision). It is not published by Black Forest Labs.

## Python packages

Packages installed from `requirements.txt` remain under their respective licenses.
They are downloaded from their package providers and are not relicensed by this
repository. Use `pip show <package>` and the linked upstream project when reviewing
a specific dependency.

## Names and marks

FLUX and Black Forest Labs names and marks belong to their respective owners. This
project is an independent integration and is not affiliated with or endorsed by
Black Forest Labs or Hugging Face.

This notice is a practical summary, not a substitute for the license texts. If a
summary conflicts with a provider's terms, the provider's terms control.
