# FLUX 3 Image Klein Reference Studio (Diffusers server and GUI app)

[日本語](README.md) | English

This project runs FLUX 3 Image Klein with Diffusers, with the goal of fitting it into 8 GB of VRAM. It also includes a GUI app for trying out the API. It supports T2I, I2I, and Ref2I with up to 8 reference images.

> [!IMPORTANT]
> The application code is licensed under Apache-2.0, but the FLUX 3 model is licensed under the FLUX Kommunity License v1.0. Model weights are not included in this repository. See [Third-party models and licenses](THIRD_PARTY_NOTICES.md) for details.

## Getting started

| Step | Command | What it does |
|---|---|---|
| 1. Setup | `./setup.sh flux3 --acknowledge-flux3-license` | Creates the Python virtual environment (`.venv`), installs dependencies, and downloads the models. **It does not start the server** |
| 2. Start the FastAPI server | `./run.sh` | Starts the API server (uvicorn). The same server also serves the web app (GUI) |
| 3. Open the web app | http://127.0.0.1:7864/ in a browser | Nothing else needs to be started |

```bash
git clone https://github.com/animede/flux3-klein-studio.git
cd flux3-klein-studio
./setup.sh flux3 --acknowledge-flux3-license   # 1. first time only
CUDA_VISIBLE_DEVICES=0 ./run.sh                 # 2. start the server (Ctrl+C to stop)
# 3. open http://127.0.0.1:7864/ in a browser
```

API guide: [docs/API.en.md](docs/API.en.md), API reference: [docs/API_REFERENCE.en.md](docs/API_REFERENCE.en.md), techniques used: [Technical guide](docs/TECHNICAL_GUIDE.en.md)

## Model

| Value | Use | Image model | Text side | Default steps | License |
|---|---|---|---|---:|---|
| `flux3` | Latest-generation generation and editing | Official INT8 ConvRot distilled | Qwen3-VL-4B-Instruct (BF16, kept in CPU memory, computed on the GPU one layer at a time) | 16 | FLUX Kommunity License v1.0 |

Downloading the model requires an explicit option confirming that you have reviewed the license.

### FLUX 3 Image Klein vs. Base

This app uses the standard `black-forest-labs/flux-3-image-klein`.

| | FLUX 3 Image Klein | FLUX 3 Image Klein Base |
|---|---|---|
| Checkpoint | Distilled | Undistilled |
| Inference | 16 steps, no CFG | 50 steps, CFG 3.5 |
| Main use | Generation, editing, self-hosting | Fine-tuning, LoRA, research |
| Official H200 measurement (1 MP) | About 3 s | About 12 s |

Base is not a "higher-quality version for everyday use"; it is the base model for further training. The distilled model is the right choice for a regular image generation app.

### Memory and prompts

The official distilled INT8 ConvRot Transformer from the Comfy distribution is connected directly to the Diffusers ModularPipeline. The pipeline runs in stages: the ~5.43 GiB Transformer stays on the GPU. The Qwen3-VL text encoder (~8 GB) stays in CPU memory and is computed on the GPU one layer at a time (sequential offload, about 1 GiB of extra VRAM): about 2.5 s per prompt even for the long captions of the upsampler, against about 30 s on the CPU (`KLEIN_TEXT_ENCODER=cpu` keeps it on the CPU). Above 1024 px (e.g. 2048 px) the VAE is tiled. The default CUDA allocator limit is 8 GiB. ComfyUI is not used.

Measured (RTX PRO 4000 24GB, INT8, 16 steps, upsampler captions): text-to-image 1024×1024 about 12 s, one-reference edit 768×1024 about 18 s, two references about 30 s (all peaking at about 7.8 GiB). The 2K sizes BFL recommends (2048×2048, 2304×1728, etc.) are also available. Above 1024×1024 the allocator limit is raised automatically to 10 GiB (`KLEIN_HIRES_MEMORY_BUDGET_GIB`) (T2I 2048×2048 takes about 50 s with a peak of about 8.2 GiB, so 2K also works on 12 GB-class GPUs). Reference images are downscaled to at most 1 MP each and 2 MP in total (0.25 MP each for 8 images).

The model card requires a prompt upsampler (short prompts give generic output, and a one-line edit instruction can change the whole composition). This app runs a local LLM with the system prompts BFL publishes for external LLMs (`pup_prompts_external_vlm.json`, t2i / i2i / t2i_bbox / i2i_bbox). If you have GPU headroom, you can also use the official upsampler (quantized) (see "Using the official upsampler" below). **The prompt upsampler is on by default**: every generation first expands the input into a full caption.
For comparison, you can generate without it (bypass): uncheck "prompt upsamplerを使う" (use the prompt upsampler) in the GUI, or send `upsample=false` to the API.
To review the caption before generating, use the "prompt upsamplerで展開して確認" (expand and review) button in the GUI or `POST /api/enhance-prompt` (generating with the reviewed caption unchanged does not expand it again). For edits, the reference images are shown to the LLM as well.

Choose the LLM with "prompt upsamplerのLLM" (prompt upsampler LLM) in the GUI, `upsampler` in the API, or the `KLEIN_UPSAMPLER` environment variable at startup. The default, `auto`, tries them in this order. If none is available, the original prompt is used and the reason is returned as a warning (with `upsample_strict=true`, the job fails instead).

1. LLM servers: `KLEIN_PROMPT_LLM_URLS` (comma-separated OpenAI-compatible URLs, tried in order, e.g. `http://127.0.0.1:8080/v1`. If unset, no LLM server is used)
2. Local gemma-4-E4B (llama.cpp `llama-server` with an mmproj for image input)
   - If free VRAM is at least `KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB` (default 8), it runs on the GPU and is stopped right after use
   - Otherwise, it runs on the CPU when free RAM is at least `KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB` (default 10) and the CPU is not busy
   - On the CPU, a CPU-optimized `llama-server` build is used (`KLEIN_LOCAL_LLM_SERVER_CPU`, default `~/llama.cpp/build-cpu/bin/llama-server`).
     A GPU-oriented build (without `GGML_NATIVE`/`AVX2`) runs at only about 2 tokens/s on the CPU, so one expansion takes several minutes.
     Build the CPU version with `cmake -B build-cpu -DGGML_NATIVE=ON -DGGML_CUDA=OFF && cmake --build build-cpu --target llama-server`
     (about 10 tokens/s on an i5-12400F, 20–40 s per expansion)
   - Related settings: `KLEIN_LOCAL_LLM_SERVER` (GPU build), `KLEIN_LOCAL_LLM_MODEL`, `KLEIN_LOCAL_LLM_MMPROJ`, `KLEIN_LOCAL_LLM_PORT` (default 7865), `KLEIN_LOCAL_LLM_CTX` (default 16384); `KLEIN_LOCAL_LLM=0` disables it

External LLMs (LLM servers and the local gemma) get the system prompts BFL published for external LLMs on 2026-10-17 (fetched automatically on first use from a pinned commit of the official repository). The t2i prompt is about 10K tokens, so the LLM needs a context of 16K or more (LLM servers reuse the processed system prompt, so later requests are faster). To go back to the previous behaviour (the official upsampler's prompts for external LLMs too), set `KLEIN_LLM_PROMPTS=pup`. The official upsampler (`pup-*`) always gets its own prompts and format.

| `upsampler` | LLM used |
|---|---|
| `auto` (default) | LLM servers → local gemma-4-E4B |
| `remote` | LLM servers only |
| `local` | Local gemma-4-E4B only |
| `pup-int8` | INT8 version of the official upsampler (official BFL release) |
| `pup-gguf` | GGUF version of the official upsampler (Q4_K_M, community-made) |

#### Using the official upsampler

The official upsampler ([flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), based on Qwen3.5-9B, Apache-2.0) is an LLM tuned for FLUX 3.
It is better than gemma-4 at phrasing that preserves the reference image's composition and at fleshing out scenes, but it sometimes adds text that was not requested.
The bf16 version is 18.8 GB, so this app uses quantized versions. Both **assume VRAM use beyond the 8 GB target**.

| | `pup-int8` | `pup-gguf` |
|---|---|---|
| Files | Official BFL `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot` (9.8 GB, [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)) | Community-made Q4_K_M + mmproj (6.5 GB, [rekuen/flux-3-image-klein-pup-GGUF](https://huggingface.co/rekuen/flux-3-image-klein-pup-GGUF)) |
| Runs with | `pup_server.py` (Transformers + comfy-kitchen INT8 kernels) | llama.cpp `llama-server` |
| Free VRAM required (default) | 12 GiB (`KLEIN_PUP_INT8_MIN_FREE_GIB`) | 8 GiB (`KLEIN_PUP_GGUF_MIN_FREE_GIB`) |
| Measured (RTX PRO 4000 24GB) | Peak 9.4 GiB, about 19 s per expansion including startup | About 6.8 GiB, about 8 s per expansion including startup |

- It is started on the GPU on each use and stopped right afterwards (to give the VRAM back to the image model).
- If free VRAM is insufficient, the image model is temporarily released before starting (it is reloaded on the next generation).
- Download: after `./setup.sh flux3 --acknowledge-flux3-license`, run `.venv/bin/python download_models.py flux3 --acknowledge-flux3-license --pup int8` (`gguf` or `all` also work)
- Related settings: `KLEIN_PUP_INT8_PORT` (default 7867), `KLEIN_PUP_GGUF_PORT` (default 7866). The GGUF version uses the GPU build of llama-server (`KLEIN_LOCAL_LLM_SERVER`).

A CUDA 13 environment and the ConvRot kernels from `comfy-kitchen` are required.

## Setup

Requires Python 3.12, an NVIDIA CUDA GPU, enough RAM, and free disk space. `.venv`, `models`, `outputs`, and `uploads` are not tracked by Git.

### 1. Setup (dependencies and models)

Read the [official model page](https://huggingface.co/black-forest-labs/flux-3-image-klein) and the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md), and accept the access conditions on Hugging Face.
The model is gated, so log in with `hf auth login` or set the `HF_TOKEN` environment variable before running:

```bash
./setup.sh flux3 --acknowledge-flux3-license
```

`setup.sh` does three things, then exits. It does not start the server.

1. Creates `.venv` with `python3.12 -m venv .venv` if it does not exist
2. Installs the dependencies in `requirements.txt`
3. Downloads the models into `models/` with `download_models.py` (the base model, the INT8 version, and the outpaint LoRA)

To install only the dependencies, use `./setup.sh none`. Setup is not needed for later launches.
To also use the official upsampler, download it afterwards with `.venv/bin/python download_models.py flux3 --acknowledge-flux3-license --pup int8` (or `gguf`, `all`).

### 2. Start the FastAPI server

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh
```

`run.sh` starts `uvicorn app:app` with the Python in `.venv` (one worker). The server is ready when `Uvicorn running on http://127.0.0.1:7864` appears. Stop it with `Ctrl+C`.
The model is loaded at the first generation (a few seconds), not at startup.

Check that it is running:

```bash
curl http://127.0.0.1:7864/api/health
```

Main environment variables at startup:

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | | GPU to use |
| `KLEIN_HOST` | `127.0.0.1` | Address to listen on |
| `KLEIN_PORT` | `7864` | Port |
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorch GPU allocation limit (GiB) for outputs up to 1024×1024 |
| `KLEIN_HIRES_MEMORY_BUDGET_GIB` | `10.0` | Allocation limit (GiB) for 2K outputs (above 1024×1024) |
| `KLEIN_PROMPT_LLM_URLS` | | LLM servers for the prompt upsampler (OpenAI-compatible, comma-separated) |
| `KLEIN_UPSAMPLER` | `auto` | Default upsampler (`auto` / `remote` / `local` / `pup-int8` / `pup-gguf`) |

```bash
KLEIN_PROMPT_LLM_URLS=http://127.0.0.1:8080/v1 CUDA_VISIBLE_DEVICES=0 ./run.sh
```

See the [API reference](docs/API_REFERENCE.en.md#environment-variables) for the other environment variables.

### 3. Start the web app (GUI)

The web app is served by the server from step 2 on the same port, so nothing else needs to be started. Open these in a browser:

- Web app: http://127.0.0.1:7864/
- API UI (Swagger UI): http://127.0.0.1:7864/docs

The left side holds the inputs (reference images, request, prompt upsampler settings); the right side switches between the "生成結果" (Result) and "ギャラリー" (Gallery) tabs (`http://127.0.0.1:7864/#gallery` opens the gallery directly).
Because the left side stays visible, you can drag a result or gallery image onto the reference images on the left to use it as a reference for the next generation (or use the "参照に追加" (Add as reference) button).

- Inputs: layout control (bounding boxes) is collapsed; expand it when needed.
- Result: shown in the aspect ratio of the output size and kept on screen while scrolling. Click it to enlarge (same view as the gallery); the image can be downloaded.
- Gallery: shows previously generated images, newest first. Click an image to enlarge it (with its prompt, seed, and upsampler; use ←/→ to move between images; close with the ✕ at the top right, Esc, or a click on the backdrop), download it, add it as a reference, or delete it. Deleting also removes its settings JSON and reference images.

By default the server listens only on `127.0.0.1`, so it can be opened only from a browser on the same machine.
To use it from another machine, use SSH port forwarding (`ssh -L 7864:127.0.0.1:7864 <server>`) or start it with `KLEIN_HOST=0.0.0.0`.
The latter has no authentication, so use it only on a trusted network.

### Data sent externally

Image generation runs entirely on local files. Because the prompt upsampler is on by default, if `KLEIN_PROMPT_LLM_URLS` is set, the prompt and (for edits) the reference images are sent to those LLM servers on every generation.
If `KLEIN_PROMPT_LLM_URLS` is not set, no LLM server is used and expansion runs only on the local LLM (llama.cpp), so nothing is sent externally. Nothing is sent when `local`, `pup-int8`, or `pup-gguf` is selected, or when the upsampler is bypassed (`upsample=false`), either.

## API

The multipart form of `POST /api/generate` accepts `model=flux3`. FLUX 3 is also the default when omitted.

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A detailed cinematic scene of a character reading by a cafe window, soft afternoon light...' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=1024' -F 'height=768' -F 'seed=42'
```

- `GET /api/health`: available models, the currently loaded model, and server status
- `GET /api/models`: model values, quantization, and license information
- `GET /api/jobs/JOB_ID`: generation progress and the model used
- `POST /api/jobs/JOB_ID/cancel`: cancel a job
- `POST /api/unload`: release the current model
- `GET /api/gallery` / `DELETE /api/gallery/{id}`: list and delete generated images

See the [API guide](docs/API.en.md) for usage and the [API reference](docs/API_REFERENCE.en.md) for every endpoint's inputs, outputs, and errors. While the server is running, Swagger UI (`/docs`), ReDoc (`/redoc`), and the OpenAPI JSON (`/openapi.json`) are also available.

The server's status and error messages, and the GUI, are in Japanese.

## License

### This application

Copyright 2026 YUZUKI

The application code written for this repository is provided under the [Apache License 2.0](LICENSE).
The copyright notice and third-party notices are in [NOTICE](NOTICE). When redistributing, include `LICENSE` and `NOTICE`
as required by the Apache License 2.0.

```text
Copyright 2026 YUZUKI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Models (not included in this repository)

Models, model weights, tokenizers, and dependencies are each covered by their provider's license.
The Apache License 2.0 does not apply to the models.

| Model | License |
|---|---|
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (INT8 version from [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)) | [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md), [Usage Policy](https://bfl.ai/legal/usage-policy) |
| [FLUX 3 Image Klein Outpaint LoRA](https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA) (community-made, optional) | FLUX Kommunity License |
| [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) (by default only `system_prompts.json` is downloaded; `--pup int8` also downloads the config files) | Apache License 2.0 |
| INT8 version of the official upsampler (`--pup int8`, in [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)) | The original model is Apache License 2.0; the distributing repository is under the FLUX Kommunity License v1.0 |
| [rekuen/flux-3-image-klein-pup-GGUF](https://huggingface.co/rekuen/flux-3-image-klein-pup-GGUF) (`--pup gguf`, community-made) | Apache License 2.0 |
| [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (for comparison; not used by the app) | FLUX Kommunity License v1.0 |

The official notice for FLUX 3 Image Klein (`NOTICE.md` in the model repository), reproduced verbatim:

> This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

This app is not an official Black Forest Labs product and is not endorsed by Black Forest Labs.
See [Third-party models and licenses](THIRD_PARTY_NOTICES.md) for details. The latest original license texts at the linked pages always take precedence.
