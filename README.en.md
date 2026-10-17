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

API guide: [docs/API.en.md](docs/API.en.md), API reference: [docs/API_REFERENCE.en.md](docs/API_REFERENCE.en.md)

## Model

| Value | Use | Image model | Text side | Default steps | License |
|---|---|---|---|---:|---|
| `flux3` | Latest-generation generation and editing | Official INT8 ConvRot distilled | Qwen3-VL-4B-Instruct (CPU, BF16) | 16 | FLUX Kommunity License v1.0 |

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

The official distilled INT8 ConvRot Transformer from the Comfy distribution is connected directly to the Diffusers ModularPipeline. The pipeline runs in stages: the ~5.43 GiB Transformer stays on the GPU, and the Qwen3-VL text encoder runs on the CPU. The default CUDA allocator limit is 8 GiB. ComfyUI is not used.

The model card requires a prompt upsampler (short prompts give generic output, and a one-line edit instruction can change the whole composition). Instead of the 18.8 GB official upsampler, this app runs an LLM with BFL's official system prompts (t2i / i2i / t2i_bbox / i2i_bbox). **The prompt upsampler is on by default**: every generation first expands the input into a full caption.
For comparison, you can generate without it (bypass): uncheck "prompt upsamplerを使う" (use the prompt upsampler) in the GUI, or send `upsample=false` to the API.
To review the caption before generating, use the "prompt upsamplerで展開して確認" (expand and review) button in the GUI or `POST /api/enhance-prompt` (generating with the reviewed caption unchanged does not expand it again). For edits, the reference images are shown to the LLM as well.

LLMs are tried in this order. If none is available, the original prompt is used and the reason is returned as a warning (with `upsample_strict=true`, the job fails instead).

1. LLM servers: `KLEIN_PROMPT_LLM_URLS` (comma-separated OpenAI-compatible URLs, tried in order, e.g. `http://127.0.0.1:8080/v1`. If unset, no LLM server is used)
2. Local gemma-4-E4B (llama.cpp `llama-server` with an mmproj for image input)
   - If free VRAM is at least `KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB` (default 8), it runs on the GPU and is stopped right after use
   - Otherwise, it runs on the CPU when free RAM is at least `KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB` (default 10) and the CPU is not busy
   - On the CPU, a CPU-optimized `llama-server` build is used (`KLEIN_LOCAL_LLM_SERVER_CPU`, default `~/llama.cpp/build-cpu/bin/llama-server`).
     A GPU-oriented build (without `GGML_NATIVE`/`AVX2`) runs at only about 2 tokens/s on the CPU, so one expansion takes several minutes.
     Build the CPU version with `cmake -B build-cpu -DGGML_NATIVE=ON -DGGML_CUDA=OFF && cmake --build build-cpu --target llama-server`
     (about 10 tokens/s on an i5-12400F, 20–40 s per expansion)
   - Related settings: `KLEIN_LOCAL_LLM_SERVER` (GPU build), `KLEIN_LOCAL_LLM_MODEL`, `KLEIN_LOCAL_LLM_MMPROJ`, `KLEIN_LOCAL_LLM_PORT` (default 7865); `KLEIN_LOCAL_LLM=0` disables it

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
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorch GPU allocation limit (GiB) |
| `KLEIN_PROMPT_LLM_URLS` | | LLM servers for the prompt upsampler (OpenAI-compatible, comma-separated) |

```bash
KLEIN_PROMPT_LLM_URLS=http://127.0.0.1:8080/v1 CUDA_VISIBLE_DEVICES=0 ./run.sh
```

See the [API reference](docs/API_REFERENCE.en.md#environment-variables) for the other environment variables.

### 3. Start the web app (GUI)

The web app is served by the server from step 2 on the same port, so nothing else needs to be started. Open these in a browser:

- Web app: http://127.0.0.1:7864/
- API UI (Swagger UI): http://127.0.0.1:7864/docs

By default the server listens only on `127.0.0.1`, so it can be opened only from a browser on the same machine.
To use it from another machine, use SSH port forwarding (`ssh -L 7864:127.0.0.1:7864 <server>`) or start it with `KLEIN_HOST=0.0.0.0`.
The latter has no authentication, so use it only on a trusted network.

### Data sent externally

Image generation runs entirely on local files. Because the prompt upsampler is on by default, if `KLEIN_PROMPT_LLM_URLS` is set, the prompt and (for edits) the reference images are sent to those LLM servers on every generation.
If `KLEIN_PROMPT_LLM_URLS` is not set, no LLM server is used and expansion runs only on the local LLM (llama.cpp), so nothing is sent externally. Nothing is sent when the upsampler is bypassed (`upsample=false`) either.

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
| [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) (only `system_prompts.json` is downloaded) | Apache License 2.0 |
| [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) (for comparison; not used by the app) | FLUX Kommunity License v1.0 |

The official notice for FLUX 3 Image Klein (`NOTICE.md` in the model repository), reproduced verbatim:

> This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

This app is not an official Black Forest Labs product and is not endorsed by Black Forest Labs.
See [Third-party models and licenses](THIRD_PARTY_NOTICES.md) for details. The latest original license texts at the linked pages always take precedence.
