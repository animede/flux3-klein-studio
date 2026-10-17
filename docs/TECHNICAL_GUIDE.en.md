# Technical Guide

[日本語](TECHNICAL_GUIDE.md) | English

This guide summarizes the techniques FLUX 3 Image Klein Reference Studio uses to run FLUX 3 Image Klein with Diffusers in 8GiB of VRAM.
For how to use the API, see the [API guide](API.en.md); for every field, see the [API reference](API_REFERENCE.en.md).

Unless noted otherwise, numbers were measured on an RTX PRO 4000 Blackwell (24GB) with a Core i5-12400F, INT8, 16 steps, and a timestep shift of 16 (2026-10-18).

## Contents

1. [Overall structure](#1-overall-structure)
2. [Loading the INT8 ConvRot transformer into Diffusers](#2-loading-the-int8-convrot-transformer-into-diffusers)
3. [Staged pipeline](#3-staged-pipeline)
4. [Memory design](#4-memory-design)
5. [Sampling settings](#5-sampling-settings)
6. [Progress reporting and cancellation](#6-progress-reporting-and-cancellation)
7. [Outpainting and positioning](#7-outpainting-and-positioning)
8. [Prompt upsampler](#8-prompt-upsampler)
9. [Running quantized versions of the official upsampler locally](#9-running-quantized-versions-of-the-official-upsampler-locally)
10. [Server and GUI](#10-server-and-gui)
11. [Limitations and caveats](#11-limitations-and-caveats)

---

## 1. Overall structure

```text
Request (+ reference images)
   │
   ▼
prompt upsampler ── LLM server / local gemma-4-E4B / official upsampler (INT8, GGUF)
   │  full scene caption
   ▼
Text encoder Qwen3-VL-4B (bf16, CPU memory → computed on GPU one layer at a time)
   │  prompt_embeds
   ▼
VAE encode (reference images, GPU)
   │
   ▼
Denoising Transformer 5.6B (official INT8 ConvRot, resident on GPU) × 16 steps
   │
   ▼
VAE decode (GPU, tiled above 1024px) → PNG
```

| Component | Format | Location |
|---|---|---|
| Transformer | BFL's official INT8 ConvRot (ComfyUI distribution, about 5.4GB) | Resident on GPU |
| Text encoder | Qwen3-VL-4B, bf16 (about 8GB) | CPU memory; sent to GPU one layer at a time during compute |
| VAE | FLUX.2 VAE, bf16 | GPU |
| prompt upsampler | External LLM (OpenAI-compatible API) or a quantized version of the official upsampler | Separate process |

A single FastAPI server serves both the API (`/api/*`) and the GUI (`/`) on the same port. Generation runs one at a time as jobs.

## 2. Loading the INT8 ConvRot transformer into Diffusers

BFL distributes the INT8 version of FLUX 3 Image Klein as a single file for ComfyUI
(`diffusion_models/flux3-image-klein-int8-convrot.safetensors` in `flux-3-image-klein-comfy`). Diffusers supports FLUX 3,
but not loading this format, so the app has its own loader that wires it in directly without conversion (`flux3_int8.py`).

**File contents**: For each quantized linear layer, there is `weight` (int8), `weight_scale` (float32, per row), and `comfy_quant`
(bytes encoding `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`).

**ConvRot**: The input dimension is split into groups (e.g. 256), and each group is rotated by a normalized Hadamard matrix H before quantization.
Weights are stored offline as `W_rot = W · Hᵀ` (per group); at inference, activations are rotated by the same H before the INT8 matmul.
The rotation spreads out outliers, so accuracy holds up even in INT8.

**Loading steps**

1. Build `Flux3Transformer2DModel` with empty weights (`accelerate.init_empty_weights`).
2. Replace layers that have `comfy_quant` with a custom `ConvRotInt8Linear` that calls `comfy-kitchen`'s `int8_linear(..., convrot=True)`.
   The int8 weights and float32 scales are kept as is (not dequantized).
   ComfyUI layer names are mapped to Diffusers layer names (`qkv` is split into `to_q` / `to_k` / `to_v`).
3. Load the remaining floating-point weights, renamed with Diffusers' `convert_flux3_transformer_checkpoint_to_diffusers`.
4. Check that no empty weights (meta tensors) remain.
5. Move with `.to("cuda")` (no dtype given, so the float32 scales are not downcast).

## 3. Staged pipeline

Diffusers' FLUX.3 is a **ModularPipeline**, split into text encode, VAE encode, denoise, and decode blocks.
The app builds a separate pipeline per block, places each component where it fits, and runs them in order while passing state (the `PipelineState` tensors) along.

```python
blocks = ModularPipeline.from_pretrained(model_dir)._blocks.sub_blocks
text_pipe = blocks["text_encoder"].init_pipeline(model_dir)
denoise_pipe = blocks["denoise"].init_pipeline(model_dir)
denoise_pipe.register_components(transformer=int8_transformer)   # INT8 Transformer from section 2
vae_encode_pipe = blocks["vae_encoder"].init_pipeline(model_dir)
decode_pipe = blocks["decode"].init_pipeline(model_dir)

state = text_pipe(prompt=caption)
state = move_tensors(state, "cuda")
if references:
    state = vae_encode_pipe(state=state, image=references)
state = denoise_pipe(state=state, width=w, height=h, num_inference_steps=16, generator=g)
image = decode_pipe(state=state, output="images", output_type="pil")[0]
```

The distilled model does not use CFG, so the Transformer is called only once per step.

## 4. Memory design

### 4.1 Allocation cap

`torch.cuda.set_per_process_memory_fraction` caps the GPU memory PyTorch can allocate in this process.
The default is 8GiB (`KLEIN_MEMORY_BUDGET_GIB`), chosen to run on 12GB-class cards.
With a cap, PyTorch releases its cache earlier, so the actual peak is slightly lower than without one
(1024×1024 T2I: 8.19GiB uncapped, 7.81GiB with an 8GiB cap).

### 4.2 Computing the text encoder on the GPU one layer at a time

The text encoder (Qwen3-VL-4B, bf16, about 8GB) exceeds 8GiB if placed on the GPU together with the Transformer.
Putting it entirely on the GPU alone brings the peak to 8.6GiB.

It was originally computed on the CPU, but for the 200–270-word captions the prompt upsampler writes, encoding took about 30 seconds,
80% of the generation time at 1024×1024. So it now uses accelerate's `cpu_offload` to keep the weights in CPU memory
and send them to the GPU one layer at a time during compute (sequential offload).

| | CPU compute | GPU, one layer at a time |
|---|---|---|
| Encode time (about 400 tokens) | 29.6 s | 2.2–2.6 s |
| Extra VRAM | 0 | about 0.95GiB |
| Difference in result (relative error vs. CPU result) | — | 0.7% |

`KLEIN_TEXT_ENCODER=cpu` switches back to CPU compute.

### 4.3 VAE tiling

At 2048×2048 (4MP), denoising peaks at about 8.2GiB, while the final VAE decode alone uses about 15.3GiB.
For images above 1024px, the VAE is tiled (`enable_tiling()`, 1024px tiles, 25% overlap).

| 4MP decode | No tiling | Tiling |
|---|---|---|
| Peak | 15.41GiB | 8.31GiB |
| Time | 1.43 s | 1.75 s |

Tiling changes pixels by about 8.4 on average, but the difference is spread thinly along edges and no seams are visible.
At 1024px or below, tiling does not kick in and results are unchanged.

### 4.4 Raising the cap only for 2K output

BFL recommends 2K (about 4MP) output. Even with tiling, 4MP denoising slightly exceeds 8GiB.
So only jobs whose output exceeds 1024×1024 (1,048,576 pixels) raise the cap to 10GiB (`KLEIN_HIRES_MEMORY_BUDGET_GIB`),
and the next job at 1MP or below brings it back to 8GiB (calling `torch.cuda.empty_cache()` when it does).
If the GPU's capacity is smaller than the cap, an error is raised before generation starts.

### 4.5 Reference image pixel count

Reference images add denoising tokens. BFL's inference code allows up to 1MP per image, but 8 images at 1MP each
peak at 11.2GiB. This app downscales them, keeping aspect ratio, to at most 1MP per image and 2MP total (`KLEIN_REFERENCE_TOTAL_MP`)
(1–2 images: 1MP each; 4 images: 0.5MP each; 8 images: 0.25MP each).

### 4.6 Measurements

| Task | Output | Generation time | Peak (allocated) | Cap |
|---|---|---|---|---|
| T2I | 1024×1024 | 11.7 s | 7.81GiB | 8GiB |
| Edit with 1 reference | 768×1024 | 18.1 s | 7.59GiB | 8GiB |
| 2 references | 1024×768 | 29.8 s | 7.59GiB | 8GiB |
| 8 references (0.25MP each) | 1024×768 | 28.2 s | 7.57GiB | 8GiB |
| T2I (2K) | 2048×2048 | 51.4 s | 8.22GiB | 10GiB |
| 2 references (2K) | 2304×1728 | 83.1 s | 9.50GiB | 10GiB |

(Includes text encoding; excludes prompt upsampler time. Model loading takes a few seconds.)

## 5. Sampling settings

| Setting | Value |
|---|---|
| Steps | 16 (distilled) |
| CFG | None |
| Sampler | Flow Matching Euler (`FlowMatchEulerDiscreteScheduler`) |
| Timestep shift (alpha) | 16 |

**Timestep shift**: A constant shift that maps sigma σ to `σ' = α·σ / (1 + (α − 1)·σ)`. A larger α spends more steps
in the noisy first half (where the overall composition is decided). On 2026-10-17, BFL changed the inference α from 6.93 to 16
(official inference code and ComfyUI workflows). Diffusers' FLUX.3 still uses 6.93, so the app calls
`scheduler.set_shift(16.0)` on every generation to match (`KLEIN_TIMESTEP_SHIFT`, API `shift`).

At 1024×1024, there was no clear quality difference between 6.93 and 16 (same composition and content; only details change).
The shift matters at high resolution, so using the official value at 2K is the safe choice. Also, in one editing example, a caption whose
composition broke down at 6.93 did not break down at 16 (a single observation).

## 6. Progress reporting and cancellation

Running block by block leaves no per-step callback for denoising. So the app attaches a
`register_forward_pre_hook` to the Transformer and counts its calls as steps. The distilled model has no CFG, so it is
called once per step (exactly 16 calls for 16 steps).

Cancel requests are handled by raising an exception (`Cancelled`) inside this hook. Generation stops at a step boundary (about every 0.5 s at 1MP),
and a cancelled job releases the model.

## 7. Outpainting and positioning

### 7.1 Outpainting

The app uses a community outpainting LoRA (`ausboss/FLUX-3-Image-Klein-Outpaint-LoRA`).

- **Applying the LoRA**: INT8 layers cannot have their weights rewritten, so `ConvRotLoRALinear` adds a bf16 low-rank correction
  `x · Aᵀ · Bᵀ × strength (1.5)` to the INT8 layer's output. When the LoRA is used, the model is reloaded with the LoRA attached.
- **Input**: The reference image is downscaled to fit the output size, centered on a gray (128) margin, and passed as the reference.
- **Output**: After generation, the original image is pasted back in the center (to keep the original pixels).

### 7.2 Positioning (bounding boxes)

Following the model card's format, a JSON array of elements is appended after the caption, separated by a single space. Coordinates are
`[top, left, bottom, right]` in 0–1000, and each element is referenced in the caption as `<id>`. Edits use `from`, `src_bbox`, and `tgt_bbox`
(`null` to delete). The array is appended after prompt upsampler expansion, so the upsampler never rewrites the boxes.

## 8. Prompt upsampler

### 8.1 Why it is required

FLUX 3 Image Klein is trained on detailed scene captions. According to the model card, short prompts produce mediocre output,
and one-line edit instructions end up changing the composition too. Reference images pin down content but not composition.
So an LLM that rewrites short requests into full captions (the prompt upsampler) sits in front. In this app it is **on by default**,
and for comparison you can also generate without it (bypass, `upsample=false`).

### 8.2 LLMs used

| `upsampler` | LLM used | System prompt |
|---|---|---|
| `auto` (default) | LLM server (`KLEIN_PROMPT_LLM_URLS`) → local gemma-4-E4B | For external LLMs |
| `remote` / `local` | One or the other | For external LLMs |
| `pup-int8` / `pup-gguf` | Quantized official upsampler (section 9) | For the official upsampler |

### 8.3 Input for the official upsampler (`pup-*`)

This matches the official inference code (`src/flux3/pup.py`).

- System prompt: `system_prompts.json` from `flux-3-image-klein-pup` (t2i / i2i / t2i_bbox / i2i_bbox)
- Edits: reference images first, followed by a three-line request
  ```text
  Reference images follow in order (the first image, the second image, ...).
  User request: <request>
  Output the caption.
  ```
- Reference images: transparency filled with white, downscaled with LANCZOS to about 1MP for one image, 724² for two, and 512² each for three or more, then encoded as JPEG (quality 92)
- Decoding: greedy (temperature 0), presence penalty 1.5 for T2I and 0 for edits, up to 3584 tokens, thinking off

### 8.4 Input for external LLMs (`auto` / `remote` / `local`)

On 2026-10-17, BFL published system prompts for LLMs other than the official upsampler (in the official inference code,
`src/flux3/pup_prompts_external_vlm.json`, Apache-2.0). The official upsampler has learned the conventions, so a short prompt suffices,
but external LLMs need the conventions spelled out in detail; the T2I prompt is about 7,400 words (about 10K tokens).

- Edits put the request (`User request: <request>`) first and the reference images after it (the reverse of the official upsampler input).
- T2I prepends a `TARGET CANVAS: width:height` line to convey the output aspect ratio (a form both system prompts accept).
- The LLM needs a context length of 16K or more. The llama.cpp server reuses the processed system prompt,
  so later calls are faster (gemma-4-31B: 20 s first → 13–16 s; E4B: 8.7 s → 3.5–4 s).
- `KLEIN_LLM_PROMPTS=pup` makes external LLMs use the official upsampler prompts too (the behavior before 2026-10-18).

### 8.5 Failure handling and caching

- If no LLM is available, generation proceeds with the original text and the job gets a `warning`. With `upsample_strict=true`, it fails instead.
- Results are cached for the latest 128 entries, keyed by a hash of the upsampler, mode, request, width/height, and reference images.
- If a caption made with the GUI's "Expand and review" button (「展開して確認」) is generated without edits, it is sent with `upsample=false` to avoid expanding it twice.

### 8.6 Findings from comparisons

In a comparison of 6 tasks × 2 seeds, the difference in raw image quality was small; the differences showed up in the following:

- The official upsampler describes reference images as "an unchanged base" and enumerates even windows and screens, so it preserves the reference composition well.
- Both the official upsampler and gemma-4 with the external-LLM prompt sometimes add text not in the request (venue names, times).
  Take care when exact text matters.

## 9. Running quantized versions of the official upsampler locally

The official upsampler (`flux-3-image-klein-pup`, based on Qwen3.5-9B) is 18.8GB in bf16 and does not fit alongside FLUX 3.
A quantized version is started on the GPU only when needed and stopped right after use, returning VRAM to the image model.
If free VRAM is insufficient, the image model is temporarily released before starting it (always within a generation job; for `/api/enhance-prompt`, only when no generation is running).

| | `pup-int8` | `pup-gguf` |
|---|---|---|
| File | BFL's official `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot` (9.8GB) | Community Q4_K_M + mmproj (6.5GB) |
| Runtime | `pup_server.py` (Transformers + comfy-kitchen) | llama.cpp llama-server |
| VRAM | about 9.4GiB | about 6.8GiB |
| Time per call (including startup) | about 19 s | about 8 s |
| Word agreement with bf16 | 0.28–0.75 | 0.28–0.66 |

(Agreement is the match rate of word sequences in captions for 6 tasks. Between gemma-4-31B and bf16 it is 0.14–0.23.)

### 9.1 Loading the INT8 version (`pup_int8.py`)

The file targets ComfyUI, but its tensor names follow the same layout as Transformers' `Qwen3_5ForConditionalGeneration`.

- Linear layers: replaced with the same `ConvRotInt8Linear` as in section 2. Layers on the image-input side have biases, and the rotation group size
  varies by layer (16 / 64 / 256), so each is read from the config.
- Token embedding layer: also INT8, and stored in rotated form (`W_rot = W · Hᵀ`). The normalized Hadamard matrix is
  symmetric and orthogonal, so `W = W_rot · H` recovers the original. Restoring the whole table (248,320 rows) is large, so
  `ConvRotInt8Embedding` dequantizes and unrotates only the looked-up rows on the fly.
- The MTP head (`mtp.*`) for speculative decoding is not used.

### 9.2 Why a separate process

The image model's process has an 8GiB allocation cap (section 4.1), and a 10GiB-class upsampler does not fit within it.
`pup_server.py` is a small server with only an OpenAI-compatible `/v1/chat/completions`, launched by upsampler.py as a child process.
Transformers has no presence penalty, so it implements the same "subtract once from already-generated tokens" logic as vLLM.

### 9.3 Cleaning up child processes

llama-server and `pup_server.py` set `prctl(PR_SET_PDEATHSIG, SIGTERM)` at startup so they do not linger as orphans if the studio crashes.
The local gemma-4-E4B runs on the CPU if free VRAM is insufficient, and on the CPU it is stopped after a few minutes of disuse.
On the CPU, a llama-server built for GPU (without `GGML_NATIVE` / `AVX2`) is extremely slow, so a separate CPU build is used.

## 10. Server and GUI

- **Jobs**: `POST /api/generate` creates a job and returns immediately; generation runs one at a time on a single-thread worker.
  State for the latest 100 jobs is kept in memory, and images and metadata JSON are saved to `outputs/`.
- **Metadata**: Records the prompt actually passed to the model, the original request, the upsampler and LLM used, the seed, the shift,
  where the text encoder ran, and time and peak VRAM.
- **Gallery**: `GET /api/gallery` returns images in `outputs/` newest first, and `DELETE /api/gallery/{id}` deletes the PNG, metadata, and reference images
  (only 32-digit hex IDs are accepted, and jobs in progress are not deleted).
- **GUI**: The input panel on the left is always shown, and the right side switches between "Results" (「生成結果」) and "Gallery" (「ギャラリー」). Dragging and dropping an image from
  the results or gallery onto the left reference images fetches it from the server and adds it as a reference. GUI files are served with `Cache-Control: no-cache`
  so stale screens do not persist after updates.

## 11. Limitations and caveats

- Only tested on the Blackwell generation (RTX PRO 4000 / 5000). INT8 compute relies on comfy-kitchen's ConvRot kernels, so
  whether earlier generations work depends on that kernel's support. A CUDA 13-capable environment is required.
- Operation on 12GB cards (8GiB cap for 1MP, 10GiB for 2K) is based on calculation and has not been tested on real hardware.
- About 32GB of system RAM is needed to hold the text encoder.
- Reference images are downscaled to 2MP total (section 4.5), so fine details may be redrawn.

## References

- FLUX 3 Image Klein: https://huggingface.co/black-forest-labs/flux-3-image-klein
- ComfyUI distribution (INT8 version): https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
- Official upsampler: https://huggingface.co/black-forest-labs/flux-3-image-klein-pup
- Official inference code (`pup.py`, system prompts for external LLMs): https://github.com/black-forest-labs/flux-3-image-klein
- Outpainting LoRA: https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA
