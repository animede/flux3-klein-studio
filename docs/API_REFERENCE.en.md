# API Reference

[日本語](API_REFERENCE.md) | English

A complete list of the endpoints, parameters, responses, and errors of the FLUX 3 Image Klein Reference Studio HTTP API.
For the overall workflow and usage examples, see the [API specification (guide)](API.en.md).

- API version: `0.3.0` (`info.version` in `/openapi.json`)
- Base URL: `http://127.0.0.1:7864` (change with `KLEIN_HOST` / `KLEIN_PORT`)
- Authentication: none. By default the server listens on `127.0.0.1` only. If you expose it externally, add authentication and access control with a reverse proxy or similar.
- Machine-readable spec: `GET /openapi.json` (OpenAPI 3.1), `GET /docs` (Swagger UI), `GET /redoc` (ReDoc)
- Encoding: UTF-8. Error messages are in Japanese.

## Contents

| Method | Path | Summary | Success |
|---|---|---|---|
| `GET` | [`/api/health`](#get-apihealth) | Server status | `200` |
| `GET` | [`/api/models`](#get-apimodels) | List of available models | `200` |
| `POST` | [`/api/enhance-prompt`](#post-apienhance-prompt) | Expand a short request into a full caption (prompt upsampler) | `200` |
| `POST` | [`/api/generate`](#post-apigenerate) | Start an image generation job | `202` |
| `GET` | [`/api/jobs/{job_id}`](#get-apijobsjob_id) | Job status | `200` |
| `POST` | [`/api/jobs/{job_id}/cancel`](#post-apijobsjob_idcancel) | Request a job to stop | `200` |
| `POST` | [`/api/unload`](#post-apiunload) | Release the model from memory | `200` |
| `GET` | [`/outputs/{file}`](#get-outputsfile) | Generated PNG and metadata JSON | `200` |
| `GET` | `/` | Web UI | `200` |

## Basic flow

1. Create a job with `POST /api/generate` (returns `202` and an `id`)
2. Poll `GET /api/jobs/{id}` about once per second until `status` becomes `done`, `error`, or `cancelled`
3. If `done`, fetch `image_url` (PNG) and `metadata_url` (JSON)

Only one generation runs at a time. Sending another `POST /api/generate` while one is running returns `409`.

---

## `GET /api/health`

Returns the server status, the loaded model, and whether model files are ready.

### Response `200`: `HealthResponse`

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` |
| `loaded` | boolean | Whether a model is loaded in memory |
| `model` | string | Hugging Face repository of the loaded model (the default if none is loaded) |
| `active_model` | string \| null | Selection value of the loaded model. `null` if none is loaded |
| `default_model` | string | Selection value of the default model (`"flux3"`) |
| `models` | object | Selection value → [`ModelInfo`](#modelinfo) |
| `busy` | boolean | Whether a generation is in progress |
| `memory_budget_gib` | number | PyTorch GPU allocation limit (GiB) |

```json
{
  "status": "ok",
  "loaded": false,
  "model": "black-forest-labs/flux-3-image-klein",
  "active_model": null,
  "default_model": "flux3",
  "models": {
    "flux3": {
      "name": "FLUX 3 Image Klein",
      "description": "最新世代・蒸留済み推論モデル",
      "license": "FLUX Kommunity License v1.0",
      "quantization": "Official INT8 ConvRot",
      "available": true
    }
  },
  "busy": false,
  "memory_budget_gib": 8.0
}
```

---

## `GET /api/models`

### Response `200`: `ModelsResponse`

| Field | Type | Description |
|---|---|---|
| `default` | string | Selection value of the default model |
| `active` | string \| null | Selection value of the loaded model |
| `models` | object | Selection value → [`ModelInfo`](#modelinfo) |

### `ModelInfo`

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name |
| `description` | string | Description |
| `license` | string | Model license name |
| `quantization` | string | Transformer quantization |
| `available` | boolean | Whether the required files are present. If `false`, fetch them with `download_models.py flux3 --acknowledge-flux3-license` |

---

## `POST /api/enhance-prompt`

Rewrites a short request into the full scene caption that FLUX 3 Image Klein expects.
It passes BFL's official system prompts (`system_prompts.json` from `flux-3-image-klein-pup`) to an LLM.
It does not generate an image. `/api/generate` performs the same expansion by default before generating, so use this endpoint to review or edit the expanded caption before generation.
When generating with a reviewed caption, add `upsample=false` to `/api/generate` to avoid expanding it twice.

### Request: `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | Yes | | Short request. Whitespace only returns `422` |
| `mode` | string | No | `auto` | `auto` / `t2i` / `i2i` / `t2i_bbox` / `i2i_bbox`. `auto` resolves to `i2i` if reference images are given, otherwise `t2i` |
| `image` | file (repeatable) | No | | Reference images for editing. Repeat the field. Only the first 8 are used; the 9th and later are ignored. Each up to 10MB and 16 megapixels |
| `width` | integer | No | | For `t2i` modes, tells the LLM the output aspect ratio (only when `height` is also given) |
| `height` | integer | No | | Same as above |

- Reference images have transparent areas filled with white, are downscaled to at most 1024px on the long side, and are passed to the LLM before the request text.
- For `i2i` modes without reference images, only the request text is passed.

### LLM selection order

1. **LLM servers**: tries the OpenAI-compatible URLs in `KLEIN_PROMPT_LLM_URLS` in the order listed. Not used if unset.
   A server is skipped if its `/models` does not respond. When reference images are present, servers whose `/models` `capabilities` show no image input support are also skipped.
2. **Local gemma-4-E4B** (llama.cpp `llama-server`): starts on the GPU if there is enough free VRAM; otherwise on the CPU if there is enough free memory and CPU headroom.
   On the GPU it stops right after use; on the CPU it stops after a period of inactivity. Disable with `KLEIN_LOCAL_LLM=0`.

If none is available, returns `503`.

### Response `200`

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Expanded caption. Can be used as-is for `prompt` in `/api/generate` |
| `mode` | string | Mode actually used (`auto` resolved) |
| `source` | string | LLM used: one of `remote:<URL>`, `local-gpu`, `local-cpu` |
| `model` | string \| null | LLM model name. `null` if the LLM server returns no name; the GGUF file name for the local LLM |
| `cached` | boolean | Whether the result came from the cache. The latest 128 results are cached, keyed by mode, request text, width/height, and reference images |

```json
{
  "prompt": "Create a full-body front view photograph of the woman from the first image, preserving her identity ...",
  "mode": "i2i",
  "source": "remote:http://127.0.0.1:8080/v1",
  "model": "gemma-4-31B",
  "cached": false
}
```

### Errors

| Code | Cause |
|---|---|
| `413` | A reference image exceeds 10MB |
| `422` | `prompt` is empty, `mode` is invalid, a reference image cannot be read as an image, or exceeds 16 megapixels |
| `503` | No LLM is available. `detail` contains why each candidate could not be used |

### Example

```bash
curl -X POST http://127.0.0.1:7864/api/enhance-prompt \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png'
```

---

## `POST /api/generate`

Creates an image generation job and returns `202` immediately. Generation runs in the background.

### Request: `multipart/form-data`

| Field | Type | Required | Default | Constraints / description |
|---|---|---|---|---|
| `prompt` | string | Yes | | 1–4000 characters after trimming whitespace. By default the prompt upsampler expands it, so a short request is fine. With `upsample=false`, pass a full scene caption |
| `image` | file (repeatable) | No | | Reference images. Repeat the field up to 8 times. Each up to 10MB and 16 megapixels. PNG / JPEG / WebP, etc. |
| `width` | integer | No | `768` | 256–1024, multiple of 32 |
| `height` | integer | No | `768` | 256–1024, multiple of 32 |
| `seed` | integer | No | `-1` | `-1` is random. To fix it, 0–4294967295 |
| `model` | string | No | `flux3` | `flux3` (case-insensitive) |
| `outpaint` | boolean | No | `false` | `true` uses the outpainting LoRA. Requires exactly 1 reference image |
| `boxes` | string (JSON) | No | `""` | Placement. See [`boxes` format](#boxes-format) |
| `upsample` | boolean | No | `true` | Prompt upsampler: expands the prompt before generation, using the same process as `/api/enhance-prompt`. `false` bypasses it (for comparison, or when sending an already-expanded caption) |
| `upsample_strict` | boolean | No | `false` | If `true`, the job fails with `error` when expansion is not possible. If `false`, it generates with the original prompt and sets `warning` |

Send booleans as `true` / `false` (`1` / `0`, `yes` / `no`, `on` / `off` are also accepted).

**Reference image handling**

- No reference image means T2I, one means editing, and two or more means multi-reference composition.
- Transparent areas are filled with white and EXIF orientation is applied.
- To limit VRAM, images are downscaled to at most `reference_max_side` on the long side. This is 768px for one image and shrinks as more are added so the total pixel count fits within 768² × 8 (minimum 256px, multiple of 32).
- For outpainting, the reference image is downscaled to fit the output size, centered, and the surroundings are filled in. The center stays as the original image.

**Prompt assembly order**: expand with `upsample` → append `boxes` → generate.

### `boxes` format

A JSON array (1–32 items). Coordinates are `[top, left, bottom, right]` normalized to 0–1000, with `top <= bottom` and `left <= right`.
Reference each item's `id` in the caption as `<id>`.

**Placement (T2I)**

| Key | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Matches `<id>` in the caption |
| `bbox` | number[4] | Yes | Where to place it |
| `desc` | string | No | Description of the element |

**Editing** (the n-th reference is referred to as `<ref_image_n>`)

| Key | Type | Required | Description |
|---|---|---|---|
| `id` | string | Yes | Matches `<id>` in the caption |
| `from` | string | No | Which reference to take it from (e.g. `"ref_image_1"`) |
| `src_bbox` | number[4] \| null | One of the two | Original position |
| `tgt_bbox` | number[4] \| null | One of the two | Destination. `null` removes the element |
| `desc` | string | No | Description of the element |

`src_bbox` and `tgt_bbox` cannot both be `null`. The server appends this array after the caption, separated by a single space (the model card format).

```json
[{"id": "apple_1", "bbox": [550, 100, 900, 400], "desc": "A shiny red apple with a green leaf."}]
```

### Response `202`: [`JobResponse`](#jobresponse)

`status` is `queued`, and `seed` is the resolved value (the chosen random value if `-1` was sent).

### Errors

| Code | Cause |
|---|---|
| `409` | Another job is running, or the model or outpainting LoRA has not been downloaded |
| `413` | A reference image exceeds 10MB |
| `422` | Input out of range (`prompt`, `width` / `height`, `seed`, `model`, `boxes`), 9 or more reference images, `outpaint=true` without exactly 1 reference image, or an unreadable image |

### Example

```bash
# Expand a short request, then edit (the prompt upsampler is on by default)
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=768' -F 'height=1024' -F 'seed=42'

# For comparison: same request and seed with the prompt upsampler bypassed
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'upsample=false' \
  -F 'width=768' -F 'height=1024' -F 'seed=42'
```

---

## `GET /api/jobs/{job_id}`

### Path parameters

| Name | Description |
|---|---|
| `job_id` | The `id` returned by `POST /api/generate` (32 hex digits) |

### Response `200`: [`JobResponse`](#jobresponse)

### Errors

| Code | Cause |
|---|---|
| `404` | Job does not exist (after a server restart, or an old job) |

---

## `POST /api/jobs/{job_id}/cancel`

Requests a job to stop. During generation it stops at the next step boundary (usually within 1–2 seconds) and `status` becomes `cancelled`.
While loading the model or expanding with the LLM, it stops after that work finishes. A stopped job releases the model.

For a job already in `done` / `error` / `cancelled`, nothing happens and the current state is returned.

### Response `200`: [`JobResponse`](#jobresponse)

The state at the time the stop request was accepted. `message` becomes "停止を受け付けました…" (stop request accepted). Check completion with `GET /api/jobs/{job_id}`.

### Errors

| Code | Cause |
|---|---|
| `404` | Job does not exist |

---

## `POST /api/unload`

Releases the loaded model from CPU and GPU memory. It is reloaded on the next generation (a few seconds).
Use this when sharing the GPU with other applications.

### Response `200`

```json
{"status": "ok", "loaded": false}
```

### Errors

| Code | Cause |
|---|---|
| `409` | Cannot release during generation |

---

## `GET /outputs/{file}`

The files pointed to by `image_url` (`/outputs/{job_id}.png`) and `metadata_url` (`/outputs/{job_id}.json`) in `JobResponse`.
URLs are relative to the base URL. Files are not deleted automatically.

### Metadata JSON

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Prompt actually passed to the model (after expansion and `boxes` appending) |
| `original_prompt` | string \| null | Original prompt when expansion or `boxes` changed it. `null` if unchanged |
| `upsample_source` | string \| null | LLM used for expansion |
| `boxes` | array \| null | The `boxes` that were sent |
| `seed` / `width` / `height` | integer | Generation settings |
| `references` | string[] | Paths of saved reference images (`uploads/{job_id}_ref_{n}.png`) |
| `reference` | string \| null | First reference image (legacy API compatibility) |
| `reference_count` | integer | Number of reference images |
| `reference_max_side` | integer \| null | Long-side limit for reference images |
| `model` / `model_name` / `model_repo` | string | Model |
| `base_revision` / `int8_revision` | string | Revisions of the model files used |
| `transformer_quantization` / `text_quantization` | string | Quantization (`Official INT8 ConvRot` / `BF16 (CPU)`) |
| Others | | All fields of [`metrics`](#metrics) |

---

## Data types

### `JobResponse`

| Field | Type | Description |
|---|---|---|
| `id` | string | Job ID |
| `status` | string | [Job status](#job-status) |
| `step` | integer | Steps completed (0–`steps`) |
| `steps` | integer | Total steps (`16`) |
| `model` | string | Model selection value |
| `model_name` | string | Model display name |
| `seed` | integer | Seed used |
| `message` | string | Status description. For `error`, the failure reason (up to 1000 characters) |
| `created_at` | number | Creation time (UNIX seconds) |
| `reference_count` | integer | Number of reference images |
| `reference_max_side` | integer \| null | Long-side limit for reference images. `null` if there are none |
| `metrics` | object \| null | Measurements on completion ([`metrics`](#metrics)) |
| `image_url` | string \| null | PNG URL on completion |
| `metadata_url` | string \| null | Metadata JSON URL on completion |
| `upsample` | boolean | Whether the job uses the prompt upsampler (`false` with `upsample=false`) |
| `upsample_source` | string \| null | LLM used for expansion |
| `warning` | string \| null | Warning when expansion failed and the original prompt was used |

### Job status

| `status` | Meaning | Terminal |
|---|---|---|
| `queued` | Accepted | |
| `upsampling` | Expanding the prompt with the LLM | |
| `switching` | Releasing the previous model to switch configuration (e.g. when outpainting is toggled) | |
| `loading` | Loading the model | |
| `encoding` | Encoding the prompt and reference images | |
| `generating` | Denoising. `step` advances by 1 | |
| `done` | Completed | ✓ |
| `error` | Failed. Reason in `message` | ✓ |
| `cancelled` | Stopped | ✓ |

### `metrics`

| Field | Type | Description |
|---|---|---|
| `model` / `model_name` | string | Model |
| `seconds` | number | Seconds spent generating (encode through decode) |
| `total_seconds` | number | Seconds including model loading (excluding expansion time) |
| `peak_allocated_gib` | number | Peak GPU memory allocated by PyTorch |
| `peak_reserved_gib` | number | Peak GPU memory reserved by PyTorch |
| `allocator_budget_gib` | number | Allocation limit |
| `gpu` | string | GPU name |
| `steps` | integer | Number of steps |
| `transformer_calls` | integer | Number of Transformer calls (equals `steps` for the distilled model without CFG) |
| `guidance` | number \| null | CFG value (`null` for the distilled model) |

### Error shape

Error messages are in Japanese.

```json
{"detail": "幅・高さは256〜1024の32の倍数で指定してください。"}
```

When FastAPI's own type validation fails (missing required fields, non-integer values, etc.), the response is `422` and `detail` is an array.

```json
{"detail": [{"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": null}]}
```

---

## Environment variables

Read at server startup (also when started via `run.sh`).

**Server and generation**

| Variable | Default | Description |
|---|---|---|
| `KLEIN_HOST` | `127.0.0.1` | Listen address (`run.sh`) |
| `KLEIN_PORT` | `7864` | Port (`run.sh`) |
| `KLEIN_DEVICE` | `cuda:0` | GPU to use |
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorch GPU allocation limit (GiB) |

**Prompt upsampler (LLM servers)**

| Variable | Default | Description |
|---|---|---|
| `KLEIN_PROMPT_LLM_URLS` | None | Comma-separated OpenAI-compatible URLs (`.../v1`). Tried in the order listed |
| `KLEIN_PROMPT_LLM_URL` | None | Legacy single-URL form (`KLEIN_PROMPT_LLM_URLS` takes precedence) |
| `KLEIN_PROMPT_LLM_MODEL` | None | Name sent as `model` to the first LLM server. If unset, or for the second and later servers, the first entry of each server's `/models` |
| `KLEIN_PROMPT_LLM_TIMEOUT_S` | `120` | Timeout per generation (seconds) |
| `KLEIN_PROMPT_LLM_PROBE_TIMEOUT_S` | `3` | Timeout for the `/models` check (seconds) |

**Prompt upsampler (local LLM)**

| Variable | Default | Description |
|---|---|---|
| `KLEIN_LOCAL_LLM` | `1` | `0` disables the local LLM |
| `KLEIN_LOCAL_LLM_SERVER` | `~/llama.cpp/build/bin/llama-server` | llama-server for GPU |
| `KLEIN_LOCAL_LLM_SERVER_CPU` | `~/llama.cpp/build-cpu/bin/llama-server` | llama-server for CPU (falls back to the GPU one if missing) |
| `KLEIN_LOCAL_LLM_MODEL` | `~/models/gemma-4-E4B-it-Q4_K_M.gguf` | GGUF model |
| `KLEIN_LOCAL_LLM_MMPROJ` | Auto-detected | mmproj for image input |
| `KLEIN_LOCAL_LLM_PORT` | `7865` | llama-server port (`127.0.0.1` only) |
| `KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB` | `8` | Free VRAM required to run on the GPU (GiB) |
| `KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB` | `10` | Free memory required to run on the CPU (GiB) |
| `KLEIN_LOCAL_LLM_CPU_MAX_LOAD` | `0.75` | Maximum load for running on the CPU (1-minute average ÷ core count) |
| `KLEIN_LOCAL_LLM_START_TIMEOUT_S` | `180` | Startup wait timeout (seconds) |
| `KLEIN_LOCAL_LLM_TIMEOUT_S` | `300` | Timeout per generation (seconds) |
| `KLEIN_LOCAL_LLM_IDLE_S` | `600` | Seconds of inactivity before stopping when running on the CPU |

---

## Limits and retention

| Item | Value |
|---|---|
| Concurrent generations | 1 |
| Prompt | 1–4000 characters |
| Reference images | Up to 8, each up to 10MB and 16 megapixels |
| Output size | 256–1024px, multiple of 32 |
| Steps | 16 (fixed) |
| `boxes` | Up to 32 items |
| Jobs kept in memory | Latest 100 (cleared on restart) |
| Upsampler cache | Latest 128 (cleared on restart) |
| Storage | Generated images and metadata in `outputs/`, reference images in `uploads/`. No automatic deletion |

## Where data is sent

Image generation runs entirely locally. Data leaves the machine only when prompt expansion (`/api/enhance-prompt`, or the default `upsample=true`)
uses an LLM server, in which case the request text and (for editing) reference images are sent to the destinations in `KLEIN_PROMPT_LLM_URLS`.
Nothing is sent externally when the local LLM is used.
