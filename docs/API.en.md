# API Specification

[日本語](API.md) | English

FLUX 3 Image Klein Reference Studio (Diffusers server and GUI app) provides a job-based HTTP API for FLUX 3. The default base URL is `http://127.0.0.1:7864`.

- Inputs, outputs, errors, and environment variables for every endpoint: [API Reference](API_REFERENCE.en.md)
- Techniques used (INT8 loading, memory design, upsampler): [Technical Guide](TECHNICAL_GUIDE.en.md)
- Eight-view character sheets: [Character Sheet API](CHARSHEET.en.md)
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI 3.1 JSON: `/openapi.json`
- Authentication: none
- Concurrent generations: 1
- Output format: PNG

By default the server listens only on `127.0.0.1`. If you expose it externally, set up authentication and access restrictions separately.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Get service status, the loaded model, and model readiness |
| `GET` | `/api/models` | Get model names, quantization, licenses, and readiness |
| `POST` | `/api/enhance-prompt` | Expand a short request into a full FLUX 3 caption (prompt upsampler) |
| `POST` | `/api/generate` | Start a generation job |
| `GET` | `/api/jobs/{job_id}` | Get job status |
| `POST` | `/api/jobs/{job_id}/cancel` | Request that a job stop |
| `POST` | `/api/unload` | Release the current model from CPU/GPU memory |
| `GET` | `/api/gallery` | List generated images (newest first) |
| `DELETE` | `/api/gallery/{id}` | Delete a generated image (with its metadata and reference images) |
| `POST` | `/api/charsheet/generate` | Start an eight-view character sheet ([Character Sheet API](CHARSHEET.en.md)) |
| `GET` | `/api/charsheet/jobs/{job_id}` | Character sheet status (image URL per direction, sheet, ZIP) |
| `POST` | `/api/charsheet/jobs/{job_id}/cancel` | Request cancellation of a character sheet |

## Image Generation

### `POST /api/generate`

The request is `multipart/form-data`.

| Field | Required | Default | Constraints |
|---|---:|---:|---|
| `prompt` | Yes | None | 1–4000 characters |
| `model` | No | `flux3` | `flux3` |
| `image` | No | None | Repeat the field up to 8 times. Each at most 10 MB and 16 megapixels |
| `width` | No | `768` | 256–2688, multiple of 32. Width × height at most 2048×2048. Above 1024×1024 the job counts as 2K and the GPU allocation limit becomes 10 GiB |
| `height` | No | `768` | 256–2688, multiple of 32 |
| `seed` | No | `-1` | `-1` is random; fixed values are 0–4294967295 |
| `outpaint` | No | `false` | `true` uses the outpaint LoRA. Requires exactly one reference image (otherwise 422) |
| `boxes` | No | Empty string | JSON array in 0–1000 coordinates, up to 32 entries. Placement is `{id, bbox, desc}`; editing is `{id, from, src_bbox, tgt_bbox, desc}` (`tgt_bbox: null` deletes). Coordinates are `[top, left, bottom, right]` |
| `upsample` | No | `true` | Prompt upsampler: expands the prompt into a full caption before generating (same processing as `/api/enhance-prompt`). `false` bypasses it (for comparison, or when sending an already-expanded caption) |
| `upsample_strict` | No | `false` | `true` stops generation when upsampling is unavailable (by default it generates with the original prompt and adds a `warning`) |
| `upsampler` | No | `KLEIN_UPSAMPLER` (default `auto`) | LLM to use: `auto`, `remote`, `local`, `pup-int8`, `pup-gguf` (see "Prompt upsampler" below) |

Omitting reference images gives T2I; one or more gives I2I / Ref2I. PNG, JPEG, and WebP are recommended. Each reference image is downscaled to at most 1 MP (the official recommendation), with the total capped at 2 MP (1 MP each for up to two images, 0.25 MP each for eight). BFL's recommended output is 2K (e.g. 2048×2048); outputs up to 1024×1024 run within 8 GiB, and 2K runs with a GPU allocation limit of 10 GiB.

Generation uses 16 steps. FLUX 3 expects a complete scene caption, so by default the prompt is expanded by the prompt upsampler before generation. When sending a caption already expanded with `/api/enhance-prompt` or a full caption you wrote yourself, add `upsample=false` to avoid expanding it twice.

Following the model card format, `boxes` is appended after the caption as a JSON array, separated by a single space. Reference each element in the caption as `<id>` (e.g. `Remove the handbag <bag_1>.`). In multi-reference edits, the nth reference can be referred to as `<ref_image_n>`.

## Prompt upsampler

### `POST /api/enhance-prompt`

| Field | Required | Default | Description |
|---|---:|---:|---|
| `prompt` | Yes | None | Short request |
| `mode` | No | `auto` | `auto` (`t2i`/`i2i` depending on whether reference images are given), `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox` |
| `image` | No | None | Reference images for editing (up to 8). Shown to the LLM in order |
| `width` / `height` | No | None | Tells the LLM the output aspect ratio for T2I |
| `upsampler` | No | `KLEIN_UPSAMPLER` (default `auto`) | LLM to use (table below) |

External LLMs (`auto`, `remote`, `local`) get the system prompts BFL publishes for external LLMs (`models/flux-3-image-klein-pup/system_prompts_external_vlm.json`, fetched automatically if missing); for edits the request (`User request: ...`) comes first and the reference images follow. The official upsampler (`pup-*`) gets its own system prompts (`system_prompts.json`) and the official format (reference images first). If no LLM is available, `503` is returned with the reason.

| `upsampler` | LLM used |
|---|---|
| `auto` | LLM servers (`KLEIN_PROMPT_LLM_URLS`, in the order listed) → local gemma-4-E4B (llama.cpp; GPU if enough VRAM is free, otherwise CPU) |
| `remote` / `local` | Only one of the above |
| `pup-int8` | INT8 version of the official upsampler (GPU, 12 GiB or more of free VRAM) |
| `pup-gguf` | GGUF Q4_K_M version of the official upsampler (GPU, 8 GiB or more of free VRAM) |

The official versions (`pup-*`) are started on each use and stopped immediately afterwards. If free VRAM is insufficient, the image model is temporarily released before starting (`/api/enhance-prompt` cannot release it during a generation, so it returns `503`). Check `upsamplers` in `GET /api/health` to see which are available.

```json
{"prompt": "Create a full-body front view photograph of the woman from the first image ...",
 "mode": "i2i", "source": "remote:http://127.0.0.1:8080/v1", "model": "...", "cached": false}
```

Results for the same request and reference images are cached (`cached: true`).

#### T2I

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A quiet library at sunset, detailed illustration.' \
  -F 'width=1024' \
  -F 'height=768' \
  -F 'seed=42'
```

#### Single Reference

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=The same character reading a book in a cafe.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=768' \
  -F 'height=768' \
  -F 'seed=42'
```

#### Multiple References

Repeat the same `image` field.

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=Use the same character, outfit, and art style.' \
  -F 'image=@/absolute/path/face.png' \
  -F 'image=@/absolute/path/outfit.png' \
  -F 'image=@/absolute/path/style.png'
```

#### Outpaint

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A detailed editorial photograph of the same character seated by a cafe window, preserving facial features and clothing, with soft afternoon light and a shallow depth of field.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'outpaint=true' \
  -F 'width=1024' \
  -F 'height=768'
```

### Accepted Response

On success, returns `202 Accepted` with the job information.

```json
{
  "id": "a12b34c56d78",
  "status": "queued",
  "step": 0,
  "steps": 16,
  "model": "flux3",
  "model_name": "FLUX 3 Image Klein",
  "seed": 42,
  "message": "生成を準備しています",
  "created_at": 1791948447.9,
  "reference_count": 0,
  "reference_max_side": null,
  "metrics": null,
  "image_url": null,
  "metadata_url": null
}
```

Pass `id` to `GET /api/jobs/{job_id}` and poll until completion.

## Job Status

### `GET /api/jobs/{job_id}`

`status` is one of the following.

| Status | Meaning |
|---|---|
| `queued` | Waiting to run |
| `upsampling` | Expanding the prompt (unless `upsample=false`) |
| `switching` | Releasing the previous model |
| `loading` | Loading the selected model |
| `encoding` | Processing the prompt and reference images |
| `generating` | Generating the image |
| `done` | Completed |
| `error` | Failed. The reason is in `message` |
| `cancelled` | Stopped |

On completion, `metrics`, `image_url`, and `metadata_url` are added. When the prompt upsampler is used, `upsample_source` (the LLM used) is included; if expansion failed and the original prompt was used, `warning` is included. The metadata JSON also records the pre-expansion prompt (`original_prompt`).

```json
{
  "id": "a12b34c56d78",
  "status": "done",
  "step": 16,
  "steps": 16,
  "model": "flux3",
  "model_name": "FLUX 3 Image Klein",
  "seed": 42,
  "message": "生成が完了しました",
  "created_at": 1791948447.9,
  "reference_count": 0,
  "reference_max_side": null,
  "metrics": {
    "model": "flux3",
    "model_name": "FLUX 3 Image Klein",
    "seconds": 3.92,
    "total_seconds": 23.98,
    "peak_allocated_gib": 3.054,
    "peak_reserved_gib": 3.355,
    "allocator_budget_gib": 8.0,
    "gpu": "NVIDIA GPU",
    "steps": 16
  },
  "image_url": "/outputs/a12b34c56d78.png",
  "metadata_url": "/outputs/a12b34c56d78.json"
}
```

The image URL and metadata URL are relative to the base URL.

## Cancellation and Model Unloading

```bash
curl -X POST http://127.0.0.1:7864/api/jobs/JOB_ID/cancel
curl -X POST http://127.0.0.1:7864/api/unload
```

During generation, a cancel request takes effect at the next step boundary (usually within 1–2 seconds). It cannot stop immediately while a model is loading. Calling `/api/unload` during generation returns `409 Conflict`.

## Gallery (Listing and Deleting Generated Images)

```bash
curl 'http://127.0.0.1:7864/api/gallery?limit=60&offset=0'
curl -X DELETE http://127.0.0.1:7864/api/gallery/JOB_ID
```

The list returns the PNGs in `outputs/`, newest first (`total` and `items`; each item has `id`, `image_url`, `metadata_url`, `created_at`, `width`, `height`, `seed`, `prompt`, `original_prompt`, `upsampler`, `upsample_source`, `reference_count`). Images remain listed after a server restart as long as the files exist.
Deleting removes the PNG, the metadata JSON, and the job's reference images (`uploads/`). Only 32-digit hexadecimal IDs are accepted, and a job that is still generating returns `409`.

## Model and Status Checks

```bash
curl http://127.0.0.1:7864/api/models
curl http://127.0.0.1:7864/api/health
```

If `available` is `false`, accept the access conditions on the official distribution page, then fetch the model with `download_models.py flux3 --acknowledge-flux3-license`. See [Third-Party Licenses and Notices](../THIRD_PARTY_NOTICES.md) for details. `active_model` is the model currently loaded in memory, or `null` if none is loaded.

## HTTP Errors

| Status code | Common causes |
|---:|---|
| `404` | The specified job or image does not exist |
| `409` | Another generation is running, the model is not downloaded, unload was requested during generation, or deletion of an image still being generated |
| `413` | A reference image exceeds 10 MB |
| `503` | No LLM is available for `/api/enhance-prompt` (reason in `detail`) |
| `422` | Input out of range, unsupported model, or unreadable image format or content |

Error bodies are usually `{"detail":"description"}`. For FastAPI's own input type validation, `detail` is an array.

## Storage and Retention

- The most recent 100 job states are kept in memory.
- Job states are lost when the server restarts.
- Generated PNGs and JSON metadata are saved to `outputs/`.
- Reference images are saved to `uploads/`.
- Saved files are not deleted automatically. They can be deleted from the gallery (in the UI or with `DELETE /api/gallery/{id}`).
