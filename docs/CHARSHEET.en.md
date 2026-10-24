# Character Sheet API

[日本語](CHARSHEET.md) | English

Turns one character image into full-body images from eight directions (front, back, both sides and four
45-degree views) and a single sheet that lays them out. Each direction is one Edit of the reference image
(asking for several directions in one image sometimes drew both side profiles facing the same way, so it is not used).

- Generation is shared with the other jobs: one at a time. While a sheet runs, `POST /api/generate` returns `409` too.
- Per direction: 7–9 seconds of generation plus the prompt upsampler (depends on the LLM). Eight directions take about 1–3 minutes.
- Direction keys, labels and the response shape match common character-sheet APIs (such as the Qwen-Image-Edit multiple-angles LoRA sheets).

## Flow

1. Send the image to `POST /api/charsheet/generate` (`202` with `job_id`)
2. Poll `GET /api/charsheet/jobs/{job_id}` every 1–2 seconds until `status` is `done`, `error` or `cancelled`
   (`views[].url` fills in as each direction finishes)
3. Fetch `sheet_url` (all directions in one PNG) and `zip_url` (every image in a ZIP)

```bash
curl -F image=@character.png -F seed=42 http://127.0.0.1:7864/api/charsheet/generate
curl http://127.0.0.1:7864/api/charsheet/jobs/JOB_ID
curl -X POST http://127.0.0.1:7864/api/charsheet/jobs/JOB_ID/cancel
```

## Directions

| Key | Label | Direction |
|---|---|---|
| `front` | 前 / Front | Facing the camera |
| `front_left_45` | 左前45度 / Front-Left 45° | The character turns 45 degrees to their own left (the face points toward the right of the image) |
| `left` | 左 / Left | 90 degrees to their own left: a profile with the nose pointing to the **right** edge of the image |
| `back_left_45` | 左後ろ45度 / Back-Left 45° | 135 degrees to their own left: mostly the back, the head toward the right of the image |
| `back` | 後ろ / Back | Directly from behind |
| `back_right_45` | 右後ろ45度 / Back-Right 45° | 135 degrees to their own right: mostly the back, the head toward the left of the image |
| `right` | 右 / Right | 90 degrees to their own right: a profile with the nose pointing to the **left** edge of the image |
| `front_right_45` | 右前45度 / Front-Right 45° | 45 degrees to their own right (the face points toward the left of the image) |

Left and right are from the character's point of view (the opposite of the direction the camera would move).
This matches the sheets made with the Qwen-Image-Edit multiple-angles LoRA.

Sheet layout (4 columns × 2 rows): `front` `front_left_45` `front_right_45` `left` on top, `right` `back_left_45` `back_right_45` `back` below.

## Preparation (prep)

An image with a background, a cropped figure or a turned pose is first converted by one Edit into a single
"white background, full body, facing front" image (prep), and every direction is made from that.

- `auto` (default): skip when at least 85% of the image border is white (bright and unsaturated); otherwise convert
- `on`: always convert / `off`: never convert

After a conversion, that image is used as `front` as is (no separate front generation) and is also available at `prep_url`.
Without one, `front` is generated from the input as well (so its size and framing match the other directions).
`reference_url` is the image every direction was made from (`prep_url` after a conversion, otherwise the same as `input_url`).

To get only the "white background, full body, facing front" image, send `views=front` and `prep=on`: one Edit.

## `POST /api/charsheet/generate`

`multipart/form-data`. On success `202` with the same body as the [job status](#get-apicharsheetjobsjob_id).

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | One character image (up to 10 MB and 16 megapixels). Transparent PNGs are composited on white; images above 1 MP are downscaled for the reference |
| `seed` | int | `-1` | One seed for every direction. `-1` for random, or 0–4294967295 |
| `views` | string | all eight | Comma-separated direction keys, generated in the given order |
| `width` / `height` | int | `768` / `1024` | Output size per direction. Multiples of 32 from 256 to 2688, at most 2048×2048 pixels (as `/api/generate`) |
| `size` | int | none | One side for a square output (only when `width` and `height` are omitted; rounded down to a multiple of 32) |
| `prep` | string | `auto` | `auto` / `on` / `off` (above) |
| `upsample` | bool | `true` | Expand each direction into a caption with the LLM, showing it the reference (as `/api/generate`) |
| `upsample_strict` | bool | `false` | Fail when no upsampler is available (by default the plain text is used and `warning` is set) |
| `upsampler` | string | environment | `auto` / `remote` / `local` / `pup-int8` / `pup-gguf` (as `/api/generate`) |
| `shift` | float | environment | Timestep shift (as `/api/generate`) |

Errors: `409` (another job running, model missing), `413` (over 10 MB), `422` (unknown direction key, prep, size or seed, unreadable image).

## `GET /api/charsheet/jobs/{job_id}`

```json
{
  "job_id": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
  "status": "running",
  "progress": 3,
  "total": 8,
  "views": [
    {"key": "front", "label_ja": "前", "label_en": "Front", "status": "done",
     "url": "/api/charsheet/jobs/0f1e…/images/front.png"},
    {"key": "back", "label_ja": "後ろ", "label_en": "Back", "status": "running", "url": null}
  ],
  "sheet_url": null,
  "zip_url": null,
  "error": null,
  "seed": 42, "width": 768, "height": 1024,
  "requested_views": ["front", "back", "left", "right", "front_left_45", "front_right_45", "back_left_45", "back_right_45"],
  "prep": "auto", "prep_applied": true, "prep_status": "done",
  "prep_url": "/api/charsheet/jobs/0f1e…/prep.png",
  "input_url": "/api/charsheet/jobs/0f1e…/input.png",
  "reference_url": "/api/charsheet/jobs/0f1e…/prep.png",
  "white_background_ratio": 0.017,
  "current": "back", "step": 9, "steps": 16, "message": "back: 生成中 9 / 16",
  "upsample": true, "upsampler": "auto", "warning": null,
  "metadata_url": "/api/charsheet/jobs/0f1e…/metadata.json",
  "created_at": 1792656173.05
}
```

| Field | Description |
|---|---|
| `status` | `queued` → `running` → `done` / `error` / `cancelled` |
| `progress` / `total` | Finished directions / requested directions (prep is not counted) |
| `views[].status` | `queued` / `running` / `done` / `error` / `cancelled` |
| `views[].url` | PNG of a finished direction |
| `sheet_url` / `zip_url` | Set when every direction is done. The sheet shows the finished directions; the ZIP holds each direction, `sheet.png`, `input.png` and `prep.png` (if any) |
| `error` | The failed stage and the reason (e.g. `back: GPUメモリが不足しました…`) |
| `prep_applied` / `prep_status` | Whether prep ran / `skipped`, `queued`, `running`, `done`, `error`, `cancelled` |
| `white_background_ratio` | Share of white pixels on the input border (used by `auto`; 0.85 or more counts as white) |
| `current` / `step` / `steps` / `message` | The stage in progress (`prep` or a direction key) and its step |
| `warning` | Set when, for example, no upsampler was available and the plain text was used |
| `metadata_url` | JSON with the actual prompt of each direction (after the upsampler), the upsampler source and timings |

Cancelling (`POST /api/charsheet/jobs/{job_id}/cancel`) takes effect at the next step; finished directions are kept.

## Files

| Path | Content |
|---|---|
| `GET /api/charsheet/jobs/{job_id}/images/{key}.png` | One direction |
| `GET /api/charsheet/jobs/{job_id}/input.png` | The input image (after decoding and downscaling) |
| `GET /api/charsheet/jobs/{job_id}/prep.png` | The white-background, full-body front image (when prep ran) |
| `GET /api/charsheet/jobs/{job_id}/sheet.png` | All directions in one sheet |
| `GET /api/charsheet/jobs/{job_id}/download.zip` | Every image in a ZIP |
| `GET /api/charsheet/jobs/{job_id}/metadata.json` | The state and the prompt of each direction |

Files are saved in `outputs/charsheet/{job_id}/` and never deleted automatically. The job state is written to `job.json`
in the same folder, so `GET /api/charsheet/jobs/{job_id}` still works after a restart (a job that was running when the
server stopped reads as `error`). Sheets do not appear in the gallery (`/api/gallery`).

## Prompts

Each direction is `BASE_PROMPT` + `VIEWS[].prompt` in `charsheet.py`. Nothing is specific to a character; the upsampler
looks at the reference and adds the outfit and other details. Points that matter:

- The direction is given both as how far the character turned to their own left or right and as which edge of the image the nose or face points to.
- The 45-degree views describe the body (exactly halfway between the front view and the profile, shoulders and hips at a diagonal, the near shoulder in front and the far arm partly hidden), because the turn tends to come out too shallow.
  Describing the eyes made the bangs cover them on an anime character, so the text leaves the eyes out.
- The rear views say that front-only details (chest prints, front pockets, a bib, buttons) are not visible, because they tend to appear on the back.
- For one-sided details (a single sock), the text says on which side of the image the character's left side appears.
