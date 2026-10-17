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

import io
import json
import logging
import math
import re
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

import upsampler
from engine import Cancelled, Engine
from model_config import DEFAULT_MODEL, MODELS, ROOT

Image.MAX_IMAGE_PIXELS = 16_000_000
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "uploads"
OUTPUTS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
MAX_UPLOAD = 10 * 1024 * 1024
MAX_REFERENCES = 8
REFERENCE_MAX_SIDE = 768

app = FastAPI(title="FLUX 3 Image Klein Reference Studio（DiffusersサーバとGUIアプリ）", version="0.3.0", description="FLUX 3 Image KleinをDiffusersで動かし、8GBのVRAMに収めた画像生成・編集サーバーです。")
@app.middleware("http")
async def _revalidate_gui(request, call_next):
    # Make browsers revalidate the GUI after an update instead of running a stale app.js.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
engine = Engine()


@app.on_event("shutdown")
def _stop_local_llm():
    upsampler.shutdown()
pool = ThreadPoolExecutor(max_workers=1)
jobs = {}
jobs_lock = threading.Lock()
log = logging.getLogger("klein")


class ModelInfo(BaseModel):
    name: str
    description: str
    license: str
    quantization: str
    available: bool


class HealthResponse(BaseModel):
    status: str
    loaded: bool
    model: str
    active_model: str | None
    default_model: str
    models: dict[str, ModelInfo]
    busy: bool
    memory_budget_gib: float
    default_upsampler: str
    upsamplers: dict[str, dict[str, Any]]


class ModelsResponse(BaseModel):
    default: str
    active: str | None
    models: dict[str, ModelInfo]


class JobResponse(BaseModel):
    id: str
    status: str
    step: int
    steps: int
    model: str
    model_name: str
    seed: int
    message: str
    created_at: float
    reference_count: int
    reference_max_side: int | None
    metrics: dict[str, Any] | None = None
    image_url: str | None = None
    metadata_url: str | None = None
    upsample: bool = True
    upsampler: str = "auto"
    upsample_source: str | None = None
    warning: str | None = None


class GalleryItem(BaseModel):
    id: str
    image_url: str
    metadata_url: str | None
    created_at: float
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    prompt: str | None = None
    original_prompt: str | None = None
    upsampler: str | None = None
    upsample_source: str | None = None
    reference_count: int = 0


class GalleryResponse(BaseModel):
    total: int
    items: list[GalleryItem]


class UnloadResponse(BaseModel):
    status: str
    loaded: bool


def model_available(item):
    return Engine.model_available(item)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health", response_model=HealthResponse, summary="サーバー状態")
def health():
    active_model = engine.active_model[0] if isinstance(engine.active_model, tuple) else engine.active_model
    return {
        "status": "ok", "loaded": engine.pipe is not None,
        "model": MODELS[active_model or DEFAULT_MODEL]["base_repo"],
        "active_model": active_model, "default_model": DEFAULT_MODEL,
        "models": {
            key: {"name": item["name"], "description": item["description"],
                  "license": item["license"], "quantization": item["quantization"],
                  "available": model_available(item)}
            for key, item in MODELS.items()
        },
        "busy": engine.lock.locked(), "memory_budget_gib": engine.memory_budget_gib,
        "default_upsampler": upsampler.default_backend(), "upsamplers": upsampler.backends(),
    }


@app.get("/api/models", response_model=ModelsResponse, summary="利用可能モデル一覧")
def models():
    active_model = engine.active_model[0] if isinstance(engine.active_model, tuple) else engine.active_model
    return {
        "default": DEFAULT_MODEL,
        "active": active_model,
        "models": {
            key: {
                "name": item["name"], "description": item["description"],
                "license": item["license"], "quantization": item["quantization"],
                "available": model_available(item),
            }
            for key, item in MODELS.items()
        },
    }


def public_job(job):
    return {k: v for k, v in job.items() if k != "cancel"}


@app.get("/api/jobs/{job_id}", response_model=JobResponse, summary="生成状態を取得",
         responses={404: {"description": "ジョブが存在しません"}})
def job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "生成ジョブが見つかりません。")
        return public_job(jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse, summary="生成を停止",
          responses={404: {"description": "ジョブが存在しません"}})
def cancel_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "生成ジョブが見つかりません。")
        if jobs[job_id]["status"] in {"done", "error", "cancelled"}:
            return public_job(jobs[job_id])
        jobs[job_id]["cancel"].set()
        jobs[job_id]["message"] = "停止を受け付けました。現在の処理が終わるまでお待ちください。"
        return public_job(jobs[job_id])


@app.post("/api/unload", response_model=UnloadResponse, summary="モデルをメモリから解放",
          responses={409: {"description": "生成中のため解放できません"}})
def unload():
    if not engine.lock.acquire(blocking=False):
        raise HTTPException(409, "生成中は解放できません。")
    try:
        engine.unload()
        return {"status": "ok", "loaded": False}
    finally:
        engine.lock.release()


JOB_ID = re.compile(r"[0-9a-f]{32}")


def gallery_item(image: Path) -> dict:
    meta = {}
    metadata = image.with_suffix(".json")
    if metadata.is_file():
        try:
            meta = json.loads(metadata.read_text())
        except (OSError, ValueError):
            meta = {}
    return {
        "id": image.stem, "image_url": f"/outputs/{image.name}",
        "metadata_url": f"/outputs/{metadata.name}" if metadata.is_file() else None,
        "created_at": image.stat().st_mtime,
        **{key: meta.get(key) for key in ("width", "height", "seed", "prompt", "original_prompt",
                                          "upsampler", "upsample_source")},
        "reference_count": meta.get("reference_count") or 0,
    }


@app.get("/api/gallery", response_model=GalleryResponse, summary="生成画像の一覧(新しい順)")
def gallery(limit: int = 60, offset: int = 0):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    images = sorted((p for p in OUTPUTS.glob("*.png") if JOB_ID.fullmatch(p.stem)),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    return {"total": len(images), "items": [gallery_item(p) for p in images[offset:offset + limit]]}


@app.delete("/api/gallery/{image_id}", summary="生成画像を削除",
            responses={404: {"description": "画像が存在しません"}, 409: {"description": "生成中のジョブです"}})
def delete_gallery_item(image_id: str):
    """PNG、メタデータJSON、そのジョブの参照画像(uploads/)を削除する。"""
    if not JOB_ID.fullmatch(image_id):
        raise HTTPException(404, "画像が見つかりません。")
    with jobs_lock:
        job = jobs.get(image_id)
        if job and job["status"] not in {"done", "error", "cancelled"}:
            raise HTTPException(409, "生成中の画像は削除できません。")
    image = OUTPUTS / f"{image_id}.png"
    if not image.is_file():
        raise HTTPException(404, "画像が見つかりません。")
    removed = []
    for path in [image, OUTPUTS / f"{image_id}.json", *UPLOADS.glob(f"{image_id}_ref_*.png")]:
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return {"id": image_id, "deleted": removed}


def reference_max_side(count: int, model_key: str = DEFAULT_MODEL) -> int:
    if count <= 0:
        return REFERENCE_MAX_SIDE
    bounded = min(REFERENCE_MAX_SIDE, math.sqrt(MODELS[model_key]["reference_pixel_budget"] / count))
    return max(256, int(bounded) // 32 * 32)


def decode_reference(data: bytes, max_side: int = REFERENCE_MAX_SIDE):
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > 16_000_000:
                raise ValueError("画像は1600万画素以内にしてください。")
            im = ImageOps.exif_transpose(im)
            if im.mode in {"RGBA", "LA"} or "transparency" in im.info:
                rgba = im.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, "white")
                canvas.alpha_composite(rgba)
                result = canvas.convert("RGB")
            else:
                result = im.convert("RGB")
            # Bound reference tokens separately from the output resolution.
            result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return result.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("画像を読み込めません。PNG、JPEG、WebPなどの画像を選んでください。") from exc


def release_vram_if_idle():
    """For /api/enhance-prompt: free the image model for an official upsampler unless a job is running."""
    if engine.lock.acquire(blocking=False):
        try:
            engine.unload()
        finally:
            engine.lock.release()


def run_job(job_id, model_key, prompt, references, reference_side, width, height, seed, outpaint=False,
            box_rows=None, upsample=False, upsample_strict=False, upsampler_backend=None, shift=None):
    def progress(status, step, message):
        with jobs_lock:
            jobs[job_id].update(status=status, step=step, message=message)

    try:
        with jobs_lock:
            cancel = jobs[job_id]["cancel"]
        original_prompt = prompt
        upsample_info = None
        if upsample:
            progress("upsampling", 0, "プロンプトをLLMで展開しています")
            try:
                # This job holds engine.lock, so the image model can be unloaded directly
                # when an official upsampler needs the VRAM; it is loaded again for generation.
                upsample_info = upsampler.upsample(
                    prompt, images=references, width=width, height=height,
                    mode="i2i" if references else "t2i", backend=upsampler_backend,
                    release_vram=engine.unload,
                )
                prompt = upsample_info["prompt"]
                with jobs_lock:
                    jobs[job_id]["upsample_source"] = upsample_info["source"]
            except upsampler.UpsampleUnavailable as exc:
                if upsample_strict:
                    raise
                with jobs_lock:
                    jobs[job_id]["warning"] = f"prompt upsamplerを使えなかったため、元のプロンプトで生成しました: {exc}"
        if box_rows:
            prompt = append_boxes(prompt, box_rows)
        if cancel.is_set():
            raise Cancelled()
        reference_paths = []
        for index, reference in enumerate(references, start=1):
            path = UPLOADS / f"{job_id}_ref_{index}.png"
            reference.save(path)
            reference_paths.append(f"uploads/{path.name}")
        model_input = references if len(references) > 1 else references[0] if references else None
        image, metrics = engine.generate(model_key, prompt, model_input, width, height, seed, progress, cancel,
                                         outpaint=outpaint, shift=shift)
        if cancel.is_set():
            raise Cancelled()
        out = OUTPUTS / f"{job_id}.png"
        image.save(out)
        metadata = {
            "prompt": prompt, "seed": seed, "width": width, "height": height,
            "original_prompt": original_prompt if prompt != original_prompt else None,
            "upsampler": upsampler_backend if upsample else None,
            "upsample_source": upsample_info["source"] if upsample_info else None,
            "boxes": box_rows,
            # Keep `reference` for clients built against the original single-image API.
            "reference": reference_paths[0] if reference_paths else None,
            "references": reference_paths,
            "reference_count": len(reference_paths),
            "reference_max_side": reference_side if reference_paths else None,
            "model": model_key,
            "model_name": MODELS[model_key]["name"],
            "model_repo": MODELS[model_key]["base_repo"],
            "base_revision": MODELS[model_key]["base_revision"],
            "int8_revision": MODELS[model_key]["int8_revision"],
            "transformer_quantization": MODELS[model_key]["quantization"],
            "text_quantization": MODELS[model_key]["text_quantization"],
            **metrics,
        }
        (OUTPUTS / f"{job_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        with jobs_lock:
            jobs[job_id].update(status="done", step=MODELS[model_key]["steps"], message="生成が完了しました", metrics=metrics,
                               image_url=f"/outputs/{job_id}.png", metadata_url=f"/outputs/{job_id}.json")
    except Cancelled:
        progress("cancelled", 0, "生成を停止しました")
        engine.unload()
    except Exception as exc:
        log.exception("Generation failed: %s", job_id)
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPUメモリが不足しました。解像度を512または768に下げて再試行してください。"
        with jobs_lock:
            jobs[job_id].update(status="error", message=message[:1000])
        engine.unload()
    finally:
        engine.lock.release()


@app.post("/api/enhance-prompt", summary="短い依頼をFLUX 3用の完全なキャプションへ展開")
async def enhance_prompt(
    prompt: str = Form(..., description="短い依頼文。"),
    mode: str = Form("auto", description="auto / t2i / i2i / t2i_bbox / i2i_bbox。autoは参照画像の有無で決める。"),
    image: list[UploadFile] | None = File(None, description="編集(i2i)の参照画像。順番どおりLLMに見せる。"),
    width: int | None = Form(None, description="T2Iの出力幅(キャンバスの縦横比を伝える)。"),
    height: int | None = Form(None, description="T2Iの出力高さ。"),
    upsampler_backend: str | None = Form(None, alias="upsampler", description="auto / remote / local / pup-int8 / pup-gguf。省略時はKLEIN_UPSAMPLER(既定auto)。"),
):
    """公式システムプロンプトでLLMにキャプションを書かせる(upsampler.py 参照)。

    リモートのLLMサーバ → ローカルのgemma-4-E4B(GPU/CPU)の順に試し、どれも使えなければ503。
    """
    from fastapi.concurrency import run_in_threadpool

    try:
        upsampler_backend = upsampler.resolve_backend(upsampler_backend)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    images = []
    for upload in [u for u in (image or []) if u.filename][:MAX_REFERENCES]:
        data = await upload.read(MAX_UPLOAD + 1)
        await upload.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "参照画像は1枚につき10MB以内にしてください。")
        try:
            # Decode here so a non-image upload is a 422, not an error inside the LLM call.
            images.append(decode_reference(data, max_side=1024))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    try:
        result = await run_in_threadpool(
            upsampler.upsample, prompt, mode=mode, images=images, width=width, height=height,
            backend=upsampler_backend, release_vram=release_vram_if_idle,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except upsampler.UpsampleUnavailable as exc:
        log.warning("Prompt upsampler unavailable: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    return result


def parse_boxes(raw: str) -> list:
    """Layout rows: {"id", "bbox", "desc"}(T2I) or edit rows {"id", "from", "src_bbox", "tgt_bbox"(null=削除), "desc"}."""
    def valid_box(box):
        return (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in box)
                and box[0] <= box[2] and box[1] <= box[3])

    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows or len(rows) > 32:
        raise ValueError("boxes must be a non-empty list of at most 32 rows")
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip():
            raise ValueError("each row needs an id")
        if "bbox" in row:
            if not valid_box(row["bbox"]):
                raise ValueError("bbox")
        elif "src_bbox" in row or "tgt_bbox" in row:
            if row.get("src_bbox") is not None and not valid_box(row["src_bbox"]):
                raise ValueError("src_bbox")
            if row.get("tgt_bbox") is not None and not valid_box(row["tgt_bbox"]):
                raise ValueError("tgt_bbox")
            if row.get("src_bbox") is None and row.get("tgt_bbox") is None:
                raise ValueError("src_bbox or tgt_bbox")
        else:
            raise ValueError("bbox / src_bbox / tgt_bbox")
    return rows


def append_boxes(prompt: str, rows: list) -> str:
    """モデルカードの書式: キャプションの後に空白1つ、そのあとにJSON配列(要素はキャプション中で <id> を引用)。"""
    return f"{prompt.rstrip()} {json.dumps(rows, ensure_ascii=False)}"


@app.post(
    "/api/generate", status_code=202, response_model=JobResponse, summary="画像生成を開始",
    responses={
        409: {"description": "別の生成を実行中、またはモデルが未取得です"},
        413: {"description": "参照画像が10MBを超えています"},
        422: {"description": "入力値が範囲外、または画像を読み込めません"},
    },
)
async def generate(
    prompt: str = Form(..., description="生成内容。1〜4000文字。"),
    image: list[UploadFile] | None = File(None, description="参照画像。imageフィールドを最大8回指定可能。"),
    width: int = Form(768, description="出力幅。256〜1024の32の倍数。"),
    height: int = Form(768, description="出力高さ。256〜1024の32の倍数。"),
    seed: int = Form(-1, description="-1でランダム、または0〜4294967295。"),
    model: str = Form(DEFAULT_MODEL, description="flux3。"),
    outpaint: bool = Form(False, description="FLUX 3のアウトペイントLoRAを使用(参照画像1枚のときのみ)"),
    boxes: str = Form("", description="任意のバウンディングボックスJSON。T2Iは{id,bbox,desc}、編集は{id,from,src_bbox,tgt_bbox,desc}(tgt_bbox=nullで削除)。"),
    upsample: bool = Form(True, description="prompt upsampler(既定で有効、モデルカードが必須としている)。生成前にLLMでプロンプトを完全なキャプションへ展開する(/api/enhance-promptと同じ)。falseでバイパス(比較用、または作成済みのキャプションを送るとき)。"),
    upsample_strict: bool = Form(False, description="upsampleが使えないときに生成を止める(既定は元のプロンプトのまま生成し警告を付ける)。"),
    upsampler_backend: str | None = Form(None, alias="upsampler", description="auto / remote / local / pup-int8 / pup-gguf。省略時はKLEIN_UPSAMPLER(既定auto)。公式版(pup-*)はGPUを多く使い、足りなければ画像モデルを一時的に解放する。"),
    shift: float | None = Form(None, description="タイムステップのシフト(alpha)。1〜50。省略時はKLEIN_TIMESTEP_SHIFT(既定16、BFLの推奨値。2026-10-17以前は6.93)。"),
):
    prompt = prompt.strip()
    if not prompt or len(prompt) > 4000:
        raise HTTPException(422, "プロンプトは1〜4000文字で入力してください。")
    box_rows = None
    if boxes.strip():
        try:
            box_rows = parse_boxes(boxes)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                422,
                "boxesはJSON配列で指定してください。各要素は id と、bbox(配置)または "
                "src_bbox / tgt_bbox(編集、tgt_bbox=nullで削除)を持ち、座標は0〜1000の[top, left, bottom, right]です。",
            ) from exc
    if any(n < 256 or n > 1024 or n % 32 for n in (width, height)):
        raise HTTPException(422, "幅・高さは256〜1024の32の倍数で指定してください。")
    if seed < -1 or seed > 2**32 - 1:
        raise HTTPException(422, "シードは-1（ランダム）または0〜4294967295で指定してください。")
    try:
        upsampler_backend = upsampler.resolve_backend(upsampler_backend)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if shift is not None and not (1.0 <= shift <= 50.0):
        raise HTTPException(422, "shiftは1〜50で指定してください。")
    model = model.lower()
    if model not in MODELS:
        raise HTTPException(422, f"モデルは {' / '.join(MODELS)} から選択してください。")
    if outpaint and model != "flux3":
        raise HTTPException(422, "アウトペイントはFLUX 3でのみ利用できます。")
    if outpaint and not (MODELS[model]["int8_dir"] / MODELS[model]["outpaint_lora_file"]).is_file():
        raise HTTPException(409, "アウトペイントLoRAが未取得です。download_models.py flux3を実行してください。")
    if not model_available(MODELS[model]):
        raise HTTPException(409, f"{MODELS[model]['name']}は未取得です。download_models.py {model} を実行してください。")
    uploads = [upload for upload in (image or []) if upload.filename]
    if outpaint and len(uploads) != 1:
        for upload in uploads:
            await upload.close()
        raise HTTPException(422, "アウトペイントは参照画像1枚で指定してください。")
    if len(uploads) > MAX_REFERENCES:
        for upload in uploads:
            await upload.close()
        raise HTTPException(422, f"参照画像は{MAX_REFERENCES}枚までです。")
    references = []
    max_side = reference_max_side(len(uploads), model)
    for upload in uploads:
        data = await upload.read(MAX_UPLOAD + 1)
        await upload.close()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "参照画像は1枚につき10MB以内にしてください。")
        try:
            references.append(decode_reference(data, max_side=max_side))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if not engine.lock.acquire(blocking=False):
        raise HTTPException(409, "別の画像を生成しています。完了後に再試行してください。")
    if seed == -1:
        seed = secrets.randbelow(2**32)
    job_id = uuid.uuid4().hex
    with jobs_lock:
        # Retain recent job status; output files remain available on disk.
        while len(jobs) >= 100:
            jobs.pop(next(iter(jobs)))
        jobs[job_id] = {"id": job_id, "status": "queued", "step": 0, "steps": MODELS[model]["steps"],
                        "model": model, "model_name": MODELS[model]["name"],
                        "seed": seed, "message": "生成を準備しています", "created_at": time.time(),
                        "reference_count": len(references),
                        "reference_max_side": max_side if references else None,
                        "upsample": bool(upsample), "upsampler": upsampler_backend,
                        "upsample_source": None, "warning": None,
                        "cancel": threading.Event()}
        response = public_job(jobs[job_id])
    try:
        pool.submit(run_job, job_id, model, prompt, references, max_side, width, height, seed, outpaint,
                    box_rows, bool(upsample), bool(upsample_strict), upsampler_backend, shift)
    except Exception:
        engine.lock.release()
        raise
    return response
