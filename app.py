import io
import json
import logging
import math
import os
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

LLM_BASE_URL = os.getenv("KLEIN_PROMPT_LLM_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("KLEIN_PROMPT_LLM_MODEL", "bartowski/google_gemma-4-31B-it-GGUF:Q6_K_L")
app = FastAPI(title="FLUX 3 Image Klein Reference Studio（評価版）", version="0.3.0", description="FLUX 3 Image Kleinでローカル画像生成・編集を行います。")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
engine = Engine()
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


def run_job(job_id, model_key, prompt, references, reference_side, width, height, seed, outpaint=False):
    def progress(status, step, message):
        with jobs_lock:
            jobs[job_id].update(status=status, step=step, message=message)

    try:
        with jobs_lock:
            cancel = jobs[job_id]["cancel"]
        reference_paths = []
        for index, reference in enumerate(references, start=1):
            path = UPLOADS / f"{job_id}_ref_{index}.png"
            reference.save(path)
            reference_paths.append(f"uploads/{path.name}")
        model_input = references if len(references) > 1 else references[0] if references else None
        if outpaint:
            image, metrics = engine.generate(model_key, prompt, model_input, width, height, seed, progress, cancel, outpaint=True)
        else:
            image, metrics = engine.generate(model_key, prompt, model_input, width, height, seed, progress, cancel)
        if cancel.is_set():
            raise Cancelled()
        out = OUTPUTS / f"{job_id}.png"
        image.save(out)
        metadata = {
            "prompt": prompt, "seed": seed, "width": width, "height": height,
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


@app.post("/api/enhance-prompt")
async def enhance_prompt(prompt: str = Form(...), mode: str = Form("t2i")):
    """Expand a short request into the full scene caption expected by FLUX 3."""
    import httpx
    system = ("You are the FLUX 3 Image Klein prompt upsampler. Rewrite the user's request "
              "as one complete visual scene caption. Include subject identity, composition, "
              "position, scale, environment, lighting, camera viewpoint, lens, materials, "
              "colors, texture, and depth of field. Preserve requested edits and references. "
              "Output only the caption, without explanations or markdown.")
    if mode == "i2i":
        system += " For editing, describe the existing scene first and fold the requested change into it."
    payload = {"model": LLM_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt.strip()}], "temperature": 0.2, "max_tokens": 1200, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=payload)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
        if not text or len(text) > 8000:
            raise ValueError("invalid completion")
        return {"prompt": text, "model": LLM_MODEL}
    except Exception as exc:
        log.warning("Prompt enhancement failed: %s", exc)
        raise HTTPException(502, "プロンプト強化LLMに接続できませんでした。") from exc


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
    outpaint: bool = Form(False, description="FLUX 3のアウトペイントLoRAを使用"),
    boxes: str = Form("", description="任意のバウンディングボックスJSON"),
):
    prompt = prompt.strip()
    if not prompt or len(prompt) > 4000:
        raise HTTPException(422, "プロンプトは1〜4000文字で入力してください。")
    if boxes.strip():
        try:
            parsed_boxes = json.loads(boxes)
            if not isinstance(parsed_boxes, list) or len(parsed_boxes) > 32:
                raise ValueError
            for item in parsed_boxes:
                bbox = item["bbox"]
                if len(bbox) != 4 or any(not isinstance(v, (int, float)) or v < 0 or v > 1000 for v in bbox):
                    raise ValueError
            prompt = f"{prompt}\n\nLayout boxes (0-1000 grid):\n{json.dumps(parsed_boxes, ensure_ascii=False)}"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(422, "boxesはbboxを含むJSON配列で指定してください。") from exc
    if any(n < 256 or n > 1024 or n % 32 for n in (width, height)):
        raise HTTPException(422, "幅・高さは256〜1024の32の倍数で指定してください。")
    if seed < -1 or seed > 2**32 - 1:
        raise HTTPException(422, "シードは-1（ランダム）または0〜4294967295で指定してください。")
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
                        "cancel": threading.Event()}
        response = public_job(jobs[job_id])
    try:
        pool.submit(run_job, job_id, model, prompt, references, max_side, width, height, seed, outpaint)
    except Exception:
        engine.lock.release()
        raise
    return response
