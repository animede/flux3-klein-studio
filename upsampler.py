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

"""Prompt upsampler for FLUX 3 Image Klein.

The model card requires a prompt upsampler: the model takes full scene captions, and
a short prompt sent as typed gives generic output (a one-line edit instruction even
reframes the shot). Instead of the 18.8 GB official upsampler this module drives an
OpenAI-compatible LLM with BFL's official system prompts
(`models/flux-3-image-klein-pup/system_prompts.json`: t2i / i2i / t2i_bbox / i2i_bbox),
attaching the reference images *before* the request text for edits, as the model card
describes.

The request is built as the official inference code does (`src/flux3/pup.py`): references
resized to the official pixel budget and sent as JPEG before a fixed three-line request for
edits, greedy decoding with a presence penalty of 1.5 for text-to-image.

Backends (`backend=` / `KLEIN_UPSAMPLER`):

- `auto` (default): remote LLM servers (`KLEIN_PROMPT_LLM_URLS`, comma separated
  OpenAI-compatible URLs, tried in order; none by default), then the local llama.cpp server
  with gemma-4-E4B (`KLEIN_LOCAL_LLM_*`), started on demand: on the GPU when enough VRAM is
  free (stopped right after the request so the image model gets the memory back), otherwise
  on the CPU when enough RAM is free and the CPU is not saturated (kept for a few idle minutes).
- `remote` / `local`: only one of the two above.
- `pup-int8`: BFL's official INT8 ConvRot upsampler (pup_server.py, ~10 GiB of VRAM).
- `pup-gguf`: the official upsampler as a community Q4_K_M GGUF on llama.cpp (~7 GiB of VRAM).
  Both official builds run on the GPU only, are started per request and stopped right after.
  When VRAM is short they call `release_vram` (the studio unloads the image model) first.

If no backend answers, `UpsampleUnavailable` explains why.
Results are cached by (backend, mode, request, canvas, image bytes).
"""
from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

import model_config

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPTS_PATH = ROOT / "models" / "flux-3-image-klein-pup" / "system_prompts.json"
PUP_REPO = "black-forest-labs/flux-3-image-klein-pup"
MODES = ("t2i", "i2i", "t2i_bbox", "i2i_bbox")
BACKENDS = ("auto", "remote", "local", "pup-int8", "pup-gguf")

log = logging.getLogger("klein.upsampler")


class UpsampleUnavailable(RuntimeError):
    """No LLM backend could expand the prompt; the message lists why."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM = (
    "You are the FLUX 3 Image Klein prompt upsampler. Rewrite the user's request as one complete "
    "visual scene caption: medium and style, scene, lighting and camera, each subject with its "
    "colors, materials and position in the frame. Preserve requested edits and references. "
    "Output only the caption, without explanations or markdown."
)
_system_prompts: dict | None = None
_system_lock = threading.Lock()


def system_prompts() -> dict:
    """Official system prompts; fetched once from the (ungated) pup repo if missing."""
    global _system_prompts
    with _system_lock:
        if _system_prompts is not None:
            return _system_prompts
        if not SYSTEM_PROMPTS_PATH.is_file():
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(PUP_REPO, "system_prompts.json", local_dir=SYSTEM_PROMPTS_PATH.parent)
            except Exception as exc:  # noqa: BLE001 - offline machines fall back to a short prompt
                log.warning("Official system prompts unavailable (%s); using a built-in prompt", exc)
        try:
            _system_prompts = json.loads(SYSTEM_PROMPTS_PATH.read_text())
        except (OSError, ValueError):
            _system_prompts = {}
        return _system_prompts


def system_prompt_for(mode: str) -> str:
    return system_prompts().get(mode) or _FALLBACK_SYSTEM


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def resolve_mode(mode: str | None, image_count: int) -> str:
    mode = (mode or "auto").strip().lower()
    if mode in ("", "auto"):
        return "i2i" if image_count else "t2i"
    if mode not in MODES:
        raise ValueError(f"mode は auto / {' / '.join(MODES)} のいずれかです")
    return mode


# The official request scaffolds and decoding (src/flux3/pup.py of the inference code).
I2I_SCAFFOLD = ("Reference images follow in order (the first image, the second image, ...).\n"
                "User request: {instruction}\n"
                "Output the caption.")
I2I_BBOX_SCAFFOLD = ("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
                     "User editing instruction: {instruction}\n"
                     "Decide the edit, imagine the target image, and output the annotation record.")
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
MAX_TOKENS = 3584
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724}  # 3 or more references: 512 * 512 each


def reference_size(width: int, height: int, n_refs: int, multiple: int = 32) -> tuple[int, int]:
    """Official reference budget: 1 MP for one reference, 724² for two, 512² each beyond."""
    import math

    cap = MAX_PIXELS_BY_REFS.get(n_refs, 512 * 512)
    floor = lambda v: max(multiple, int(math.floor(v / multiple)) * multiple)  # noqa: E731
    w = max(multiple, round(width / multiple) * multiple)
    h = max(multiple, round(height / multiple) * multiple)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    return w, h


def _image_part(image, n_refs: int = 1) -> dict:
    """PIL image or raw bytes -> OpenAI image_url part, preprocessed as the official upsampler does:
    RGB over white, LANCZOS to the size budget, JPEG q92."""
    from PIL import Image, ImageOps

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=92)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def build_messages(mode: str, request: str, images=(), width: int | None = None, height: int | None = None) -> list:
    text = request.strip()
    if mode.startswith("t2i") and width and height:
        # The t2i prompts accept a leading canvas line and never mention it in the caption.
        text = f"TARGET CANVAS: {width}:{height}\n{text}"
    if mode.startswith("i2i") and images:
        scaffold = I2I_BBOX_SCAFFOLD if mode == "i2i_bbox" else I2I_SCAFFOLD
        content = [_image_part(img, len(images)) for img in images]
        content.append({"type": "text", "text": scaffold.format(instruction=text)})
    else:
        content = text
    return [{"role": "system", "content": system_prompt_for(mode)}, {"role": "user", "content": content}]


def _cache_key(backend: str, mode: str, request: str, images, width, height) -> str:
    digest = hashlib.sha256(f"{backend}\n{mode}\n{width}x{height}\n{request}".encode())
    for img in images or ():
        if isinstance(img, (bytes, bytearray)):
            digest.update(bytes(img))
        else:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            digest.update(buf.getvalue())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

# No remote LLM by default: set KLEIN_PROMPT_LLM_URLS (comma separated OpenAI-compatible URLs,
# e.g. "http://127.0.0.1:8080/v1"). Without it the local llama.cpp fallback is tried directly.
DEFAULT_REMOTE_URLS = ""


def remote_urls() -> list[str]:
    raw = os.getenv("KLEIN_PROMPT_LLM_URLS")
    if raw is None:
        # Backwards compatible single-server setting.
        single = os.getenv("KLEIN_PROMPT_LLM_URL")
        raw = single if single else DEFAULT_REMOTE_URLS
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


def _model_info(base_url: str, timeout: float) -> dict | None:
    """First model on an OpenAI-compatible server, or None when it does not answer."""
    try:
        res = httpx.get(f"{base_url}/models", timeout=timeout)
        res.raise_for_status()
        data = res.json()
    except (httpx.HTTPError, ValueError):
        return None
    items = data.get("data") or data.get("models") or []
    if not items:
        return {"id": None, "multimodal": None}
    first = items[0]
    caps = first.get("capabilities")
    return {"id": first.get("id") or first.get("name") or first.get("model"),
            "multimodal": ("multimodal" in caps) if isinstance(caps, list) else None}


def _chat(base_url: str, model: str | None, messages: list, timeout: float, mode: str = "t2i") -> str:
    payload = {"messages": messages, "temperature": 0, "top_p": 0.95, "top_k": 20,
               "presence_penalty": PRESENCE_PENALTY[mode[:3]], "max_tokens": MAX_TOKENS, "stream": False}
    if model:
        payload["model"] = model
    res = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    res.raise_for_status()
    text = (res.json()["choices"][0]["message"]["content"] or "").strip()
    if not text or len(text) > 12000:
        raise ValueError("empty or oversized completion")
    return text


def _die_with_parent() -> None:
    """Linux: stop the llama-server child when the studio process dies (no orphans holding RAM/VRAM)."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - other platforms: best effort
        pass


class LocalLLM:
    """On-demand llama.cpp server with gemma-4-E4B (vision via mmproj)."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.device: str | None = None
        self.lock = threading.Lock()
        self.idle_timer: threading.Timer | None = None

    # -- configuration -----------------------------------------------------
    @staticmethod
    def server_path(device: str = "gpu") -> Path:
        """llama-server binary. CPU mode prefers a CPU-optimized build: a CUDA-oriented build made
        without GGML_NATIVE/AVX2 decodes gemma-4-E4B at ~2 tok/s on the CPU (measured on an
        i5-12400F), which turns one caption into several minutes."""
        gpu_server = Path(os.path.expanduser(os.getenv("KLEIN_LOCAL_LLM_SERVER", "~/llama.cpp/build/bin/llama-server")))
        if device == "cpu":
            cpu_server = Path(os.path.expanduser(os.getenv("KLEIN_LOCAL_LLM_SERVER_CPU", "~/llama.cpp/build-cpu/bin/llama-server")))
            if cpu_server.is_file():
                return cpu_server
        return gpu_server

    @staticmethod
    def model_path() -> Path:
        return Path(os.path.expanduser(os.getenv("KLEIN_LOCAL_LLM_MODEL", "~/models/gemma-4-E4B-it-Q4_K_M.gguf")))

    @staticmethod
    def mmproj_path() -> Path | None:
        explicit = os.getenv("KLEIN_LOCAL_LLM_MMPROJ")
        if explicit:
            path = Path(os.path.expanduser(explicit))
            return path if path.is_file() else None
        cache = Path(os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots"))
        found = sorted(cache.glob("*/mmproj-*.gguf")) if cache.is_dir() else []
        return found[0] if found else None

    @staticmethod
    def port() -> int:
        return int(os.getenv("KLEIN_LOCAL_LLM_PORT", "7865"))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port()}/v1"

    # -- resources ---------------------------------------------------------
    @staticmethod
    def free_vram_gib() -> float | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info(os.getenv("KLEIN_DEVICE", "cuda:0"))
            return free / 2**30
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def available_ram_gib() -> float | None:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
        except OSError:
            return None
        return None

    def choose_device(self, need_vision: bool, release_vram=None) -> tuple[str | None, str]:
        """('gpu'|'cpu'|None, reason). gemma-4-E4B falls back to the CPU instead of releasing VRAM."""
        if not self.server_path("gpu").is_file() and not self.server_path("cpu").is_file():
            return None, f"llama-server が見つかりません({self.server_path('gpu')})"
        if not self.model_path().is_file():
            return None, f"ローカル LLM のモデルが見つかりません({self.model_path()})"
        if need_vision and self.mmproj_path() is None:
            return None, "ローカル LLM の画像入力用 mmproj が見つかりません(KLEIN_LOCAL_LLM_MMPROJ)"
        vram = self.free_vram_gib()
        if vram is not None and vram >= _env_float("KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB", 8.0):
            return "gpu", f"空き VRAM {vram:.1f}GiB"
        ram = self.available_ram_gib()
        load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        cpus = os.cpu_count() or 1
        if ram is not None and ram < _env_float("KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB", 10.0):
            return None, f"空きメモリ不足({ram:.1f}GiB)"
        if load1 > cpus * _env_float("KLEIN_LOCAL_LLM_CPU_MAX_LOAD", 0.75):
            return None, f"CPU 負荷が高い(load {load1:.1f} / {cpus} コア)"
        return "cpu", f"空きメモリ {ram:.1f}GiB / load {load1:.1f}" if ram is not None else "cpu"

    # -- lifecycle ---------------------------------------------------------
    def _port_open(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", self.port())) == 0

    def command(self, device: str) -> list[str]:
        cpus = os.cpu_count() or 2
        # --reasoning off: the model card runs the upsampler with thinking off; gemma-4 thinks by
        # default, which on the CPU meant minutes of reasoning tokens before the caption.
        cmd = [str(self.server_path(device)), "-m", str(self.model_path()), "--host", "127.0.0.1",
               "--port", str(self.port()), "-c", "8192", "-ngl", "99" if device == "gpu" else "0",
               "-t", str(max(1, cpus // 2)), "--reasoning", "off"]
        mmproj = self.mmproj_path()
        if mmproj is not None:
            cmd += ["--mmproj", str(mmproj)]
        return cmd

    def source(self, device: str) -> str:
        return f"local-{device}"

    def model_name(self) -> str:
        return self.model_path().name

    def _start(self, device: str, need_vision: bool) -> None:
        self.stop()
        cmd = self.command(device)
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        log.info("Starting local LLM on %s: %s", device, " ".join(cmd))
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     preexec_fn=_die_with_parent)
        self.device = device
        deadline = time.monotonic() + _env_float("KLEIN_LOCAL_LLM_START_TIMEOUT_S", 180.0)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("ローカル LLM の起動に失敗しました")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port()}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        self.stop()
        raise RuntimeError("ローカル LLM の起動がタイムアウトしました")

    def stop(self) -> None:
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.device = None

    def chat(self, messages: list, need_vision: bool, mode: str = "t2i", release_vram=None) -> tuple[str, str]:
        with self.lock:
            device, reason = self.choose_device(need_vision, release_vram)
            if device is None:
                raise UpsampleUnavailable(reason)
            if self.proc is None or self.proc.poll() is not None or self.device != device:
                self._start(device, need_vision)
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None
            try:
                text = _chat(self.base_url, None, messages, timeout=_env_float("KLEIN_LOCAL_LLM_TIMEOUT_S", 300.0),
                             mode=mode)
            finally:
                if self.device == "gpu":
                    # Give the VRAM back right away: the image model loads next.
                    self.stop()
                else:
                    self.idle_timer = threading.Timer(_env_float("KLEIN_LOCAL_LLM_IDLE_S", 600.0), self.stop)
                    self.idle_timer.daemon = True
                    self.idle_timer.start()
            return text, self.source(device)


class OfficialUpsampler(LocalLLM):
    """BFL's prompt upsampler, run on the GPU per request. These builds need more VRAM than the
    image model's 8 GiB budget: when the GPU is short, ``release_vram`` (the studio unloads the
    image model) is called before giving up."""

    key = ""
    default_port = 0
    default_min_free_gib = 0.0

    def env_name(self, suffix: str) -> str:
        return "KLEIN_" + self.key.upper().replace("-", "_") + "_" + suffix

    def port(self) -> int:  # type: ignore[override]
        return int(os.getenv(self.env_name("PORT"), str(self.default_port)))

    def min_free_gib(self) -> float:
        return _env_float(self.env_name("MIN_FREE_GIB"), self.default_min_free_gib)

    def source(self, device: str) -> str:
        return self.key

    def missing(self) -> str | None:
        raise NotImplementedError

    def choose_device(self, need_vision: bool, release_vram=None) -> tuple[str | None, str]:  # type: ignore[override]
        problem = self.missing()
        if problem:
            return None, problem
        need = self.min_free_gib()
        vram = self.free_vram_gib()
        if vram is None:
            return None, "CUDA の GPU が見つかりません"
        if vram < need and release_vram is not None:
            release_vram()
            vram = self.free_vram_gib() or 0.0
        if vram < need:
            return None, f"空き VRAM が足りません({vram:.1f}GiB / 必要 {need:.0f}GiB)"
        return "gpu", f"空き VRAM {vram:.1f}GiB"


class PupGGUF(OfficialUpsampler):
    """Community Q4_K_M GGUF of the official upsampler (with its vision mmproj) on llama.cpp."""

    key = "pup-gguf"
    default_port = 7866
    default_min_free_gib = 8.0

    def model_path(self) -> Path:  # type: ignore[override]
        return model_config.PUP_GGUF_PATH

    def mmproj_path(self) -> Path | None:  # type: ignore[override]
        return model_config.PUP_MMPROJ_PATH if model_config.PUP_MMPROJ_PATH.is_file() else None

    def missing(self) -> str | None:
        if not self.server_path("gpu").is_file():
            return f"llama-server が見つかりません({self.server_path('gpu')})"
        if not self.model_path().is_file() or self.mmproj_path() is None:
            return "公式 upsampler の GGUF が未取得です(download_models.py flux3 --pup gguf)"
        return None


class PupINT8(OfficialUpsampler):
    """BFL's official INT8 ConvRot build, served by pup_server.py (Transformers + comfy-kitchen)."""

    key = "pup-int8"
    default_port = 7867
    default_min_free_gib = 12.0

    def model_path(self) -> Path:  # type: ignore[override]
        return model_config.PUP_INT8_PATH

    def missing(self) -> str | None:
        if not self.model_path().is_file() or not (model_config.PUP_CONFIG_DIR / "config.json").is_file():
            return "公式 upsampler の INT8 版が未取得です(download_models.py flux3 --pup int8)"
        return None

    def command(self, device: str) -> list[str]:
        import sys

        return [sys.executable, str(ROOT / "pup_server.py"), "--port", str(self.port()),
                "--device", os.getenv("KLEIN_DEVICE", "cuda:0")]

    def model_name(self) -> str:
        return self.model_path().name


local_llm = LocalLLM()
pup_gguf = PupGGUF()
pup_int8 = PupINT8()
OFFICIAL = {"pup-gguf": pup_gguf, "pup-int8": pup_int8}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 128


def default_backend() -> str:
    backend = (os.getenv("KLEIN_UPSAMPLER") or "auto").strip().lower()
    return backend if backend in BACKENDS else "auto"


def resolve_backend(backend: str | None) -> str:
    backend = (backend or "").strip().lower() or default_backend()
    if backend not in BACKENDS:
        raise ValueError(f"upsampler は {' / '.join(BACKENDS)} のいずれかです")
    return backend


def backends() -> dict:
    """Which backends can run here: {name: {"available": bool, "detail": str}}."""
    out = {"auto": {"available": True, "detail": "LLM サーバ → ローカル gemma-4-E4B"}}
    urls = remote_urls()
    out["remote"] = {"available": bool(urls), "detail": f"{len(urls)} 台" if urls else "KLEIN_PROMPT_LLM_URLS 未設定"}
    local_ok = local_llm.server_path("gpu").is_file() and local_llm.model_path().is_file()
    out["local"] = {"available": local_ok, "detail": local_llm.model_path().name if local_ok else "llama-server またはモデルが未配置"}
    for name, server in OFFICIAL.items():
        problem = server.missing()
        out[name] = {"available": problem is None, "detail": problem or f"GPU、空き VRAM {server.min_free_gib():.0f}GiB 以上"}
    return out


def upsample(request: str, *, mode: str = "auto", images=(), width: int | None = None,
             height: int | None = None, use_local: bool = True, backend: str | None = None,
             release_vram=None) -> dict:
    """Expand `request` into a full FLUX 3 caption.

    `backend` is one of BACKENDS (default `KLEIN_UPSAMPLER`, else "auto"). `release_vram` is called
    when an official upsampler needs more VRAM than is free (the studio unloads the image model).
    Returns {"prompt", "mode", "source", "model", "cached"}. Raises UpsampleUnavailable
    when every backend failed (the message lists each reason).
    """
    if not request or not request.strip():
        raise ValueError("prompt が空です")
    images = list(images or ())
    mode = resolve_mode(mode, len(images))
    backend = resolve_backend(backend)
    key = _cache_key(backend, mode, request, images, width, height)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return {**_cache[key], "cached": True}

    messages = build_messages(mode, request, images, width, height)
    need_vision = mode.startswith("i2i") and bool(images)
    reasons: list[str] = []
    result: dict | None = None
    override_model = os.getenv("KLEIN_PROMPT_LLM_MODEL")
    if backend in OFFICIAL:
        server = OFFICIAL[backend]
        try:
            text, source = server.chat(messages, need_vision, mode, release_vram)
            result = {"prompt": text, "mode": mode, "source": source, "model": server.model_name()}
        except UpsampleUnavailable as exc:
            reasons.append(f"{backend}: {exc}")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{backend}: {type(exc).__name__}: {exc}")
    for index, url in enumerate(remote_urls() if backend in ("auto", "remote") else []):
        info = _model_info(url, timeout=_env_float("KLEIN_PROMPT_LLM_PROBE_TIMEOUT_S", 3.0))
        if info is None:
            reasons.append(f"{url}: 応答なし")
            continue
        if need_vision and info.get("multimodal") is False:
            reasons.append(f"{url}: 画像入力に未対応")
            continue
        model = override_model if (override_model and index == 0) else info.get("id")
        try:
            text = _chat(url, model, messages, timeout=_env_float("KLEIN_PROMPT_LLM_TIMEOUT_S", 120.0), mode=mode)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{url}: {type(exc).__name__}")
            continue
        result = {"prompt": text, "mode": mode, "source": f"remote:{url}", "model": model}
        break
    if (result is None and use_local and backend in ("auto", "local")
            and os.getenv("KLEIN_LOCAL_LLM", "1") != "0"):
        try:
            text, source = local_llm.chat(messages, need_vision, mode)
            result = {"prompt": text, "mode": mode, "source": source, "model": local_llm.model_path().name}
        except UpsampleUnavailable as exc:
            reasons.append(f"ローカル LLM: {exc}")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"ローカル LLM: {type(exc).__name__}: {exc}")
    if result is None:
        raise UpsampleUnavailable("prompt upsamplerに使える LLM がありません(" + " / ".join(reasons) + ")")
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return {**result, "cached": False}


def shutdown() -> None:
    for server in (local_llm, *OFFICIAL.values()):
        server.stop()


__all__ = ["UpsampleUnavailable", "upsample", "resolve_mode", "resolve_backend", "default_backend", "backends",
           "build_messages", "system_prompt_for", "remote_urls", "local_llm", "pup_gguf", "pup_int8",
           "shutdown", "MODES", "BACKENDS"]

