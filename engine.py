"""Local FLUX 3 Image Klein inference."""
import gc
import os
import threading
import time

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from model_config import MODELS


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self):
        self.pipe = None
        self.active_model = None
        self.device = os.getenv("KLEIN_DEVICE", "cuda:0")
        self.memory_budget_gib = float(os.getenv("KLEIN_MEMORY_BUDGET_GIB", "8.0"))
        self.lock = threading.Lock()

    def load(self, model_key, progress, outpaint=False):
        if model_key not in MODELS:
            raise ValueError(f"未対応のモデルです: {model_key}")
        active_key = (model_key, bool(outpaint))
        if self.pipe is not None and self.active_model == active_key:
            return
        if self.pipe is not None:
            progress("switching", 0, "モデルを切り替えるため、現在のモデルを解放しています")
            self.unload()

        import torch

        config = MODELS[model_key]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA対応GPUが見つかりません。NVIDIA GPUとドライバーを確認してください。")
        if not self.model_available(config):
            raise RuntimeError(f"{config['name']}が未準備です。download_models.py を実行してください。")
        torch.cuda.set_device(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        budget_gib = self.memory_budget_gib
        torch.cuda.set_per_process_memory_fraction(min(budget_gib * 2**30 / total, 1.0), self.device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        progress("loading", 0, f"{config['name']}を読み込んでいます")
        try:
            pipe = self._load_flux3(config, dtype, outpaint=outpaint)
        except Exception:
            self.pipe = None
            self.active_model = None
            gc.collect()
            torch.cuda.empty_cache()
            raise
        self.pipe = pipe
        self.active_model = active_key

    @staticmethod
    def model_available(config):
        root = config["model_dir"]
        return ((root / "modular_model_index.json").is_file()
                and all(any((root / part).glob("*.safetensors"))
                        for part in ("text_encoder", "vae"))
                and (config["int8_dir"] / config["int8_file"]).is_file())

    def _load_flux3(self, config, dtype, outpaint=False):
        from flux3_int8 import Flux3ConvRotPipeline

        return Flux3ConvRotPipeline(
            config["model_dir"],
            config["int8_dir"] / config["int8_file"],
            device=self.device,
            dtype=dtype,
            lora_checkpoint=(config["int8_dir"] / config["outpaint_lora_file"])
            if outpaint and (config["int8_dir"] / config["outpaint_lora_file"]).is_file() else None,
        )

    def generate(self, model_key, prompt, reference, width, height, seed, progress, cancel, outpaint=False):
        import torch
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.load(model_key, progress, outpaint=outpaint)
        if cancel.is_set():
            raise Cancelled()
        progress("encoding", 0, "プロンプトと参照画像を読み取っています")

        config = MODELS[model_key]

        generation_start = time.monotonic()
        with torch.inference_mode():
            progress("generating", 0, f"{config['steps']}ステップで生成を開始します")
            padded_reference = reference
            original = None
            if outpaint and reference is not None and not isinstance(reference, list):
                from PIL import Image
                original = reference.convert("RGB")
                padded_reference = Image.new("RGB", (width, height), (128, 128, 128))
                scale = min(width / original.width, height / original.height)
                resized = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
                pad_xy = ((width - resized.width) // 2, (height - resized.height) // 2)
                padded_reference.paste(resized, pad_xy)
            else:
                pad_xy = None
            kwargs = {
                "prompt": prompt, "width": width, "height": height,
                "num_inference_steps": config["steps"],
                "generator": torch.Generator(device=self.device).manual_seed(seed),
                "output": "images",
                "output_type": "pil",
            }
            if padded_reference is not None:
                kwargs["image"] = padded_reference
            # ステップごとの進捗と停止: パイプラインにステップのコールバックが無いため、
            # Transformer の forward 呼び出し(蒸留版は CFG なしで 1 ステップ 1 回)を数える。
            # 停止要求はここで Cancelled を投げて生成を打ち切る。
            steps_total = config["steps"]
            calls = {"n": 0}

            def _on_transformer_call(_module, _args):
                if cancel.is_set():
                    raise Cancelled()
                calls["n"] += 1
                step = min(calls["n"], steps_total)
                progress("generating", step, f"生成中 {step} / {steps_total}")

            handle = self.pipe.transformer.register_forward_pre_hook(_on_transformer_call)
            try:
                output = self.pipe(**kwargs)
            finally:
                handle.remove()
            metrics_calls = calls["n"]
            result = output[0] if isinstance(output, (list, tuple)) else output.images[0]
            if original is not None:
                result.paste(resized, pad_xy)
        torch.cuda.synchronize(self.device)
        return result, {
            "model": model_key,
            "model_name": MODELS[model_key]["name"],
            "seconds": round(time.monotonic() - generation_start, 2),
            "total_seconds": round(time.monotonic() - started, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(self.device) / 2**30, 3),
            "allocator_budget_gib": round(
                self.memory_budget_gib, 3
            ),
            "gpu": torch.cuda.get_device_name(self.device),
            "steps": config["steps"],
            "transformer_calls": metrics_calls,
            "guidance": config["guidance"],
        }

    def unload(self):
        if self.pipe is not None:
            if hasattr(self.pipe, "unload"):
                self.pipe.unload()
            if hasattr(self.pipe, "remove_all_hooks"):
                self.pipe.remove_all_hooks()
            self.pipe = None
        self.active_model = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
