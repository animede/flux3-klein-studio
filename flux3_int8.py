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

"""Load the official Comfy-format FLUX 3 ConvRot INT8 DiT into Diffusers.

Black Forest Labs publishes the 8 GB INT8 checkpoint as a single ComfyUI
safetensors file.  Diffusers supports FLUX 3, but it does not currently have a
loader for Comfy's ``int8_tensorwise`` + ConvRot layout.  This module bridges
that format without converting or dequantizing the weights.
"""

from __future__ import annotations

import json
import gc
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn


class ConvRotInt8Linear(nn.Module):
    """A weight-only INT8 linear backed by comfy-kitchen's ConvRot kernel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        convrot_groupsize: int = 256,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.convrot_groupsize = convrot_groupsize
        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("weight_scale", weight_scale.float(), persistent=True)
        self.register_buffer("bias", bias, persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        import comfy_kitchen as ck

        return ck.int8_linear(
            inputs,
            self.weight,
            self.weight_scale,
            self.bias,
            self.compute_dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, convrot_groupsize={self.convrot_groupsize}"
        )


class ConvRotLoRALinear(nn.Module):
    """INT8 ConvRot linear with a BF16 low-rank residual adapter."""

    def __init__(self, base: ConvRotInt8Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, strength: float):
        super().__init__()
        self.base = base
        self.strength = float(strength)
        self.register_buffer("lora_a", lora_a, persistent=False)
        self.register_buffer("lora_b", lora_b, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        residual = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs.to(self.lora_a.dtype), self.lora_a), self.lora_b
        )
        return result + residual.to(result.dtype) * self.strength

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def _quant_target_names(source_name: str) -> list[str]:
    """Map an original FLUX 3 linear name to one or three Diffusers names."""
    name = source_name
    if name.startswith("content_mode_blocks."):
        name = "mode_blocks." + name.removeprefix("content_mode_blocks.")
    elif name.startswith("single_blocks."):
        name = "single_transformer_blocks." + name.removeprefix("single_blocks.")

    suffixes = {
        ".attn_out": ".attn.to_out.0",
        ".mlp_in": ".ff.linear_in",
        ".mlp_out": ".ff.linear_out",
    }
    for old, new in suffixes.items():
        if name.endswith(old):
            return [name[: -len(old)] + new]
    if name.endswith(".qkv"):
        prefix = name[: -len(".qkv")] + ".attn."
        return [prefix + "to_q", prefix + "to_k", prefix + "to_v"]
    raise ValueError(f"Unsupported quantized FLUX 3 tensor: {source_name}")


def _replace_module(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, name, module)


def load_flux3_convrot_transformer(
    checkpoint: str | Path,
    config_dir: str | Path,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load BFL's official distilled ConvRot INT8 transformer on CPU.

    The returned module keeps INT8 weights and FP32 row scales intact. Move it
    to CUDA without a dtype argument so the FP32 scales are not downcast.
    """
    from diffusers import Flux3Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers

    checkpoint = Path(checkpoint)
    config_dir = Path(config_dir)
    config = Flux3Transformer2DModel.load_config(config_dir)
    with init_empty_weights():
        model = Flux3Transformer2DModel.from_config(config)

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())

        quant_bases = sorted(
            key.removesuffix(".comfy_quant") for key in keys if key.endswith(".comfy_quant")
        )
        for source_base in quant_bases:
            quant_config = json.loads(bytes(tensors.get_tensor(source_base + ".comfy_quant").tolist()))
            if quant_config.get("format") != "int8_tensorwise" or not quant_config.get("convrot"):
                raise ValueError(f"Unsupported quantization config for {source_base}: {quant_config}")

            weight = tensors.get_tensor(source_base + ".weight")
            weight_scale = tensors.get_tensor(source_base + ".weight_scale")
            targets = _quant_target_names(source_base)
            weights = weight.chunk(len(targets), dim=0)
            scales = weight_scale.chunk(len(targets), dim=0)
            for target, target_weight, target_scale in zip(targets, weights, scales):
                old = model.get_submodule(target)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at {target}, found {type(old).__name__}")
                if tuple(target_weight.shape) != (old.out_features, old.in_features):
                    raise ValueError(
                        f"Shape mismatch for {target}: checkpoint {tuple(target_weight.shape)}, "
                        f"model {(old.out_features, old.in_features)}"
                    )
                _replace_module(
                    model,
                    target,
                    ConvRotInt8Linear(
                        old.in_features,
                        old.out_features,
                        target_weight.contiguous(),
                        target_scale.contiguous(),
                        compute_dtype=compute_dtype,
                        convrot_groupsize=int(quant_config.get("convrot_groupsize", 256)),
                    ),
                )

        floating = {
            key: tensors.get_tensor(key)
            for key in keys
            if tensors.get_slice(key).get_dtype() in {"BF16", "F16", "F32"}
            and not key.endswith(".weight_scale")
        }

    floating = convert_flux3_transformer_checkpoint_to_diffusers(floating)
    expected = dict(model.named_parameters())
    missing = set(expected) - set(floating)
    unexpected = set(floating) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"FLUX 3 floating tensor mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, value in floating.items():
        set_module_tensor_to_device(model, name, "cpu", value=value, dtype=compute_dtype)

    meta = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"FLUX 3 loader left meta tensors: {meta[:10]}")
    model.eval()
    return model


def convrot_model_bytes(model: nn.Module) -> int:
    """Return resident parameter and buffer bytes without double counting."""
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


def _lora_target_name(source_name: str) -> str:
    name = source_name.removeprefix("diffusion_model.")
    if name.startswith("content_mode_blocks."):
        name = "mode_blocks." + name.removeprefix("content_mode_blocks.")
    elif name.startswith("single_blocks."):
        name = "single_transformer_blocks." + name.removeprefix("single_blocks.")
    replacements = {
        ".attn_out": ".attn.to_out.0",
        ".q_proj": ".attn.to_q",
        ".k_proj": ".attn.to_k",
        ".v_proj": ".attn.to_v",
        ".mlp_in": ".ff.linear_in",
        ".mlp_out": ".ff.linear_out",
    }
    for old, new in replacements.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    raise ValueError(f"Unsupported FLUX 3 LoRA tensor: {source_name}")


def load_flux3_outpaint_lora(model: nn.Module, checkpoint: str | Path, strength: float = 1.5) -> nn.Module:
    """Attach an AusBoss-style FLUX 3 outpaint LoRA to ConvRot linears."""
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        bases = sorted(key.removesuffix(".lora_A.weight") for key in tensors.keys() if key.endswith(".lora_A.weight"))
        for base_name in bases:
            target = _lora_target_name(base_name)
            module = model.get_submodule(target)
            if isinstance(module, ConvRotLoRALinear):
                module = module.base
            if not isinstance(module, ConvRotInt8Linear):
                raise TypeError(f"LoRA target is not ConvRot INT8: {target} ({type(module).__name__})")
            a = tensors.get_tensor(base_name + ".lora_A.weight").contiguous()
            b = tensors.get_tensor(base_name + ".lora_B.weight").contiguous()
            if tuple(a.shape) != (a.shape[0], module.in_features) or tuple(b.shape) != (module.out_features, a.shape[0]):
                raise ValueError(f"LoRA shape mismatch at {target}: A={tuple(a.shape)}, B={tuple(b.shape)}")
            _replace_module(model, target, ConvRotLoRALinear(module, a, b, strength))
    return model


def _move_state(state, device: str):
    """Move tensor values carried between ModularPipeline stages."""
    for name, value in list(state.values.items()):
        if isinstance(value, torch.Tensor):
            state.values[name] = value.to(device)
    return state


class Flux3ConvRotPipeline:
    """Staged Diffusers pipeline for BFL's distilled ConvRot INT8 release.

    Prompt encoding stays on CPU. The transformer and VAE execute on CUDA,
    while explicit state transfers avoid loading the 9 GB text encoder into
    the constrained GPU address space.
    """

    def __init__(
        self,
        model_dir: str | Path,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        lora_checkpoint: str | Path | None = None,
        lora_strength: float = 1.5,
    ):
        from diffusers import ModularPipeline

        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        modular = ModularPipeline.from_pretrained(self.model_dir, local_files_only=True)
        blocks = modular._blocks.sub_blocks

        self.text_pipe = blocks["text_encoder"].init_pipeline(self.model_dir)
        self.text_pipe.load_components(dtype=dtype, local_files_only=True)

        self.transformer = load_flux3_convrot_transformer(
            checkpoint, self.model_dir / "transformer", compute_dtype=dtype
        ).to(device)
        if lora_checkpoint is not None:
            load_flux3_outpaint_lora(self.transformer, lora_checkpoint, lora_strength)
        self.denoise_pipe = blocks["denoise"].init_pipeline(self.model_dir)
        self.denoise_pipe.register_components(transformer=self.transformer)
        self.denoise_pipe.load_components(local_files_only=True)
        self.denoise_pipe.to(device)

        self.vae_encode_pipe = blocks["vae_encoder"].init_pipeline(self.model_dir)
        self.vae_encode_pipe.load_components(dtype=dtype, local_files_only=True)
        self.vae_encode_pipe.to(device)
        self.decode_pipe = blocks["decode"].init_pipeline(self.model_dir)
        self.decode_pipe.load_components(dtype=dtype, local_files_only=True)
        self.decode_pipe.to(device)
        # Tiled VAE above 1024 px: at 4 MP the untiled decode peaks at ~15 GiB (the denoiser at ~8 GiB);
        # tiles of 1024 px with 25% overlap bring it to ~8.3 GiB with no visible seams. Below 1024 px
        # nothing changes (tiling only kicks in for larger images).
        self.vae_encode_pipe.vae.enable_tiling()
        self.decode_pipe.vae.enable_tiling()

    def __call__(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        generator: torch.Generator,
        image=None,
        output: str = "images",
        output_type: str = "pil",
        shift: float | None = None,
    ):
        if shift is not None:
            # The constant timestep shift ("alpha"). Diffusers' FLUX.3 blocks default to 6.93; BFL's reference
            # code and ComfyUI workflows use 16 since 2026-10-17.
            self.denoise_pipe.scheduler.set_shift(float(shift))
        state = self.text_pipe(prompt=prompt)
        state = _move_state(state, self.device)
        if image is not None:
            state = self.vae_encode_pipe(state=state, image=image)
        state = self.denoise_pipe(
            state=state,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        result = self.decode_pipe(state=state, output=output, output_type=output_type)
        return result

    def remove_all_hooks(self):
        for pipe in (
            self.text_pipe,
            self.denoise_pipe,
            self.vae_encode_pipe,
            self.decode_pipe,
        ):
            if hasattr(pipe, "remove_all_hooks"):
                pipe.remove_all_hooks()

    def unload(self):
        self.remove_all_hooks()
        self.text_pipe = None
        self.denoise_pipe = None
        self.vae_encode_pipe = None
        self.decode_pipe = None
        self.transformer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
