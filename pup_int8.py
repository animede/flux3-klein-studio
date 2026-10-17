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

"""Load the official ConvRot INT8 prompt upsampler (Comfy format) into Transformers.

BFL publishes ``qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors`` in
flux-3-image-klein-comfy. Its tensor names already match
``Qwen3_5ForConditionalGeneration``; every quantized tensor is ``int8_tensorwise``
with a ConvRot rotation (group size 256 in the language model, 16/64 in the vision
tower). Linears run on comfy-kitchen's INT8 kernel as in flux3_int8.py; the token
embedding is quantized too, so its rows are dequantized and rotated back on lookup.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn

from flux3_int8 import ConvRotInt8Linear, _replace_module


def _hadamard(size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix used by ConvRot (symmetric and orthogonal)."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    return h / size**0.5


def dequantize_convrot_rows(weight: torch.Tensor, scale: torch.Tensor, group_size: int,
                            dtype: torch.dtype) -> torch.Tensor:
    """INT8 rows stored as W @ H^T (per input group) -> W. H is symmetric and orthogonal, so W = W_rot @ H."""
    rows, features = weight.shape
    rotated = (weight.float() * scale.float().reshape(rows, 1)).reshape(rows, features // group_size, group_size)
    return (rotated @ _hadamard(group_size, weight.device)).reshape(rows, features).to(dtype)


class ConvRotInt8Embedding(nn.Module):
    """Token embedding kept in INT8; only the looked-up rows are dequantized."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, *, compute_dtype: torch.dtype,
                 convrot_groupsize: int, padding_idx: int | None = None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.compute_dtype = compute_dtype
        self.convrot_groupsize = convrot_groupsize
        self.padding_idx = padding_idx
        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("weight_scale", weight_scale.float().reshape(-1), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        rows = dequantize_convrot_rows(self.weight[flat], self.weight_scale[flat], self.convrot_groupsize,
                                       self.compute_dtype)
        return rows.reshape(*input_ids.shape, self.embedding_dim)


def load_pup_int8(checkpoint: str | Path, config_dir: str | Path, *,
                  compute_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Build the upsampler on CPU with INT8 weights. Move it with ``.to("cuda")`` (no dtype)."""
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(config_dir)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration._from_config(config, dtype=compute_dtype)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        # The multi-token-prediction head is for speculative decoding; Transformers does not use it.
        keys = {k for k in keys if not k.startswith("mtp.")}
        quantized = sorted(k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant"))
        consumed = set()
        for base in quantized:
            quant = json.loads(bytes(tensors.get_tensor(base + ".comfy_quant").tolist()))
            if quant.get("format") != "int8_tensorwise" or not quant.get("convrot"):
                raise ValueError(f"Unsupported quantization for {base}: {quant}")
            group = int(quant.get("convrot_groupsize", 256))
            weight = tensors.get_tensor(base + ".weight").contiguous()
            scale = tensors.get_tensor(base + ".weight_scale").contiguous()
            consumed |= {base + ".comfy_quant", base + ".weight", base + ".weight_scale"}
            old = model.get_submodule(base)
            if isinstance(old, nn.Linear):
                bias = None
                if base + ".bias" in keys:
                    bias = tensors.get_tensor(base + ".bias").to(compute_dtype)
                    consumed.add(base + ".bias")
                if tuple(weight.shape) != (old.out_features, old.in_features):
                    raise ValueError(f"Shape mismatch for {base}: {tuple(weight.shape)}")
                new = ConvRotInt8Linear(old.in_features, old.out_features, weight, scale,
                                        compute_dtype=compute_dtype, convrot_groupsize=group, bias=bias)
            elif isinstance(old, nn.Embedding):
                new = ConvRotInt8Embedding(weight, scale, compute_dtype=compute_dtype, convrot_groupsize=group,
                                           padding_idx=old.padding_idx)
            else:
                raise TypeError(f"Unexpected module at {base}: {type(old).__name__}")
            if "." in base:
                _replace_module(model, base, new)
            else:
                setattr(model, base, new)  # lm_head

        for key in sorted(keys - consumed):
            set_module_tensor_to_device(model, key, "cpu", value=tensors.get_tensor(key), dtype=compute_dtype)

    meta = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if meta:
        raise ValueError(f"Upsampler loader left meta tensors: {meta[:10]}")
    model.eval()
    return model
