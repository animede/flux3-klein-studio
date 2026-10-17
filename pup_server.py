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

"""Minimal OpenAI-compatible server for the official INT8 prompt upsampler.

upsampler.py starts it on demand (``python pup_server.py --port 7867``) and stops it after
the request, like the local llama.cpp servers. It runs in its own process because the studio
caps its CUDA allocator for the image model, and the 9B upsampler does not fit under that cap.

Only what upsampler.py sends is supported: one system message, one user message whose content
is text or a list of ``image_url`` (data URIs) parts followed by a text part, greedy decoding
with an optional presence penalty.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList

from model_config import PUP_CONFIG_DIR, PUP_INT8_PATH
from pup_int8 import load_pup_int8

MODEL_ID = "flux-3-image-klein-pup-int8"
app = FastAPI(title="FLUX 3 Image Klein prompt upsampler (INT8)")
state: dict = {}
lock = threading.Lock()


class PresencePenalty(LogitsProcessor):
    """vLLM semantics: subtract the penalty once from every token already generated."""

    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len, self.penalty = prompt_len, penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len:]
        if self.penalty and generated.shape[1]:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, generated, True)
            scores = scores - mask.to(scores.dtype) * self.penalty
        return scores


def _content(content):
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith("data:"):
                raise HTTPException(422, "only data: URIs are supported")
            data = base64.b64decode(url.split(",", 1)[1])
            parts.append({"type": "image", "image": Image.open(io.BytesIO(data)).convert("RGB")})
        elif part.get("type") == "text":
            parts.append({"type": "text", "text": part["text"]})
    return parts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "capabilities": ["completion", "multimodal"]}]}


@app.post("/v1/chat/completions")
def chat(body: dict):
    messages = [{"role": m["role"], "content": _content(m["content"])} for m in body["messages"]]
    with lock:
        processor, model = state["processor"], state["model"]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt",
                                               enable_thinking=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        penalty = float(body.get("presence_penalty") or 0.0)
        started = time.monotonic()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=int(body.get("max_tokens") or 3584), do_sample=False,
                                 logits_processor=LogitsProcessorList([PresencePenalty(prompt_len, penalty)]))
        text = processor.decode(ids[0, prompt_len:], skip_special_tokens=True).strip()
    return {"id": f"pup-{int(started)}", "object": "chat.completion", "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_len, "completion_tokens": int(ids.shape[1] - prompt_len)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7867)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    state["processor"] = AutoProcessor.from_pretrained(PUP_CONFIG_DIR)
    state["model"] = load_pup_int8(PUP_INT8_PATH, PUP_CONFIG_DIR).to(args.device)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
