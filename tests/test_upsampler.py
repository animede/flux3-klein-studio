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

"""upsampler.py (prompt upsampler with official system prompts and fallbacks)."""
import io

import httpx
import pytest
from PIL import Image

import upsampler


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    upsampler._cache.clear()
    monkeypatch.setattr(upsampler, "_system_prompts", {"t2i": "SYS-T2I", "i2i": "SYS-I2I"})
    monkeypatch.setattr(upsampler, "_external_prompts", {"t2i": "EXT-T2I", "i2i": "EXT-I2I"})
    monkeypatch.delenv("KLEIN_LLM_PROMPTS", raising=False)
    monkeypatch.setenv("KLEIN_PROMPT_LLM_URLS", "http://big/v1,http://small/v1")
    monkeypatch.delenv("KLEIN_PROMPT_LLM_MODEL", raising=False)
    monkeypatch.setenv("KLEIN_LOCAL_LLM", "0")


def png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


def test_mode_resolution():
    assert upsampler.resolve_mode("auto", 0) == "t2i"
    assert upsampler.resolve_mode("auto", 2) == "i2i"
    assert upsampler.resolve_mode("i2i_bbox", 0) == "i2i_bbox"
    with pytest.raises(ValueError):
        upsampler.resolve_mode("bogus", 0)


def test_edit_messages_put_images_before_text_and_t2i_gets_canvas():
    msgs = upsampler.build_messages("i2i", "remove the bag", [png(), png()])
    assert msgs[0]["content"] == "SYS-I2I"
    parts = msgs[1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    # Official scaffold and JPEG references, as in the official inference code.
    assert parts[2]["text"] == ("Reference images follow in order (the first image, the second image, ...).\n"
                                "User request: remove the bag\nOutput the caption.")
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    t2i = upsampler.build_messages("t2i", "an apple", width=1024, height=768)
    assert t2i[1]["content"].startswith("TARGET CANVAS: 1024:768\n")


def fake_backends(monkeypatch, alive, answers, multimodal=None):
    calls = []

    def model_info(url, timeout):
        return {"id": url.split("//")[1].split("/")[0] + "-model", "multimodal": (multimodal or {}).get(url)} if url in alive else None

    def chat(url, model, messages, timeout, mode="t2i"):
        calls.append(url)
        if isinstance(answers.get(url), Exception):
            raise answers[url]
        return answers[url]

    monkeypatch.setattr(upsampler, "_model_info", model_info)
    monkeypatch.setattr(upsampler, "_chat", chat)
    return calls


def test_first_server_down_falls_back_to_second(monkeypatch):
    calls = fake_backends(monkeypatch, alive={"http://small/v1"}, answers={"http://small/v1": "A full caption."})
    out = upsampler.upsample("an apple")
    assert out["prompt"] == "A full caption." and out["source"] == "remote:http://small/v1"
    assert out["model"] == "small-model" and calls == ["http://small/v1"]


def test_text_only_server_is_skipped_for_edits(monkeypatch):
    fake_backends(monkeypatch, alive={"http://big/v1", "http://small/v1"},
                  answers={"http://small/v1": "edit caption"}, multimodal={"http://big/v1": False})
    out = upsampler.upsample("remove the bag", images=[png()])
    assert out["source"] == "remote:http://small/v1" and out["mode"] == "i2i"


def test_everything_down_raises_with_reasons(monkeypatch):
    fake_backends(monkeypatch, alive={"http://big/v1"}, answers={"http://big/v1": httpx.ConnectError("x")})
    with pytest.raises(upsampler.UpsampleUnavailable) as err:
        upsampler.upsample("an apple")
    assert "http://big/v1" in str(err.value) and "http://small/v1" in str(err.value)


def test_results_are_cached(monkeypatch):
    calls = fake_backends(monkeypatch, alive={"http://big/v1"}, answers={"http://big/v1": "cap"})
    first = upsampler.upsample("an apple", width=512, height=512)
    second = upsampler.upsample("an apple", width=512, height=512)
    assert not first["cached"] and second["cached"] and calls == ["http://big/v1"]


def test_local_fallback_reports_why_it_is_unavailable(monkeypatch):
    monkeypatch.setenv("KLEIN_LOCAL_LLM", "1")
    fake_backends(monkeypatch, alive=set(), answers={})
    monkeypatch.setattr(upsampler.local_llm, "choose_device", lambda need_vision, release_vram=None: (None, "空きメモリ不足(3.0GiB)"))
    with pytest.raises(upsampler.UpsampleUnavailable) as err:
        upsampler.upsample("an apple")
    assert "空きメモリ不足" in str(err.value)


def test_no_remote_llm_by_default(monkeypatch):
    monkeypatch.delenv("KLEIN_PROMPT_LLM_URLS", raising=False)
    monkeypatch.delenv("KLEIN_PROMPT_LLM_URL", raising=False)
    assert upsampler.remote_urls() == []


def test_reference_size_follows_the_official_budget():
    assert upsampler.reference_size(768, 768, 1) == (768, 768)
    w, h = upsampler.reference_size(2048, 1536, 1)
    assert w * h <= 1024 * 1024 and w % 32 == 0 and h % 32 == 0
    w, h = upsampler.reference_size(1024, 768, 2)
    assert w * h <= 724 * 724
    w, h = upsampler.reference_size(1024, 768, 3)
    assert w * h <= 512 * 512


def test_backend_choice_is_validated_and_defaults_from_env(monkeypatch):
    monkeypatch.delenv("KLEIN_UPSAMPLER", raising=False)
    assert upsampler.resolve_backend(None) == "auto"
    monkeypatch.setenv("KLEIN_UPSAMPLER", "pup-gguf")
    assert upsampler.resolve_backend(None) == "pup-gguf" and upsampler.resolve_backend("remote") == "remote"
    with pytest.raises(ValueError):
        upsampler.resolve_backend("bogus")


def test_official_backend_skips_remote_and_reports_its_source(monkeypatch):
    calls = fake_backends(monkeypatch, alive={"http://big/v1"}, answers={"http://big/v1": "remote caption"})
    seen = {}

    def chat(messages, need_vision, mode, release_vram):
        seen["release"] = release_vram
        return "official caption", "pup-int8"
    monkeypatch.setattr(upsampler.pup_int8, "chat", chat)
    release = lambda: None  # noqa: E731
    out = upsampler.upsample("an apple", backend="pup-int8", release_vram=release)
    assert out["prompt"] == "official caption" and out["source"] == "pup-int8" and calls == []
    assert seen["release"] is release
    # Different backends are cached separately.
    assert upsampler.upsample("an apple", backend="remote")["prompt"] == "remote caption"


def test_official_backend_releases_vram_when_short(monkeypatch):
    server = upsampler.PupGGUF()
    monkeypatch.setattr(server, "missing", lambda: None)
    free = {"gib": 3.0}
    monkeypatch.setattr(server, "free_vram_gib", lambda: free["gib"])

    def release():
        free["gib"] = 20.0
    assert server.choose_device(True, release) == ("gpu", "空き VRAM 20.0GiB")
    free["gib"] = 3.0
    device, reason = server.choose_device(True, None)
    assert device is None and "空き VRAM が足りません" in reason


def test_convrot_rows_dequantize_back_to_the_original():
    import torch
    from pup_int8 import _hadamard, dequantize_convrot_rows

    torch.manual_seed(0)
    weight = torch.randn(8, 512)
    rotated = (weight.reshape(8, 2, 256) @ _hadamard(256, "cpu").T).reshape(8, 512)  # W @ H^T per group
    scale = rotated.abs().amax(dim=1, keepdim=True) / 127
    q = torch.round(rotated / scale).to(torch.int8)
    restored = dequantize_convrot_rows(q, scale, 256, torch.float32)
    assert torch.allclose(restored, weight, atol=0.05)


def test_external_llms_get_bfl_external_prompts_with_the_request_first():
    msgs = upsampler.build_messages("i2i", "remove the bag", [png(), png()], style="external")
    assert msgs[0]["content"] == "EXT-I2I"
    parts = msgs[1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[0]["text"] == "User request: remove the bag"
    t2i = upsampler.build_messages("t2i", "an apple", width=1024, height=768, style="external")
    assert t2i[0]["content"] == "EXT-T2I" and t2i[1]["content"].startswith("TARGET CANVAS: 1024:768\n")


def test_remote_gets_external_prompts_and_official_gets_its_own(monkeypatch):
    seen = {}

    def model_info(url, timeout):
        return {"id": "m", "multimodal": True}

    def chat(url, model, messages, timeout, mode="t2i"):
        seen["remote"] = messages[0]["content"]
        return "remote caption"
    monkeypatch.setattr(upsampler, "_model_info", model_info)
    monkeypatch.setattr(upsampler, "_chat", chat)
    upsampler.upsample("an apple", backend="remote")
    assert seen["remote"] == "EXT-T2I"

    def official(messages, need_vision, mode, release_vram):
        seen["official"] = messages[0]["content"]
        return "official caption", "pup-gguf"
    monkeypatch.setattr(upsampler.pup_gguf, "chat", official)
    upsampler.upsample("an apple", backend="pup-gguf")
    assert seen["official"] == "SYS-T2I"


def test_llm_prompt_style_can_be_switched_back(monkeypatch):
    monkeypatch.setenv("KLEIN_LLM_PROMPTS", "pup")
    assert upsampler.llm_prompt_style() == "pup"
    monkeypatch.setenv("KLEIN_LLM_PROMPTS", "external")
    assert upsampler.llm_prompt_style() == "external"
