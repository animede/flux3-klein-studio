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
    t2i = upsampler.build_messages("t2i", "an apple", width=1024, height=768)
    assert t2i[1]["content"].startswith("TARGET CANVAS: 1024:768\n")


def fake_backends(monkeypatch, alive, answers, multimodal=None):
    calls = []

    def model_info(url, timeout):
        return {"id": url.split("//")[1].split("/")[0] + "-model", "multimodal": (multimodal or {}).get(url)} if url in alive else None

    def chat(url, model, messages, timeout):
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
    monkeypatch.setattr(upsampler.local_llm, "choose_device", lambda need_vision: (None, "空きメモリ不足(3.0GiB)"))
    with pytest.raises(upsampler.UpsampleUnavailable) as err:
        upsampler.upsample("an apple")
    assert "空きメモリ不足" in str(err.value)


def test_no_remote_llm_by_default(monkeypatch):
    monkeypatch.delenv("KLEIN_PROMPT_LLM_URLS", raising=False)
    monkeypatch.delenv("KLEIN_PROMPT_LLM_URL", raising=False)
    assert upsampler.remote_urls() == []
