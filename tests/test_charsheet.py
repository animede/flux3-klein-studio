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
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as service
import charsheet
from engine import Cancelled

EIGHT = ['front', 'back', 'left', 'right', 'front_left_45', 'front_right_45', 'back_left_45', 'back_right_45']


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(service, 'UPLOADS', tmp_path)
    monkeypatch.setattr(service, 'model_available', lambda item: True)
    monkeypatch.setattr(service.engine, 'unload', lambda: None)

    def no_llm(*args, **kwargs):
        raise service.upsampler.UpsampleUnavailable('no llm in tests')
    monkeypatch.setattr(service.upsampler, 'upsample', no_llm)
    service.charsheet_jobs.clear()
    with TestClient(service.app) as client:
        yield client


def png(color=(255, 255, 255), size=(600, 800), figure=True):
    image = Image.new('RGB', size, color)
    if figure:  # a dark figure in the middle; the border keeps the background color
        image.paste((40, 40, 90), (size[0] // 3, size[1] // 5, size[0] * 2 // 3, size[1] * 4 // 5))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def wait(client, job_id):
    for _ in range(300):
        state = client.get(f'/api/charsheet/jobs/{job_id}').json()
        if state['status'] in ('done', 'error', 'cancelled') and not service.engine.lock.locked():
            return state
        time.sleep(.01)
    pytest.fail('charsheet job did not finish')


def recording_engine(monkeypatch, calls):
    def generate(model, prompt, ref, width, height, seed, progress, cancel, **kwargs):
        calls.append({'prompt': prompt, 'ref': ref, 'size': (width, height), 'seed': seed, **kwargs})
        progress('generating', 16, 'done')
        return Image.new('RGB', (width, height), (len(calls) * 20 % 255, 0, 0)), {'seconds': 1}
    monkeypatch.setattr(service.engine, 'generate', generate)


def test_eight_views_on_a_white_background_skip_prep(client, monkeypatch):
    calls = []
    recording_engine(monkeypatch, calls)
    r = client.post('/api/charsheet/generate', files={'image': ('c.png', png())}, data={'seed': 5})
    assert r.status_code == 202
    body = r.json()
    assert body['total'] == 8 and body['prep_applied'] is False and body['prep_status'] == 'skipped'
    assert [v['key'] for v in body['views']] == EIGHT
    state = wait(client, body['job_id'])
    assert state['status'] == 'done' and state['progress'] == 8 and state['error'] is None
    assert state['width'] == 768 and state['height'] == 1024 and state['prep_url'] is None
    assert state['reference_url'] == state['input_url']
    # front is generated too (normalized to the output size), one Edit per view with the same seed
    assert len(calls) == 8 and {c['seed'] for c in calls} == {5} and {c['size'] for c in calls} == {(768, 1024)}
    assert calls[0]['prompt'] == charsheet.view_prompt('front')
    assert all(v['status'] == 'done' and v['url'] == f"/api/charsheet/jobs/{state['job_id']}/images/{v['key']}.png"
               for v in state['views'])
    view = client.get(state['views'][2]['url'])
    assert view.status_code == 200 and Image.open(io.BytesIO(view.content)).size == (768, 1024)
    sheet = Image.open(io.BytesIO(client.get(state['sheet_url']).content))
    assert sheet.width > sheet.height
    names = zipfile.ZipFile(io.BytesIO(client.get(state['zip_url']).content)).namelist()
    assert {f'{k}.png' for k in EIGHT} | {'sheet.png', 'input.png'} <= set(names)
    meta = client.get(state['metadata_url']).json()
    assert set(meta['details']) == set(EIGHT) and 'cancel' not in meta


def test_prep_auto_converts_a_colored_background_and_reuses_it_as_front(client, monkeypatch):
    calls = []
    recording_engine(monkeypatch, calls)
    r = client.post('/api/charsheet/generate', files={'image': ('c.png', png((90, 140, 200)))},
                    data={'views': 'back,front,left', 'width': 512, 'height': 768})
    body = r.json()
    assert body['prep_applied'] and body['prep_status'] == 'queued' and body['white_background_ratio'] < 0.1
    assert [v['key'] for v in body['views']] == ['back', 'front', 'left']
    state = wait(client, body['job_id'])
    assert state['status'] == 'done' and state['prep_status'] == 'done'
    assert state['prep_url'].endswith('/prep.png') and state['reference_url'] == state['prep_url']
    # prep + back + left: front is the prepared image itself
    assert len(calls) == 3 and calls[0]['prompt'] == charsheet.PREP_PROMPT
    prepared = Image.open(io.BytesIO(client.get(state['prep_url']).content))
    front = Image.open(io.BytesIO(client.get(state['views'][1]['url']).content))
    assert prepared.size == front.size == (512, 768) and prepared.getpixel((0, 0)) == front.getpixel((0, 0))
    # the views read the prepared image, not the upload
    assert calls[1]['ref'].getpixel((0, 0)) == prepared.getpixel((0, 0))
    assert 'prep.png' in zipfile.ZipFile(io.BytesIO(client.get(state['zip_url']).content)).namelist()


def test_prep_on_and_off_override_the_detection(client, monkeypatch):
    calls = []
    recording_engine(monkeypatch, calls)
    on = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                  data={'views': 'front', 'prep': 'on'}).json()['job_id'])
    assert on['prep_applied'] and len(calls) == 1 and calls[0]['prompt'] == charsheet.PREP_PROMPT
    off = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png((0, 120, 0)))},
                                   data={'views': 'front', 'prep': 'OFF'}).json()['job_id'])
    assert not off['prep_applied'] and off['prep_status'] == 'skipped' and len(calls) == 2


def test_size_sets_a_square_output(client, monkeypatch):
    calls = []
    recording_engine(monkeypatch, calls)
    state = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                     data={'views': 'left', 'size': 1000}).json()['job_id'])
    assert (state['width'], state['height']) == (992, 992) and calls[0]['size'] == (992, 992)


def test_bad_inputs(client):
    files = {'image': ('c.png', png())}
    for data in ({'views': 'front,up'}, {'prep': 'maybe'}, {'width': 300}, {'seed': -2}, {'upsampler': 'bogus'},
                 {'shift': 0}):
        assert client.post('/api/charsheet/generate', files=files, data=data).status_code == 422, data
    assert client.post('/api/charsheet/generate', data={}).status_code == 422
    assert client.post('/api/charsheet/generate', files={'image': ('x.png', b'not an image')}).status_code == 422
    assert client.post('/api/charsheet/generate',
                       files={'image': ('x.png', b'x' * (service.MAX_UPLOAD + 1))}).status_code == 413
    assert client.get('/api/charsheet/jobs/' + 'a' * 32).status_code == 404
    assert client.get('/api/charsheet/jobs/../app.py').status_code == 404
    assert client.get('/api/charsheet/jobs/' + 'a' * 32 + '/images/up.png').status_code == 404
    assert client.get('/api/charsheet/jobs/' + 'a' * 32 + '/job.json').status_code == 404


def test_busy_cancel_and_recovery(client, monkeypatch):
    calls = []

    def generate(model, prompt, ref, width, height, seed, progress, cancel, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return Image.new('RGB', (width, height)), {'seconds': 1}
        assert cancel.wait(3)
        raise Cancelled()
    monkeypatch.setattr(service.engine, 'generate', generate)
    job = client.post('/api/charsheet/generate', files={'image': ('c.png', png())}).json()['job_id']
    # one job at a time, shared with /api/generate
    assert client.post('/api/charsheet/generate', files={'image': ('c.png', png())}).status_code == 409
    assert client.post('/api/generate', data={'prompt': 'x'}).status_code == 409
    for _ in range(300):
        if len(calls) == 2:
            break
        time.sleep(.01)
    assert client.post(f'/api/charsheet/jobs/{job}/cancel').status_code == 200
    state = wait(client, job)
    assert state['status'] == 'cancelled' and state['sheet_url'] is None
    assert [v['status'] for v in state['views']] == ['done'] + ['cancelled'] * 7
    assert not client.get('/api/health').json()['busy']
    assert client.post(f'/api/charsheet/jobs/{job}/cancel').json()['status'] == 'cancelled'


def test_failure_reports_the_view_and_releases_the_worker(client, monkeypatch):
    def fail(model, prompt, *args, **kwargs):
        if 'rear view' in prompt:
            raise RuntimeError('CUDA out of memory')
        return Image.new('RGB', (8, 8)), {'seconds': 1}
    monkeypatch.setattr(service.engine, 'generate', fail)
    state = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())}).json()['job_id'])
    assert state['status'] == 'error' and state['error'].startswith('back:') and 'メモリ' in state['error']
    assert [v['status'] for v in state['views']][:3] == ['done', 'error', 'queued']
    assert not client.get('/api/health').json()['busy']


def test_upsampler_sees_the_reference_per_view_and_falls_back(client, monkeypatch):
    calls, seen = [], []
    recording_engine(monkeypatch, calls)

    def upsample(request, **kwargs):
        seen.append(kwargs)
        return {'prompt': 'CAPTION ' + request[-20:], 'source': 'remote:x'}
    monkeypatch.setattr(service.upsampler, 'upsample', upsample)
    state = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                     data={'views': 'left,back', 'upsampler': 'local'}).json()['job_id'])
    assert state['status'] == 'done' and state['upsampler'] == 'local' and state['warning'] is None
    assert [k['mode'] for k in seen] == ['i2i', 'i2i'] and all(len(k['images']) == 1 for k in seen)
    assert seen[0]['backend'] == 'local' and all(c['prompt'].startswith('CAPTION') for c in calls)
    assert client.get(state['metadata_url']).json()['details']['left']['upsample_source'] == 'remote:x'

    def unavailable(*args, **kwargs):
        raise service.upsampler.UpsampleUnavailable('no llm')
    monkeypatch.setattr(service.upsampler, 'upsample', unavailable)
    loose = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                     data={'views': 'left'}).json()['job_id'])
    assert loose['status'] == 'done' and 'no llm' in loose['warning']
    strict = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                      data={'views': 'left', 'upsample_strict': 'true'}).json()['job_id'])
    assert strict['status'] == 'error'
    bypass = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                      data={'views': 'left', 'upsample': 'false'}).json()['job_id'])
    assert bypass['status'] == 'done' and calls[-1]['prompt'] == charsheet.view_prompt('left')


def test_state_survives_a_restart(client, monkeypatch):
    recording_engine(monkeypatch, [])
    state = wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                                     data={'views': 'front'}).json()['job_id'])
    service.charsheet_jobs.clear()
    restored = client.get(f"/api/charsheet/jobs/{state['job_id']}").json()
    assert restored['status'] == 'done' and restored['views'] == state['views']
    # a job that was running when the server stopped reads as an error
    path = service.charsheet_root() / state['job_id'] / 'job.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), 'status': 'running'}))
    assert client.get(f"/api/charsheet/jobs/{state['job_id']}").json()['status'] == 'error'


def test_charsheet_outputs_stay_out_of_the_gallery(client, monkeypatch):
    recording_engine(monkeypatch, [])
    wait(client, client.post('/api/charsheet/generate', files={'image': ('c.png', png())},
                             data={'views': 'front'}).json()['job_id'])
    assert client.get('/api/gallery').json()['total'] == 0


def test_view_definitions_and_directions():
    assert [v['key'] for v in charsheet.VIEWS] == EIGHT
    assert sorted(k for row in charsheet.SHEET_LAYOUT for k in row) == sorted(EIGHT)
    # `left`: the character turns to their own left, so the nose points to the right edge of the image
    assert 'their own left' in charsheet.VIEW_BY_KEY['left']['prompt']
    assert 'right edge of the image' in charsheet.VIEW_BY_KEY['left']['prompt']
    assert 'left edge of the image' in charsheet.VIEW_BY_KEY['right']['prompt']
    assert charsheet.parse_views(' left , left,back ') == ['left', 'back']
    assert charsheet.parse_views('') == EIGHT
    with pytest.raises(ValueError):
        charsheet.parse_views('front,top')


def test_white_background_detection():
    assert charsheet.has_white_background(Image.open(io.BytesIO(png())))
    assert charsheet.has_white_background(Image.open(io.BytesIO(png((246, 244, 248)))))  # studio off-white
    assert not charsheet.has_white_background(Image.open(io.BytesIO(png((200, 180, 230)))))  # lavender
    assert not charsheet.has_white_background(Image.open(io.BytesIO(png((30, 30, 30)))))
