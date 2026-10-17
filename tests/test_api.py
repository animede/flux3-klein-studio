import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import app as service
from engine import Cancelled


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(service, 'UPLOADS', tmp_path)
    monkeypatch.setattr(service, 'model_available', lambda item: True)
    service.jobs.clear()
    with TestClient(service.app) as client:
        yield client


def wait_job(client, job):
    for _ in range(100):
        state = client.get('/api/jobs/' + job).json()
        if state['status'] in ('done', 'error', 'cancelled') and not service.engine.lock.locked():
            return state
        time.sleep(.01)
    pytest.fail('job did not finish')


def png_bytes(color=(0, 0, 0, 0), size=(1024, 512)):
    image = io.BytesIO()
    Image.new('RGBA', size, color).save(image, format='PNG')
    return image.getvalue()


def test_bad_inputs(client):
    for data in ({'prompt': ' '}, {'prompt': 'hello', 'width': 257}, {'prompt': 'hello', 'seed': -2}, {'prompt': 'hello', 'model': 'bad'}):
        assert client.post('/api/generate', data=data).status_code == 422
    assert client.post('/api/generate', data={'prompt': 'hello'}, files={'image': ('bad.png', b'not an image')}).status_code == 422
    assert client.post('/api/generate', data={'prompt': 'hello'}, files={'image': ('huge.png', b'x' * (service.MAX_UPLOAD + 1))}).status_code == 413
    assert client.get('/api/jobs/missing').status_code == 404


def test_reference_and_result(client, monkeypatch):
    def generate(model, prompt, ref, width, height, seed, progress, cancel):
        assert model == 'flux3'
        assert ref.mode == 'RGB' and ref.size == (768, 384)
        assert ref.getpixel((0, 0)) == (255, 255, 255)
        assert seed == 12
        return Image.new('RGB', (width, height)), {'seconds': 1}
    monkeypatch.setattr(service.engine, 'generate', generate)
    r = client.post('/api/generate', data={'prompt': 'test', 'seed': 12}, files={'image': ('ref.png', png_bytes())})
    assert r.status_code == 202
    state = wait_job(client, r.json()['id'])
    assert state['status'] == 'done'
    assert (service.OUTPUTS / (state['id'] + '.json')).exists()
    assert 'cancel' not in state


def test_eight_references_adaptive_size_and_limit(client, monkeypatch):
    def generate(model, prompt, refs, width, height, seed, progress, cancel):
        assert model == 'flux3'
        assert isinstance(refs, list) and len(refs) == 8
        assert all(ref.mode == 'RGB' for ref in refs)
        assert all(ref.size == (768, 384) for ref in refs)
        return Image.new('RGB', (width, height)), {'seconds': 1}
    monkeypatch.setattr(service.engine, 'generate', generate)
    files = [('image', (f'ref-{index}.png', png_bytes((index, 0, 0, 255)), 'image/png')) for index in range(8)]
    response = client.post('/api/generate', data={'prompt': 'combine eight'}, files=files)
    state = wait_job(client, response.json()['id'])
    assert state['status'] == 'done' and state['reference_count'] == 8
    assert state['reference_max_side'] == 768
    metadata = (service.OUTPUTS / (state['id'] + '.json')).read_text()
    assert '"reference_count": 8' in metadata
    assert '"reference_max_side": 768' in metadata
    assert all((service.UPLOADS / f"{state['id']}_ref_{index}.png").exists() for index in range(1, 9))

    nine_files = [('image', (f'ref-{index}.png', png_bytes(), 'image/png')) for index in range(9)]
    too_many = client.post('/api/generate', data={'prompt': 'too many'}, files=nine_files)
    assert too_many.status_code == 422
    assert '8枚まで' in too_many.json()['detail']


def test_flux3_uses_its_step_count(client, monkeypatch):
    def generate(model, prompt, refs, width, height, seed, progress, cancel):
        assert model == 'flux3'
        return Image.new('RGB', (width, height)), {'seconds': 1, 'steps': 16}
    monkeypatch.setattr(service.engine, 'generate', generate)
    response = client.post('/api/generate', data={'prompt': 'a detailed scene', 'model': 'flux3'})
    assert response.status_code == 202
    assert response.json()['steps'] == 16
    state = wait_job(client, response.json()['id'])
    assert state['status'] == 'done'
    assert state['step'] == state['steps'] == 16


def test_busy_cancel_and_recovery(client, monkeypatch):
    def generate(model, prompt, ref, width, height, seed, progress, cancel):
        assert cancel.wait(3)
        raise Cancelled()
    monkeypatch.setattr(service.engine, 'generate', generate)
    monkeypatch.setattr(service.engine, 'unload', lambda: None)
    r = client.post('/api/generate', data={'prompt': 'test'}).json()
    assert client.post('/api/generate', data={'prompt': 'second'}).status_code == 409
    assert client.post('/api/unload').status_code == 409
    assert client.post('/api/jobs/' + r['id'] + '/cancel').status_code == 200
    assert wait_job(client, r['id'])['status'] == 'cancelled'
    assert client.post('/api/unload').status_code == 200


def test_failure_releases_worker(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(service.engine, 'generate', fail)
    monkeypatch.setattr(service.engine, 'unload', lambda: None)
    r = client.post('/api/generate', data={'prompt': 'test'}).json()
    state = wait_job(client, r['id'])
    assert state['status'] == 'error' and 'メモリ' in state['message']
    assert not client.get('/api/health').json()['busy']


def test_health_lists_flux3(client):
    health = client.get('/api/health').json()
    assert health['default_model'] == 'flux3'
    assert set(health['models']) == {'flux3'}
    assert health['models']['flux3']['available']
    models = client.get('/api/models').json()
    assert models['default'] == 'flux3'
    assert models['models']['flux3']['quantization'] == 'Official INT8 ConvRot'
    assert models['models']['flux3']['available']
