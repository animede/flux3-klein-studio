/*
 * Copyright 2026 YUZUKI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
const $ = id => document.getElementById(id);
let activeJob = null, lastSeed = null, referenceURLs = [], polling = false;
// Caption produced by the "展開して確認" button; sending it unchanged skips a second upsampling pass.
let lastUpsampled = null, upsamplerNote = '';

function showStatus(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function busy(value) {
  $('generate').disabled = value;
  $('unload').disabled = value;
  $('model').disabled = value;
  $('cancel').hidden = !value;
  $('cancel').disabled = false;
  $('progress-track').hidden = !value;
}
const modelInfo = {
  'flux3': { label: 'FLUX 3 Image Klein · 最新世代', badge: 'LOCAL · FLUX 3 · 5.6B', note: '公式INT8 ConvRot · 16ステップ · FLUX Kommunity License v1.0', footer: 'FLUX 3 Image Klein · 公式INT8 ConvRot · 16ステップ · FLUX Kommunity License v1.0', status: 'FLUX 3を選択しました。短文ではなく、構図・被写体・背景・照明まで含む詳しいプロンプトを推奨します。' },
};
function updateModelInfo() {
  const key = $('model').value;
  const info = modelInfo[key];
  $('model-badge').textContent = info.badge;
  $('model-note').textContent = info.note;
  $('model-note').classList.add('license-warning');
  $('model-footer').textContent = info.footer;
  localStorage.setItem('klein-model', key);
  showStatus(info.status);
}
const savedModel = localStorage.getItem('klein-model');
$('model').value = 'flux3';
$('model').addEventListener('change', updateModelInfo);
updateModelInfo();
async function loadModelAvailability() {
  const data = await request('/api/models');
  for (const option of $('model').options) {
    const available = data.models[option.value]?.available;
    option.disabled = !available;
    option.textContent = `${modelInfo[option.value].label}${available ? '' : '（未取得）'}`;
  }
  if ($('model').selectedOptions[0].disabled) {
    const available = [...$('model').options].find(option => !option.disabled);
    if (available) $('model').value = available.value;
  }
  updateModelInfo();
}
loadModelAvailability().catch(() => showStatus('モデルの準備状態を確認できませんでした。', true));
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'リクエストを処理できませんでした。');
  return data;
}
function preview() {
  referenceURLs.forEach(url => URL.revokeObjectURL(url));
  referenceURLs = [];
  const files = [...$('image-input').files];
  const previews = $('reference-previews');
  previews.replaceChildren();
  previews.dataset.count = files.length;
  previews.hidden = !files.length;
  $('upload-empty').hidden = !!files.length;
  $('clear-image').hidden = !files.length;
  $('reference-name').textContent = files.length
    ? `${files.length}枚選択 · ${files.map(file => file.name).join(' / ')}`
    : '画像なしの場合はテキストから生成します';
  for (const [index, file] of files.entries()) {
    const url = URL.createObjectURL(file);
    referenceURLs.push(url);
    const item = document.createElement('div');
    item.className = 'reference-item';
    const img = document.createElement('img');
    img.src = url; img.alt = `参照画像 ${index + 1}`;
    const number = document.createElement('span');
    number.textContent = index + 1;
    item.append(img, number); previews.append(item);
  }
}
$('image-input').addEventListener('change', () => {
  if ($('image-input').files.length > 8) {
    $('image-input').value = '';
    preview(); showStatus('参照画像は8枚まで選択できます。', true); return;
  }
  preview(); showStatus('参照画像を選択しました。');
});
$('clear-image').onclick = () => { $('image-input').value = ''; preview(); };
for (const event of ['dragenter', 'dragover']) $('drop-zone').addEventListener(event, () => $('drop-zone').classList.add('drag'));
for (const event of ['dragleave', 'drop']) $('drop-zone').addEventListener(event, () => $('drop-zone').classList.remove('drag'));
$('example-prompt').onclick = () => {
  $('prompt').value = 'Create a new image of the same character from the reference image, sitting by a cafe window and reading a book. Preserve the character\'s facial features, hairstyle, hair color, eye color, distinctive accessories, and art style. Change the pose and setting. Soft afternoon light.';
  $('prompt').focus();
};

async function poll(id) {
  if (polling) return;
  polling = true;
  let failures = 0;
  try {
    while (activeJob === id) {
      let job;
      try { job = await request(`/api/jobs/${id}`); failures = 0; }
      catch (error) {
        failures++;
        if (failures >= 5) throw error;
        showStatus('接続を再確認しています…');
        await new Promise(r => setTimeout(r, 2000)); continue;
      }
      showStatus(job.warning ? `${job.message}（${job.warning}）` : job.message, job.status === 'error' || !!job.warning);
      $('result-state').textContent = job.status.toUpperCase();
      $('progress-fill').style.width = `${job.step / job.steps * 100}%`;
      if (job.status === 'done') {
        $('result-image').src = job.image_url;
        $('result-image').hidden = false;
        $('result-empty').hidden = true;
        $('result-actions').hidden = false;
        $('download').href = job.image_url;
        $('metadata').href = job.metadata_url;
        $('metadata').download = 'generation.json';
        $('metadata').hidden = false;
        lastSeed = job.seed;
        const m = job.metrics;
        const quant = '公式INT8 ConvRot · FLUX Kommunity License v1.0';
        const upsampled = `prompt upsampler：${job.upsample_source || (job.upsample ? '使えず元の文で生成' : (upsamplerNote || 'バイパス'))}\n`;
        $('metrics').textContent = `モデル：${job.model_name}\nシード：${job.seed}\n${upsampled}生成時間：${m.seconds}秒（読み込み込み ${m.total_seconds}秒）\nピークVRAM：割り当て ${m.peak_allocated_gib} GiB / 予約 ${m.peak_reserved_gib} GiB\nGPU：${m.gpu}\nPyTorch割り当て上限：${m.allocator_budget_gib} GiB\n${m.steps}ステップ · ${quant}\n※VRAM値はPyTorch計測。ドライバー等の使用量は含みません。`;
      }
      if (['done', 'error', 'cancelled'].includes(job.status)) {
        activeJob = null; localStorage.removeItem('klein-job'); busy(false); break;
      }
      await new Promise(r => setTimeout(r, 1000));
    }
  } catch (error) {
    showStatus(`${error.message} ページを再読み込みすると進行状況を再確認できます。`, true);
    activeJob = null; busy(false);
  } finally { polling = false; }
}

$('generate-form').addEventListener('submit', async event => {
  event.preventDefault(); busy(true); $('progress-fill').style.width = '0%';
  $('result-state').textContent = 'PREPARING'; showStatus('生成をリクエストしています');
  const [width, height] = $('size').value.split(',');
  const data = new FormData();
  data.set('prompt', $('prompt').value); data.set('model', $('model').value); data.set('width', width); data.set('height', height);
  data.set('seed', $('seed').value);
  data.set('outpaint', $('outpaint').checked ? 'true' : 'false');
  data.set('boxes', $('boxes').value);
  const preExpanded = $('auto-upsample').checked && lastUpsampled !== null && $('prompt').value === lastUpsampled;
  upsamplerNote = preExpanded ? '展開済みのキャプションを使用' : 'バイパス（入力文をそのまま使用）';
  data.set('upsample', $('auto-upsample').checked && !preExpanded ? 'true' : 'false');
  for (const file of $('image-input').files) data.append('image', file);
  try {
    const job = await request('/api/generate', { method: 'POST', body: data });
    activeJob = job.id; localStorage.setItem('klein-job', job.id); await poll(job.id);
  } catch (error) { showStatus(error.message, true); busy(false); $('result-state').textContent = 'ERROR'; }
});
$('expand-prompt').onclick = () => {
  const field = $('prompt');
  const text = field.value.trim();
  if (!text) return;
  field.value = `${text}. Full scene caption: describe the subject's appearance, exact position and scale in the frame, environment and background, lighting direction and quality, camera angle and lens perspective, color palette, materials, textures, depth of field, and the continuity of all reference images. Preserve important identity and composition details.`;
  field.focus();
};
$('llm-prompt').onclick = async () => {
  const field = $('prompt');
  if (!field.value.trim()) return;
  const button = $('llm-prompt'); button.disabled = true; button.textContent = 'prompt upsamplerで展開中…';
  try {
    // 編集(参照画像あり)では画像も LLM に見せる(公式 i2i 形式は参照画像の記述に編集内容を織り込むため)。
    const data = new FormData(); data.set('prompt', field.value); data.set('mode', 'auto');
    const [w, h] = $('size').value.split(','); data.set('width', w); data.set('height', h);
    for (const file of $('image-input').files) data.append('image', file);
    const result = await request('/api/enhance-prompt', {method: 'POST', body: data});
    field.value = result.prompt; lastUpsampled = result.prompt;
    showStatus(`prompt upsamplerで展開しました（${result.source}${result.cached ? '・キャッシュ' : ''}）。このまま生成すると、もう一度は展開しません`);
  } catch (error) { showStatus(error.message, true); }
  finally { button.disabled = false; button.textContent = '✦ prompt upsamplerで展開して確認'; }
};
const boxCanvas = $('box-canvas');
const boxCtx = boxCanvas.getContext('2d');
let boxImage = null; let drawnBoxes = []; let boxStart = null;
function paintBoxes() {
  boxCtx.fillStyle = '#111'; boxCtx.fillRect(0, 0, boxCanvas.width, boxCanvas.height);
  if (boxImage) { const scale = Math.min(boxCanvas.width / boxImage.width, boxCanvas.height / boxImage.height); const w = boxImage.width * scale, h = boxImage.height * scale; boxCtx.drawImage(boxImage, (boxCanvas.width-w)/2, (boxCanvas.height-h)/2, w, h); }
  drawnBoxes.forEach((b, i) => { const x=b.bbox[1]*boxCanvas.width/1000, y=b.bbox[0]*boxCanvas.height/1000, w=(b.bbox[3]-b.bbox[1])*boxCanvas.width/1000, h=(b.bbox[2]-b.bbox[0])*boxCanvas.height/1000; boxCtx.strokeStyle='#c0efbb'; boxCtx.lineWidth=2; boxCtx.strokeRect(x,y,w,h); boxCtx.fillStyle='#c0efbb'; boxCtx.fillText(b.id || `box_${i+1}`, x+4, y+14); });
}
function syncBoxes() { $('boxes').value = JSON.stringify(drawnBoxes, null, 2); paintBoxes(); }
$('image-input').addEventListener('change', () => { const file=$('image-input').files[0]; if (!file) return; const img=new Image(); img.onload=()=>{boxImage=img;paintBoxes();}; img.src=URL.createObjectURL(file); });
boxCanvas.addEventListener('pointerdown', e => { const r=boxCanvas.getBoundingClientRect(); boxStart=[(e.clientX-r.left)*boxCanvas.width/r.width,(e.clientY-r.top)*boxCanvas.height/r.height]; boxCanvas.setPointerCapture(e.pointerId); });
boxCanvas.addEventListener('pointerup', e => { if (!boxStart) return; const r=boxCanvas.getBoundingClientRect(); const x=(e.clientX-r.left)*boxCanvas.width/r.width,y=(e.clientY-r.top)*boxCanvas.height/r.height; const x1=Math.max(0,Math.min(boxStart[0],x))*1000/boxCanvas.width,y1=Math.max(0,Math.min(boxStart[1],y))*1000/boxCanvas.height,x2=Math.min(boxCanvas.width,Math.max(boxStart[0],x))*1000/boxCanvas.width,y2=Math.min(boxCanvas.height,Math.max(boxStart[1],y))*1000/boxCanvas.height; drawnBoxes.push({id:`box_${drawnBoxes.length+1}`,bbox:[Math.round(y1),Math.round(x1),Math.round(y2),Math.round(x2)],desc:'配置要素'}); boxStart=null; syncBoxes(); });
$('clear-boxes').onclick = () => { drawnBoxes=[]; syncBoxes(); };
$('cancel').onclick = async () => {
  if (!activeJob) return;
  $('cancel').disabled = true;
  try { const job = await request(`/api/jobs/${activeJob}/cancel`, { method: 'POST' }); showStatus(job.message); }
  catch (error) { showStatus(error.message, true); }
};
$('reuse-seed').onclick = () => { if (lastSeed !== null) $('seed').value = lastSeed; };
$('unload').onclick = async () => {
  $('unload').disabled = true;
  try { await request('/api/unload', { method: 'POST' }); showStatus('モデルを解放しました。次の生成時に再読み込みします。'); }
  catch (error) { showStatus(error.message, true); }
  finally { $('unload').disabled = false; }
};
const savedJob = localStorage.getItem('klein-job');
if (savedJob) { activeJob = savedJob; busy(true); poll(savedJob); }
