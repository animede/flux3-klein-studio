# API リファレンス

日本語 | [English](API_REFERENCE.en.md)

FLUX 3 Image Klein Reference Studio の HTTP API の全エンドポイント、パラメータ、レスポンス、エラーの一覧です。
使い方の流れと呼び出し例は [API 仕様(ガイド)](API.md) を参照してください。

- API バージョン: `0.3.0`(`/openapi.json` の `info.version`)
- ベース URL: `http://127.0.0.1:7864`(`KLEIN_HOST` / `KLEIN_PORT` で変更)
- 認証: なし。既定では `127.0.0.1` だけで待ち受けます。外部に公開する場合は、リバースプロキシなどで認証とアクセス制限を掛けてください。
- 機械可読な仕様: `GET /openapi.json`(OpenAPI 3.1)、`GET /docs`(Swagger UI)、`GET /redoc`(ReDoc)
- 文字コード: UTF-8。エラーメッセージは日本語です。

## 目次

| メソッド | パス | 概要 | 成功時 |
|---|---|---|---|
| `GET` | [`/api/health`](#get-apihealth) | サーバーの状態 | `200` |
| `GET` | [`/api/models`](#get-apimodels) | 利用できるモデルの一覧 | `200` |
| `POST` | [`/api/enhance-prompt`](#post-apienhance-prompt) | 短い依頼を完全なキャプションに展開(prompt upsampler) | `200` |
| `POST` | [`/api/generate`](#post-apigenerate) | 画像生成ジョブを開始 | `202` |
| `GET` | [`/api/jobs/{job_id}`](#get-apijobsjob_id) | ジョブの状態 | `200` |
| `POST` | [`/api/jobs/{job_id}/cancel`](#post-apijobsjob_idcancel) | ジョブの停止を要求 | `200` |
| `POST` | [`/api/unload`](#post-apiunload) | モデルをメモリから解放 | `200` |
| `GET` | [`/outputs/{file}`](#get-outputsfile) | 生成した PNG とメタデータ JSON | `200` |
| `GET` | `/` | Web 画面 | `200` |

## 基本の流れ

1. `POST /api/generate` でジョブを作る(`202` と `id` が返る)
2. `GET /api/jobs/{id}` を 1 秒程度の間隔で、`status` が `done` / `error` / `cancelled` のどれかになるまで呼ぶ
3. `done` なら `image_url`(PNG)と `metadata_url`(JSON)を取得する

生成は同時に 1 件だけです。実行中に次の `POST /api/generate` を送ると `409` になります。

---

## `GET /api/health`

サーバーの状態、読み込み中のモデル、モデルファイルの準備状態を返します。

### レスポンス `200`: `HealthResponse`

| フィールド | 型 | 内容 |
|---|---|---|
| `status` | string | 常に `"ok"` |
| `loaded` | boolean | モデルがメモリに読み込まれているか |
| `model` | string | 読み込み中(未読み込みなら既定)のモデルの Hugging Face リポジトリ名 |
| `active_model` | string \| null | 読み込み中のモデルの選択値。未読み込みなら `null` |
| `default_model` | string | 既定のモデルの選択値(`"flux3"`) |
| `models` | object | 選択値 → [`ModelInfo`](#modelinfo) |
| `busy` | boolean | 生成中か |
| `memory_budget_gib` | number | PyTorch の GPU 割り当て上限(GiB) |

```json
{
  "status": "ok",
  "loaded": false,
  "model": "black-forest-labs/flux-3-image-klein",
  "active_model": null,
  "default_model": "flux3",
  "models": {
    "flux3": {
      "name": "FLUX 3 Image Klein",
      "description": "最新世代・蒸留済み推論モデル",
      "license": "FLUX Kommunity License v1.0",
      "quantization": "Official INT8 ConvRot",
      "available": true
    }
  },
  "busy": false,
  "memory_budget_gib": 8.0
}
```

---

## `GET /api/models`

### レスポンス `200`: `ModelsResponse`

| フィールド | 型 | 内容 |
|---|---|---|
| `default` | string | 既定のモデルの選択値 |
| `active` | string \| null | 読み込み中のモデルの選択値 |
| `models` | object | 選択値 → [`ModelInfo`](#modelinfo) |

### `ModelInfo`

| フィールド | 型 | 内容 |
|---|---|---|
| `name` | string | 表示名 |
| `description` | string | 説明 |
| `license` | string | モデルのライセンス名 |
| `quantization` | string | Transformer の量子化 |
| `available` | boolean | 必要なファイルがそろっているか。`false` なら `download_models.py flux3 --acknowledge-flux3-license` で取得する |

---

## `POST /api/enhance-prompt`

短い依頼文を、FLUX 3 Image Klein が前提とする完全なシーンのキャプションに書き直します。
BFL 公式のシステムプロンプト(`flux-3-image-klein-pup` の `system_prompts.json`)を LLM に渡します。
生成はしません。`/api/generate` は既定でこれと同じ展開をしてから生成するので、このエンドポイントは展開結果を生成前に確認・編集したいときに使います。
確認したキャプションで生成するときは、`/api/generate` に `upsample=false` を付けて二重の展開を避けてください。

### リクエスト: `multipart/form-data`

| フィールド | 型 | 必須 | 既定値 | 内容 |
|---|---|---|---|---|
| `prompt` | string | はい | | 短い依頼文。空白だけは `422` |
| `mode` | string | いいえ | `auto` | `auto` / `t2i` / `i2i` / `t2i_bbox` / `i2i_bbox`。`auto` は参照画像があれば `i2i`、なければ `t2i` |
| `image` | file(複数可) | いいえ | | 編集の参照画像。同名フィールドを繰り返す。先頭の 8 枚だけを使い、9 枚目以降は無視する。1 枚 10MB・1600 万画素以内 |
| `width` | integer | いいえ | | `t2i` 系で、出力の縦横比を LLM に伝える(`height` と両方指定したときだけ) |
| `height` | integer | いいえ | | 同上 |

- 参照画像は、透過部分を白で埋め、長辺 1024px 以内に縮小してから、依頼文より先に LLM へ渡します。
- `i2i` 系で参照画像がない場合は、依頼文だけを渡します。

### LLM を選ぶ順序

1. **LLM サーバ**: `KLEIN_PROMPT_LLM_URLS` に書いた OpenAI 互換の URL を、書いた順に試す。未設定なら使わない。
   各サーバの `/models` に応答がなければ飛ばす。参照画像があるときは、`/models` の `capabilities` で画像入力に対応しないと分かるサーバも飛ばす。
2. **ローカルの gemma-4-E4B**(llama.cpp の `llama-server`): 空き VRAM が十分なら GPU、そうでなく空きメモリと CPU に余裕があれば CPU で起動する。
   GPU で動かした場合は使い終わるとすぐ停止し、CPU の場合は一定時間使われなければ停止する。`KLEIN_LOCAL_LLM=0` で無効。

どれも使えなければ `503` を返します。

### レスポンス `200`

| フィールド | 型 | 内容 |
|---|---|---|
| `prompt` | string | 展開したキャプション。そのまま `/api/generate` の `prompt` に使える |
| `mode` | string | 実際に使ったモード(`auto` は解決済み) |
| `source` | string | 使った LLM。`remote:<URL>`、`local-gpu`、`local-cpu` のいずれか |
| `model` | string \| null | LLM のモデル名。LLM サーバが名前を返さなければ `null`、ローカル LLM なら GGUF のファイル名 |
| `cached` | boolean | キャッシュから返したか。モード・依頼文・縦横・参照画像が同じなら、直近 128 件をキャッシュから返す |

```json
{
  "prompt": "Create a full-body front view photograph of the woman from the first image, preserving her identity ...",
  "mode": "i2i",
  "source": "remote:http://127.0.0.1:8080/v1",
  "model": "gemma-4-31B",
  "cached": false
}
```

### エラー

| コード | 原因 |
|---|---|
| `413` | 参照画像が 1 枚 10MB を超える |
| `422` | `prompt` が空、`mode` が不正、参照画像を画像として読めない、1600 万画素を超える |
| `503` | 使える LLM がない。`detail` に各候補を使えなかった理由が入る |

### 例

```bash
curl -X POST http://127.0.0.1:7864/api/enhance-prompt \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png'
```

---

## `POST /api/generate`

画像生成のジョブを作り、すぐに `202` を返します。生成はバックグラウンドで実行されます。

### リクエスト: `multipart/form-data`

| フィールド | 型 | 必須 | 既定値 | 制約・内容 |
|---|---|---|---|---|
| `prompt` | string | はい | | 前後の空白を除いて 1〜4000 文字。既定では prompt upsampler が展開するので短い依頼でよい。`upsample=false` のときは完全なシーンのキャプションを渡す |
| `image` | file(複数可) | いいえ | | 参照画像。同名フィールドを最大 8 回。1 枚 10MB・1600 万画素以内。PNG / JPEG / WebP など |
| `width` | integer | いいえ | `768` | 256〜1024、32 の倍数 |
| `height` | integer | いいえ | `768` | 256〜1024、32 の倍数 |
| `seed` | integer | いいえ | `-1` | `-1` はランダム。固定する場合は 0〜4294967295 |
| `model` | string | いいえ | `flux3` | `flux3`(大文字小文字は区別しない) |
| `outpaint` | boolean | いいえ | `false` | `true` でアウトペイント LoRA を使う。参照画像はちょうど 1 枚 |
| `boxes` | string(JSON) | いいえ | `""` | 位置指定。[boxes の書式](#boxes-の書式) |
| `upsample` | boolean | いいえ | `true` | prompt upsampler。生成の前に `/api/enhance-prompt` と同じ処理でプロンプトを展開する。`false` でバイパス(比較用、または展開済みのキャプションを送るとき) |
| `upsample_strict` | boolean | いいえ | `false` | `true` なら、展開できないときにジョブを `error` にする。`false` なら元のプロンプトのまま生成し、`warning` を付ける |

boolean は `true` / `false`(`1` / `0`、`yes` / `no`、`on` / `off` も可)で送ります。

**参照画像の扱い**

- 参照画像なしなら T2I、1 枚なら編集、2 枚以上なら複数参照の合成です。
- 透過部分は白で埋め、EXIF の向きを反映します。
- VRAM を抑えるため、長辺を `reference_max_side` 以内に縮小します。値は 1 枚なら 768px で、枚数が増えると合計の画素数(768² × 8)に収まるよう小さくなります(最小 256px、32 の倍数)。
- アウトペイントでは、参照画像を出力サイズに収まるよう縮小して中央に置き、周りを描き足します。中央部分は元の画像のまま残ります。

**プロンプトの組み立て順**: `upsample` で展開 → `boxes` を末尾に付加 → 生成。

### `boxes` の書式

JSON の配列(1〜32 件)です。座標は 0〜1000 に正規化した `[top, left, bottom, right]` で、`top <= bottom`、`left <= right` です。
各要素の `id` は、キャプションの中で `<id>` の形で引用してください。

**配置(T2I)**

| キー | 型 | 必須 | 内容 |
|---|---|---|---|
| `id` | string | はい | キャプション中の `<id>` と対応 |
| `bbox` | number[4] | はい | 置く位置 |
| `desc` | string | いいえ | 要素の説明 |

**編集**(n 枚目の参照は `<ref_image_n>` で指す)

| キー | 型 | 必須 | 内容 |
|---|---|---|---|
| `id` | string | はい | キャプション中の `<id>` と対応 |
| `from` | string | いいえ | どの参照から取るか(例: `"ref_image_1"`) |
| `src_bbox` | number[4] \| null | どちらか | 元の位置 |
| `tgt_bbox` | number[4] \| null | どちらか | 移動先。`null` でその要素を取り除く |
| `desc` | string | いいえ | 要素の説明 |

`src_bbox` と `tgt_bbox` の両方を `null` にはできません。サーバーはこの配列を、キャプションの後に空白を 1 つ挟んで付けます(モデルカードの書式)。

```json
[{"id": "apple_1", "bbox": [550, 100, 900, 400], "desc": "A shiny red apple with a green leaf."}]
```

### レスポンス `202`: [`JobResponse`](#jobresponse)

`status` は `queued`、`seed` は確定した値(`-1` を送った場合は選ばれた乱数)です。

### エラー

| コード | 原因 |
|---|---|
| `409` | 別のジョブを実行中。モデルまたはアウトペイント LoRA が未取得 |
| `413` | 参照画像が 1 枚 10MB を超える |
| `422` | 入力が範囲外(`prompt`、`width` / `height`、`seed`、`model`、`boxes`)、参照画像が 9 枚以上、`outpaint=true` で参照画像が 1 枚でない、画像を読めない |

### 例

```bash
# 短い依頼を展開してから編集(prompt upsampler は既定で有効)
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=768' -F 'height=1024' -F 'seed=42'

# 比較用: 同じ依頼・同じ seed で prompt upsampler をバイパス
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'prompt=Put her on a plain white studio background, full body, hands empty.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'upsample=false' \
  -F 'width=768' -F 'height=1024' -F 'seed=42'
```

---

## `GET /api/jobs/{job_id}`

### パスパラメータ

| 名前 | 内容 |
|---|---|
| `job_id` | `POST /api/generate` が返した `id`(32 桁の 16 進数) |

### レスポンス `200`: [`JobResponse`](#jobresponse)

### エラー

| コード | 原因 |
|---|---|
| `404` | ジョブが存在しない(サーバーの再起動後、または古いジョブ) |

---

## `POST /api/jobs/{job_id}/cancel`

ジョブの停止を要求します。生成中なら次のステップの区切り(通常 1〜2 秒以内)で止まり、`status` が `cancelled` になります。
モデルの読み込み中と LLM での展開中は、その処理が終わってから止まります。停止したジョブはモデルを解放します。

すでに `done` / `error` / `cancelled` のジョブは、何もせず現在の状態を返します。

### レスポンス `200`: [`JobResponse`](#jobresponse)

停止を受け付けた時点の状態です。`message` が「停止を受け付けました…」になります。完了は `GET /api/jobs/{job_id}` で確認してください。

### エラー

| コード | 原因 |
|---|---|
| `404` | ジョブが存在しない |

---

## `POST /api/unload`

読み込み中のモデルを CPU・GPU のメモリから解放します。次の生成で再び読み込みます(数秒)。
同じ GPU を他のアプリと共有するときに使います。

### レスポンス `200`

```json
{"status": "ok", "loaded": false}
```

### エラー

| コード | 原因 |
|---|---|
| `409` | 生成中のため解放できない |

---

## `GET /outputs/{file}`

`JobResponse` の `image_url`(`/outputs/{job_id}.png`)と `metadata_url`(`/outputs/{job_id}.json`)で示されるファイルです。
URL はベース URL からの相対パスです。ファイルは自動では削除されません。

### メタデータ JSON

| フィールド | 型 | 内容 |
|---|---|---|
| `prompt` | string | 実際にモデルへ渡したプロンプト(展開・`boxes` 付加の後) |
| `original_prompt` | string \| null | 展開や `boxes` でプロンプトが変わったときの元のプロンプト。変わらなければ `null` |
| `upsample_source` | string \| null | 展開に使った LLM |
| `boxes` | array \| null | 送った `boxes` |
| `seed` / `width` / `height` | integer | 生成条件 |
| `references` | string[] | 保存した参照画像のパス(`uploads/{job_id}_ref_{n}.png`) |
| `reference` | string \| null | 1 枚目の参照画像(旧 API 互換) |
| `reference_count` | integer | 参照画像の枚数 |
| `reference_max_side` | integer \| null | 参照画像の長辺の上限 |
| `model` / `model_name` / `model_repo` | string | モデル |
| `base_revision` / `int8_revision` | string | 使ったモデルファイルのリビジョン |
| `transformer_quantization` / `text_quantization` | string | 量子化(`Official INT8 ConvRot` / `BF16 (CPU)`) |
| その他 | | [`metrics`](#metrics) の全フィールド |

---

## データ型

### `JobResponse`

| フィールド | 型 | 内容 |
|---|---|---|
| `id` | string | ジョブ ID |
| `status` | string | [ジョブの状態](#ジョブの状態) |
| `step` | integer | 進んだステップ数(0〜`steps`) |
| `steps` | integer | 総ステップ数(`16`) |
| `model` | string | モデルの選択値 |
| `model_name` | string | モデルの表示名 |
| `seed` | integer | 使う seed |
| `message` | string | 状態の説明。`error` のときは失敗の理由(最大 1000 文字) |
| `created_at` | number | 作成時刻(UNIX 秒) |
| `reference_count` | integer | 参照画像の枚数 |
| `reference_max_side` | integer \| null | 参照画像の長辺の上限。参照なしなら `null` |
| `metrics` | object \| null | 完了時の計測値([`metrics`](#metrics)) |
| `image_url` | string \| null | 完了時の PNG の URL |
| `metadata_url` | string \| null | 完了時のメタデータ JSON の URL |
| `upsample` | boolean | prompt upsampler を使うジョブか(`upsample=false` なら `false`) |
| `upsample_source` | string \| null | 展開に使った LLM |
| `warning` | string \| null | 展開できず元のプロンプトで生成したときの警告 |

### ジョブの状態

| `status` | 意味 | 終了状態 |
|---|---|---|
| `queued` | 受付済み | |
| `upsampling` | LLM でプロンプトを展開中 | |
| `switching` | 構成を切り替えるため、前のモデルを解放中(アウトペイントの有無が変わったときなど) | |
| `loading` | モデルを読み込み中 | |
| `encoding` | プロンプトと参照画像をエンコード中 | |
| `generating` | ノイズ除去中。`step` が 1 ずつ進む | |
| `done` | 完了 | ✓ |
| `error` | 失敗。`message` に理由 | ✓ |
| `cancelled` | 停止済み | ✓ |

### `metrics`

| フィールド | 型 | 内容 |
|---|---|---|
| `model` / `model_name` | string | モデル |
| `seconds` | number | 生成(エンコード〜デコード)にかかった秒数 |
| `total_seconds` | number | モデルの読み込みを含む秒数(展開の時間は含まない) |
| `peak_allocated_gib` | number | PyTorch が割り当てた GPU メモリのピーク |
| `peak_reserved_gib` | number | PyTorch が確保した GPU メモリのピーク |
| `allocator_budget_gib` | number | 割り当て上限 |
| `gpu` | string | GPU 名 |
| `steps` | integer | ステップ数 |
| `transformer_calls` | integer | Transformer の呼び出し回数(CFG なしの蒸留版では `steps` と同じ) |
| `guidance` | number \| null | CFG の値(蒸留版は `null`) |

### エラーの形

```json
{"detail": "幅・高さは256〜1024の32の倍数で指定してください。"}
```

FastAPI 自身の型検証(必須フィールドの欠落、整数でない値など)で失敗した場合は `422` で、`detail` が配列になります。

```json
{"detail": [{"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": null}]}
```

---

## 環境変数

サーバーの起動時に読みます(`run.sh` から起動する場合も同じ)。

**サーバーと生成**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_HOST` | `127.0.0.1` | 待ち受けるアドレス(`run.sh`) |
| `KLEIN_PORT` | `7864` | ポート(`run.sh`) |
| `KLEIN_DEVICE` | `cuda:0` | 使う GPU |
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorch の GPU 割り当て上限(GiB) |

**prompt upsampler(LLM サーバ)**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_PROMPT_LLM_URLS` | なし | OpenAI 互換の URL(`.../v1`)をカンマ区切りで。書いた順に試す |
| `KLEIN_PROMPT_LLM_URL` | なし | 1 件だけ指定する旧形式(`KLEIN_PROMPT_LLM_URLS` が優先) |
| `KLEIN_PROMPT_LLM_MODEL` | なし | 1 番目の LLM サーバに `model` として送る名前。未指定、または 2 番目以降のサーバでは、各サーバの `/models` の先頭 |
| `KLEIN_PROMPT_LLM_TIMEOUT_S` | `120` | 1 回の生成のタイムアウト(秒) |
| `KLEIN_PROMPT_LLM_PROBE_TIMEOUT_S` | `3` | `/models` の確認のタイムアウト(秒) |

**prompt upsampler(ローカル LLM)**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_LOCAL_LLM` | `1` | `0` でローカル LLM を使わない |
| `KLEIN_LOCAL_LLM_SERVER` | `~/llama.cpp/build/bin/llama-server` | GPU 用の llama-server |
| `KLEIN_LOCAL_LLM_SERVER_CPU` | `~/llama.cpp/build-cpu/bin/llama-server` | CPU 用の llama-server(なければ GPU 用を使う) |
| `KLEIN_LOCAL_LLM_MODEL` | `~/models/gemma-4-E4B-it-Q4_K_M.gguf` | GGUF のモデル |
| `KLEIN_LOCAL_LLM_MMPROJ` | 自動検出 | 画像入力用の mmproj |
| `KLEIN_LOCAL_LLM_PORT` | `7865` | llama-server のポート(`127.0.0.1` のみ) |
| `KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB` | `8` | GPU で動かすのに必要な空き VRAM(GiB) |
| `KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB` | `10` | CPU で動かすのに必要な空きメモリ(GiB) |
| `KLEIN_LOCAL_LLM_CPU_MAX_LOAD` | `0.75` | CPU で動かす上限の負荷(1 分平均 ÷ コア数) |
| `KLEIN_LOCAL_LLM_START_TIMEOUT_S` | `180` | 起動待ちのタイムアウト(秒) |
| `KLEIN_LOCAL_LLM_TIMEOUT_S` | `300` | 1 回の生成のタイムアウト(秒) |
| `KLEIN_LOCAL_LLM_IDLE_S` | `600` | CPU で動かしたとき、使われずに停止するまでの秒数 |

---

## 制限と保持

| 項目 | 値 |
|---|---|
| 同時に実行できる生成 | 1 件 |
| プロンプト | 1〜4000 文字 |
| 参照画像 | 8 枚まで、1 枚 10MB・1600 万画素以内 |
| 出力サイズ | 256〜1024px、32 の倍数 |
| ステップ数 | 16(固定) |
| `boxes` | 32 件まで |
| メモリに保持するジョブ | 直近 100 件(再起動で消える) |
| upsampler のキャッシュ | 直近 128 件(再起動で消える) |
| 保存先 | 生成画像とメタデータは `outputs/`、参照画像は `uploads/`。自動削除なし |

## データの送信先

画像生成はローカルだけで完結します。外部へ送るのは、プロンプトの展開(`/api/enhance-prompt`、または既定の `upsample=true`)で
LLM サーバを使った場合だけで、依頼文と(編集では)参照画像を `KLEIN_PROMPT_LLM_URLS` の送信先に渡します。
ローカル LLM を使った場合は外部へ送りません。
