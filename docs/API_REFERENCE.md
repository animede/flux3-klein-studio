# API リファレンス

日本語 | [English](API_REFERENCE.en.md)

FLUX 3 Image Klein Reference Studio の HTTP API の全エンドポイント、パラメータ、レスポンス、エラーの一覧です。
使い方の流れと呼び出し例は [API 仕様(ガイド)](API.md) を参照してください。

- API バージョン: `0.4.0`(`/openapi.json` の `info.version`)
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
| `GET` | [`/api/gallery`](#get-apigallery) | 生成画像の一覧(新しい順) | `200` |
| `DELETE` | [`/api/gallery/{id}`](#delete-apigalleryid) | 生成画像を削除 | `200` |
| `GET` | [`/outputs/{file}`](#get-outputsfile) | 生成した PNG とメタデータ JSON | `200` |
| `POST` | [`/api/charsheet/generate`](CHARSHEET.md#post-apicharsheetgenerate) | 8 方向のキャラクターシートを開始 | `202` |
| `GET` | [`/api/charsheet/jobs/{job_id}`](CHARSHEET.md#get-apicharsheetjobsjob_id) | キャラクターシートの状態 | `200` |
| `POST` | [`/api/charsheet/jobs/{job_id}/cancel`](CHARSHEET.md#get-apicharsheetjobsjob_id) | キャラクターシートの停止を要求 | `200` |
| `GET` | [`/api/charsheet/jobs/{job_id}/…`](CHARSHEET.md#ファイル) | 方向ごとの画像・シート・ZIP・メタデータ | `200` |
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
| `memory_budget_gib` | number | PyTorch の GPU 割り当て上限(GiB)。1024×1024 以下の出力 |
| `hires_memory_budget_gib` | number | 1024×1024 を超える出力(2K)のときの割り当て上限(GiB) |
| `default_upsampler` | string | 既定の upsampler(`KLEIN_UPSAMPLER`、既定 `auto`) |
| `upsamplers` | object | upsampler の名前 → `{"available": boolean, "detail": string}`。使えるか、使えない理由や必要な空き VRAM |

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
外部の LLM(`auto`・`remote`・`local`)には、BFL が外部 LLM 向けに公開しているシステムプロンプト(公式推論コードの `pup_prompts_external_vlm.json`)を、公式 upsampler(`pup-*`)には公式 upsampler 用のシステムプロンプト(`flux-3-image-klein-pup` の `system_prompts.json`)を渡します。
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
| `upsampler` | string | いいえ | `KLEIN_UPSAMPLER`(既定 `auto`) | 使う LLM。[LLM の選び方](#llm-の選び方) |

入力は公式の推論コード(`src/flux3/pup.py`)と同じ形にします。

- 参照画像は、透過部分を白で埋め、公式の画素数(1 枚なら約 1MP、2 枚なら 724²、3 枚以上は 512² ずつ)に縮小し、JPEG にして、依頼文より先に渡します。
- 編集では、依頼文を公式の 3 行の書式(`Reference images follow in order ...` / `User request: ...` / `Output the caption.`)に入れます。
- `t2i` 系で縦横を指定すると、先頭に `TARGET CANVAS: 幅:高さ` の行を付けます(公式の t2i システムプロンプトが受け付ける形)。
- デコードは貪欲法(temperature 0)、presence penalty は `t2i` 系 1.5・`i2i` 系 0、最大 3584 トークン。
- `i2i` 系で参照画像がない場合は、依頼文だけを渡します。

### LLM の選び方

| `upsampler` | 使う LLM |
|---|---|
| `auto` | 下の 1 → 2 の順に試す |
| `remote` | 1 だけ |
| `local` | 2 だけ |
| `pup-int8` | 公式 upsampler の INT8 版(BFL 公式、`pup_server.py`)。GPU で空き VRAM 12GiB 以上 |
| `pup-gguf` | 公式 upsampler の GGUF 版(Q4_K_M+mmproj、コミュニティ製、llama-server)。GPU で空き VRAM 8GiB 以上 |

公式版(`pup-*`)は、使うたびに GPU で起動し、使い終わるとすぐ止めます。空き VRAM が足りないときは、画像モデルを一時的に解放してから起動します。
`/api/generate` の中ではいつでも解放でき、`/api/enhance-prompt` では生成中でないときだけ解放します(生成中なら `503`)。

1. **LLM サーバ**: `KLEIN_PROMPT_LLM_URLS` に書いた OpenAI 互換の URL を、書いた順に試す。未設定なら使わない。
   各サーバの `/models` に応答がなければ飛ばす。参照画像があるときは、`/models` の `capabilities` で画像入力に対応しないと分かるサーバも飛ばす。
2. **ローカルの gemma-4-E4B**(llama.cpp の `llama-server`): 空き VRAM が十分なら GPU、そうでなく空きメモリと CPU に余裕があれば CPU で起動する。
   GPU で動かした場合は使い終わるとすぐ停止し、CPU の場合は一定時間使われなければ停止する。`KLEIN_LOCAL_LLM=0` で無効。

使える LLM がなければ `503` を返します。

### レスポンス `200`

| フィールド | 型 | 内容 |
|---|---|---|
| `prompt` | string | 展開したキャプション。そのまま `/api/generate` の `prompt` に使える |
| `mode` | string | 実際に使ったモード(`auto` は解決済み) |
| `source` | string | 使った LLM。`remote:<URL>`、`local-gpu`、`local-cpu`、`pup-int8`、`pup-gguf` のいずれか |
| `model` | string \| null | LLM のモデル名。LLM サーバが名前を返さなければ `null`、ローカル・公式版ならモデルのファイル名 |
| `cached` | boolean | キャッシュから返したか。upsampler・モード・依頼文・縦横・参照画像が同じなら、直近 128 件をキャッシュから返す |

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
| `422` | `prompt` が空、`mode` または `upsampler` が不正、参照画像を画像として読めない、1600 万画素を超える |
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
| `width` | integer | いいえ | `768` | 256〜2688、32 の倍数。幅×高さは 2048×2048(4,194,304 画素)以下。1024×1024 を超えると 2K 扱いで、割り当て上限が `KLEIN_HIRES_MEMORY_BUDGET_GIB`(既定 10GiB)になる。GPU の容量が足りなければジョブは `error` |
| `height` | integer | いいえ | `768` | 256〜2688、32 の倍数 |
| `seed` | integer | いいえ | `-1` | `-1` はランダム。固定する場合は 0〜4294967295 |
| `model` | string | いいえ | `flux3` | `flux3`(大文字小文字は区別しない) |
| `outpaint` | boolean | いいえ | `false` | `true` でアウトペイント LoRA を使う。参照画像はちょうど 1 枚 |
| `boxes` | string(JSON) | いいえ | `""` | 位置指定。[boxes の書式](#boxes-の書式) |
| `upsample` | boolean | いいえ | `true` | prompt upsampler。生成の前に `/api/enhance-prompt` と同じ処理でプロンプトを展開する。`false` でバイパス(比較用、または展開済みのキャプションを送るとき) |
| `upsample_strict` | boolean | いいえ | `false` | `true` なら、展開できないときにジョブを `error` にする。`false` なら元のプロンプトのまま生成し、`warning` を付ける |
| `shift` | number | いいえ | `KLEIN_TIMESTEP_SHIFT`(既定 `16`) | タイムステップのシフト(alpha)。1〜50。BFL の推奨値は 16(2026-10-17 に 6.93 から変更) |
| `upsampler` | string | いいえ | `KLEIN_UPSAMPLER`(既定 `auto`) | 展開に使う LLM。[LLM の選び方](#llm-の選び方)。公式版で VRAM が足りなければ、画像モデルを解放してから展開し、生成の前に読み込み直す |

boolean は `true` / `false`(`1` / `0`、`yes` / `no`、`on` / `off` も可)で送ります。

**参照画像の扱い**

- 参照画像なしなら T2I、1 枚なら編集、2 枚以上なら複数参照の合成です。
- 透過部分は白で埋め、EXIF の向きを反映します。
- 1 枚あたり最大 1MP(BFL の推論コードと同じ)に、縦横比を保って縮小します。ただし参照画像の合計は 2MP(`KLEIN_REFERENCE_TOTAL_MP`)までに抑えるので、1〜2 枚なら各 1MP、4 枚なら各 0.5MP、8 枚なら各 0.25MP です(1 枚あたりの上限は `reference_max_pixels`)。8 枚を 1MP ずつにすると 8GiB に収まらないための制限です。
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

## `GET /api/gallery`

`outputs/` にある生成画像を、新しい順(ファイルの更新時刻順)に返します。ジョブの記録(直近 100 件、再起動で消える)とは別に、
ファイルが残っている画像はすべて対象です。

### クエリパラメータ

| 名前 | 既定値 | 内容 |
|---|---|---|
| `limit` | `60` | 返す件数(1〜500 に丸める) |
| `offset` | `0` | 先頭から飛ばす件数 |

### レスポンス `200`

| フィールド | 型 | 内容 |
|---|---|---|
| `total` | integer | 画像の総数 |
| `items` | [`GalleryItem`](#galleryitem)[] | 画像の一覧 |

### `GalleryItem`

| フィールド | 型 | 内容 |
|---|---|---|
| `id` | string | ジョブ ID(32 桁の 16 進数) |
| `image_url` | string | PNG の URL(`/outputs/{id}.png`) |
| `metadata_url` | string \| null | メタデータ JSON の URL。ファイルがなければ `null` |
| `created_at` | number | 生成時刻(ファイルの更新時刻、UNIX 秒) |
| `width` / `height` / `seed` | integer \| null | 生成条件(メタデータから) |
| `prompt` | string \| null | モデルに渡したプロンプト |
| `original_prompt` | string \| null | 展開前の依頼文 |
| `upsampler` / `upsample_source` | string \| null | 選んだ upsampler と、実際に使った LLM |
| `reference_count` | integer | 参照画像の枚数 |

---

## `DELETE /api/gallery/{id}`

生成画像を削除します。PNG、メタデータ JSON、そのジョブの参照画像(`uploads/{id}_ref_*.png`)を消します。元に戻せません。

### レスポンス `200`

```json
{"id": "a12b34c56d78e90f...", "deleted": ["a12b....png", "a12b....json", "a12b..._ref_1.png"]}
```

### エラー

| コード | 原因 |
|---|---|
| `404` | ID が 32 桁の 16 進数でない、または画像が存在しない |
| `409` | そのジョブが生成中(終了状態でない) |

---

## `GET /outputs/{file}`

`JobResponse` の `image_url`(`/outputs/{job_id}.png`)と `metadata_url`(`/outputs/{job_id}.json`)で示されるファイルです。
URL はベース URL からの相対パスです。ファイルは自動では削除されません。

### メタデータ JSON

| フィールド | 型 | 内容 |
|---|---|---|
| `prompt` | string | 実際にモデルへ渡したプロンプト(展開・`boxes` 付加の後) |
| `original_prompt` | string \| null | 展開や `boxes` でプロンプトが変わったときの元のプロンプト。変わらなければ `null` |
| `upsampler` | string \| null | 選んだ upsampler(使わなかったときは `null`) |
| `upsample_source` | string \| null | 展開に使った LLM |
| `boxes` | array \| null | 送った `boxes` |
| `seed` / `width` / `height` | integer | 生成条件 |
| `references` | string[] | 保存した参照画像のパス(`uploads/{job_id}_ref_{n}.png`) |
| `reference` | string \| null | 1 枚目の参照画像(旧 API 互換) |
| `reference_count` | integer | 参照画像の枚数 |
| `reference_max_side` | integer \| null | 参照画像 1 枚あたりの画素数の上限を正方形の一辺に換算した値(目安) |
| `reference_max_pixels` | integer \| null | 参照画像 1 枚あたりの画素数の上限 |
| `model` / `model_name` / `model_repo` | string | モデル |
| `base_revision` / `int8_revision` | string | 使ったモデルファイルのリビジョン |
| `transformer_quantization` / `text_quantization` | string | 量子化(`Official INT8 ConvRot` / `BF16 (CPU memory, GPU sequential offload)`) |
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
| `reference_max_side` | integer \| null | 参照画像 1 枚あたりの画素数の上限を正方形の一辺に換算した値(目安)。参照なしなら `null` |
| `reference_max_pixels` | integer \| null | 参照画像 1 枚あたりの画素数の上限。参照なしなら `null` |
| `metrics` | object \| null | 完了時の計測値([`metrics`](#metrics)) |
| `image_url` | string \| null | 完了時の PNG の URL |
| `metadata_url` | string \| null | 完了時のメタデータ JSON の URL |
| `upsample` | boolean | prompt upsampler を使うジョブか(`upsample=false` なら `false`) |
| `upsampler` | string | 選んだ upsampler(`auto` など) |
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
| `shift` | number | 使ったタイムステップのシフト(alpha) |
| `text_encoder` | string | テキストエンコーダーの計算場所(`gpu-offload` / `cpu`) |

### エラーの形

```json
{"detail": "幅・高さは256〜2688の32の倍数で、合計4,194,304画素(2048×2048)以下で指定してください。"}
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
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorch の GPU 割り当て上限(GiB)。1024×1024 以下の出力 |
| `KLEIN_HIRES_MEMORY_BUDGET_GIB` | `10.0` | 1024×1024 を超える出力(2K)の割り当て上限。4MP の T2I は約 9GiB、参照画像付きは約 10GiB 必要 |
| `KLEIN_REFERENCE_TOTAL_MP` | `2` | 参照画像の合計画素数の上限(MP)。1 枚あたりは最大 1MP |
| `KLEIN_TIMESTEP_SHIFT` | `16` | タイムステップのシフト(alpha)の既定値。BFL は 2026-10-17 に 6.93 から 16 に変更 |
| `KLEIN_TEXT_ENCODER` | `gpu-offload` | テキストエンコーダーの計算場所。`gpu-offload`(CPU メモリに置き GPU で 1 層ずつ計算、約 2.5 秒)/ `cpu`(約 30 秒) |

**prompt upsampler(LLM サーバ)**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_PROMPT_LLM_URLS` | なし | OpenAI 互換の URL(`.../v1`)をカンマ区切りで。書いた順に試す |
| `KLEIN_PROMPT_LLM_URL` | なし | 1 件だけ指定する旧形式(`KLEIN_PROMPT_LLM_URLS` が優先) |
| `KLEIN_PROMPT_LLM_MODEL` | なし | 1 番目の LLM サーバに `model` として送る名前。未指定、または 2 番目以降のサーバでは、各サーバの `/models` の先頭 |
| `KLEIN_PROMPT_LLM_TIMEOUT_S` | `120` | 1 回の生成のタイムアウト(秒) |
| `KLEIN_PROMPT_LLM_PROBE_TIMEOUT_S` | `3` | `/models` の確認のタイムアウト(秒) |
| `KLEIN_UPSAMPLER` | `auto` | 既定の upsampler(`auto` / `remote` / `local` / `pup-int8` / `pup-gguf`) |

**prompt upsampler(ローカル LLM)**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_LOCAL_LLM` | `1` | `0` でローカル LLM を使わない |
| `KLEIN_LOCAL_LLM_SERVER` | `~/llama.cpp/build/bin/llama-server` | GPU 用の llama-server |
| `KLEIN_LOCAL_LLM_SERVER_CPU` | `~/llama.cpp/build-cpu/bin/llama-server` | CPU 用の llama-server(なければ GPU 用を使う) |
| `KLEIN_LOCAL_LLM_MODEL` | `~/models/gemma-4-E4B-it-Q4_K_M.gguf` | GGUF のモデル |
| `KLEIN_LOCAL_LLM_MMPROJ` | 自動検出 | 画像入力用の mmproj |
| `KLEIN_LOCAL_LLM_PORT` | `7865` | llama-server のポート(`127.0.0.1` のみ) |
| `KLEIN_LOCAL_LLM_CTX` | `16384` | llama-server のコンテキスト長(外部 LLM 向けの T2I のシステムプロンプトは約 1 万トークン) |
| `KLEIN_LLM_PROMPTS` | `external` | 外部の LLM に渡すシステムプロンプト。`external`(BFL の外部 LLM 向け)/ `pup`(公式 upsampler 用。2026-10-18 より前の動き) |
| `KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB` | `8` | GPU で動かすのに必要な空き VRAM(GiB) |
| `KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB` | `10` | CPU で動かすのに必要な空きメモリ(GiB) |
| `KLEIN_LOCAL_LLM_CPU_MAX_LOAD` | `0.75` | CPU で動かす上限の負荷(1 分平均 ÷ コア数) |
| `KLEIN_LOCAL_LLM_START_TIMEOUT_S` | `180` | 起動待ちのタイムアウト(秒) |
| `KLEIN_LOCAL_LLM_TIMEOUT_S` | `300` | 1 回の生成のタイムアウト(秒) |
| `KLEIN_LOCAL_LLM_IDLE_S` | `600` | CPU で動かしたとき、使われずに停止するまでの秒数 |

**prompt upsampler(公式版)**

| 変数 | 既定値 | 内容 |
|---|---|---|
| `KLEIN_PUP_INT8_PORT` | `7867` | INT8 版(`pup_server.py`)のポート(`127.0.0.1` のみ) |
| `KLEIN_PUP_INT8_MIN_FREE_GIB` | `12` | INT8 版の起動に必要な空き VRAM(GiB)。足りなければ画像モデルを解放する |
| `KLEIN_PUP_GGUF_PORT` | `7866` | GGUF 版(llama-server)のポート(`127.0.0.1` のみ) |
| `KLEIN_PUP_GGUF_MIN_FREE_GIB` | `8` | GGUF 版の起動に必要な空き VRAM(GiB) |

起動のタイムアウトは `KLEIN_LOCAL_LLM_START_TIMEOUT_S`、生成のタイムアウトは `KLEIN_LOCAL_LLM_TIMEOUT_S` を共用します。GGUF 版は GPU 向けの llama-server(`KLEIN_LOCAL_LLM_SERVER`)を使います。

---

## 制限と保持

| 項目 | 値 |
|---|---|
| 同時に実行できる生成 | 1 件 |
| プロンプト | 1〜4000 文字 |
| 参照画像 | 8 枚まで、1 枚 10MB・1600 万画素以内 |
| 出力サイズ | 各辺 256〜2688px・32 の倍数、合計 2048×2048 以下。1024×1024 超は割り当て上限 10GiB |
| 参照画像の大きさ | 1 枚あたり最大 1MP、合計 2MP まで |
| ステップ数 | 16(固定) |
| `boxes` | 32 件まで |
| メモリに保持するジョブ | 直近 100 件(再起動で消える) |
| upsampler のキャッシュ | 直近 128 件(再起動で消える) |
| 保存先 | 生成画像とメタデータは `outputs/`、参照画像は `uploads/`。自動削除なし(`DELETE /api/gallery/{id}` で削除) |

## データの送信先

画像生成はローカルだけで完結します。外部へ送るのは、プロンプトの展開(`/api/enhance-prompt`、または既定の `upsample=true`)で
LLM サーバを使った場合だけで、依頼文と(編集では)参照画像を `KLEIN_PROMPT_LLM_URLS` の送信先に渡します。
ローカル LLM と公式版(`local`・`pup-int8`・`pup-gguf`)を使った場合は外部へ送りません。
