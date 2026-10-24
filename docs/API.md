# API仕様

日本語 | [English](API.en.md)

FLUX 3 Image Klein Reference Studio（DiffusersサーバとGUIアプリ）は、FLUX 3向けのジョブ方式HTTP APIを提供します。既定のベースURLは`http://127.0.0.1:7864`です。

- 全エンドポイントの入出力・エラー・環境変数の一覧: [APIリファレンス](API_REFERENCE.md)
- 使っている手法(INT8の読み込み、メモリの設計、upsampler): [テクニカルガイド](TECHNICAL_GUIDE.md)
- プロンプトの書き方と機能ごとの作例: [プロンプトガイド](PROMPT_GUIDE.md)
- 8方向のキャラクターシート: [キャラクターシートAPI](CHARSHEET.md)
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI 3.1 JSON: `/openapi.json`
- 認証: なし
- 同時生成: 1件
- 生成形式: PNG

サーバーは既定で`127.0.0.1`だけに待ち受けます。外部公開する場合は、別途認証とアクセス制限を設定してください。

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| `GET` | `/api/health` | 稼働状態、読み込み中のモデル、モデル準備状態を取得 |
| `GET` | `/api/models` | モデル名、量子化、ライセンス、準備状態を取得 |
| `POST` | `/api/enhance-prompt` | 短い依頼をFLUX 3用の完全なキャプションへ展開(prompt upsampler) |
| `POST` | `/api/generate` | 生成ジョブを開始 |
| `GET` | `/api/jobs/{job_id}` | ジョブ状態を取得 |
| `POST` | `/api/jobs/{job_id}/cancel` | ジョブの停止を要求 |
| `POST` | `/api/unload` | 現在のモデルをCPU/GPUメモリから解放 |
| `GET` | `/api/gallery` | 生成画像の一覧(新しい順) |
| `DELETE` | `/api/gallery/{id}` | 生成画像を削除(メタデータと参照画像も) |
| `POST` | `/api/charsheet/generate` | 8方向のキャラクターシートを開始([キャラクターシートAPI](CHARSHEET.md)) |
| `GET` | `/api/charsheet/jobs/{job_id}` | キャラクターシートの状態(方向ごとの画像URL、シート、ZIP) |
| `POST` | `/api/charsheet/jobs/{job_id}/cancel` | キャラクターシートの停止を要求 |

## 画像生成

### `POST /api/generate`

リクエストは`multipart/form-data`です。

| フィールド | 必須 | 既定値 | 制約 |
|---|---:|---:|---|
| `prompt` | はい | なし | 1〜4000文字 |
| `model` | いいえ | `flux3` | `flux3` |
| `image` | いいえ | なし | 同名フィールドを最大8回指定。各10MB・1600万画素以内 |
| `width` | いいえ | `768` | 256〜2688、32の倍数。幅×高さは2048×2048以下。1024×1024を超えると2K扱いで、GPUの割り当て上限が10GiBになる |
| `height` | いいえ | `768` | 256〜2688、32の倍数 |
| `seed` | いいえ | `-1` | `-1`はランダム、固定値は0〜4294967295 |
| `outpaint` | いいえ | `false` | `true`でアウトペイントLoRAを使用。参照画像はちょうど1枚(それ以外は422) |
| `boxes` | いいえ | 空文字 | 0〜1000座標のJSON配列、最大32件。配置は`{id, bbox, desc}`、編集は`{id, from, src_bbox, tgt_bbox, desc}`(`tgt_bbox: null`で削除)。座標は`[top, left, bottom, right]` |
| `upsample` | いいえ | `true` | prompt upsampler。生成前にプロンプトを完全なキャプションへ展開(`/api/enhance-prompt`と同じ処理)。`false`でバイパス(比較用、または展開済みのキャプションを送るとき) |
| `upsample_strict` | いいえ | `false` | `true`ならupsampleが使えないときに生成を止める(既定は元のプロンプトで生成し`warning`を付ける) |
| `upsampler` | いいえ | `KLEIN_UPSAMPLER`(既定`auto`) | 使うLLM。`auto`、`remote`、`local`、`pup-int8`、`pup-gguf`(下の「prompt upsampler」) |

参照画像を省略するとT2I、1枚以上指定するとI2I / Ref2Iになります。PNG、JPEG、WebPを推奨します。参照画像は1枚あたり最大1MP(公式の推奨)に縮小し、合計は2MPまでに抑えます(2枚までは各1MP、8枚なら各0.25MP)。BFLの推奨出力は2K(2048×2048など)で、1024×1024以下は8GiB、2KはGPUの割り当て上限10GiBで動きます。

生成は16 stepです。FLUX 3は完全なscene captionを前提にしているため、既定ではprompt upsamplerで展開してから生成します。`/api/enhance-prompt`で展開済みのキャプションや、自分で書いた完全なキャプションを送る場合は、二重に展開しないよう`upsample=false`を付けてください。

`boxes`はモデルカードの書式どおり、キャプションの後に空白1つを挟んでJSON配列として付けます。各要素はキャプション中で`<id>`の形で引用してください(例: `Remove the handbag <bag_1>.`)。複数参照の編集では、n枚目の参照を`<ref_image_n>`で指せます。

## prompt upsampler

### `POST /api/enhance-prompt`

| フィールド | 必須 | 既定値 | 内容 |
|---|---:|---:|---|
| `prompt` | はい | なし | 短い依頼文 |
| `mode` | いいえ | `auto` | `auto`(参照画像の有無で`t2i`/`i2i`)、`t2i`、`i2i`、`t2i_bbox`、`i2i_bbox` |
| `image` | いいえ | なし | 編集の参照画像(最大8枚)。順番どおりLLMに見せる |
| `width` / `height` | いいえ | なし | T2Iで出力の縦横比をLLMに伝える |
| `upsampler` | いいえ | `KLEIN_UPSAMPLER`(既定`auto`) | 使うLLM(下の表) |

外部のLLM(`auto`・`remote`・`local`)には、BFLが外部LLM向けに公開しているシステムプロンプト(`models/flux-3-image-klein-pup/system_prompts_external_vlm.json`、無ければ自動取得)を渡し、編集では依頼文(`User request: ...`)の後に参照画像を付けます。公式upsampler(`pup-*`)には、公式upsampler用のシステムプロンプト(`system_prompts.json`)と公式の書式(参照画像が先)を使います。使えるLLMがなければ`503`と理由を返します。

| `upsampler` | 使うLLM |
|---|---|
| `auto` | LLMサーバ(`KLEIN_PROMPT_LLM_URLS`、書いた順)→ ローカルのgemma-4-E4B(llama.cpp。空きVRAMが十分ならGPU、なければCPU) |
| `remote` / `local` | 上のどちらか一方だけ |
| `pup-int8` | 公式upsamplerのINT8版(GPU、空きVRAM 12GiB以上) |
| `pup-gguf` | 公式upsamplerのGGUF版Q4_K_M(GPU、空きVRAM 8GiB以上) |

公式版(`pup-*`)は使うたびに起動し、使い終わったらすぐ止めます。空きVRAMが足りないときは画像モデルを一時的に解放してから起動します(生成中の`/api/enhance-prompt`では解放できないので`503`)。どれが使えるかは`GET /api/health`の`upsamplers`で分かります。

```json
{"prompt": "Create a full-body front view photograph of the woman from the first image ...",
 "mode": "i2i", "source": "remote:http://127.0.0.1:8080/v1", "model": "...", "cached": false}
```

同じ依頼・同じ参照画像の結果はキャッシュされます(`cached: true`)。

#### T2I

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A quiet library at sunset, detailed illustration.' \
  -F 'width=1024' \
  -F 'height=768' \
  -F 'seed=42'
```

#### 1枚参照

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=The same character reading a book in a cafe.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=768' \
  -F 'height=768' \
  -F 'seed=42'
```

#### 複数参照

同じ`image`フィールドを繰り返します。

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=Use the same character, outfit, and art style.' \
  -F 'image=@/absolute/path/face.png' \
  -F 'image=@/absolute/path/outfit.png' \
  -F 'image=@/absolute/path/style.png'
```

#### アウトペイント

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A detailed editorial photograph of the same character seated by a cafe window, preserving facial features and clothing, with soft afternoon light and a shallow depth of field.' \
  -F 'image=@/absolute/path/character.png' \
  -F 'outpaint=true' \
  -F 'width=1024' \
  -F 'height=768'
```

### 受付レスポンス

成功時は`202 Accepted`とジョブ情報を返します。

```json
{
  "id": "a12b34c56d78",
  "status": "queued",
  "step": 0,
  "steps": 16,
  "model": "flux3",
  "model_name": "FLUX 3 Image Klein",
  "seed": 42,
  "message": "生成を準備しています",
  "created_at": 1791948447.9,
  "reference_count": 0,
  "reference_max_side": null,
  "metrics": null,
  "image_url": null,
  "metadata_url": null
}
```

`id`を`GET /api/jobs/{job_id}`へ渡して完了まで確認します。

## ジョブ状態

### `GET /api/jobs/{job_id}`

`status`は次のいずれかです。

| 状態 | 意味 |
|---|---|
| `queued` | 実行待ち |
| `upsampling` | プロンプトを展開中(`upsample=false`以外) |
| `switching` | 前のモデルを解放中 |
| `loading` | 選択モデルを読み込み中 |
| `encoding` | プロンプト・参照画像を処理中 |
| `generating` | 画像生成中 |
| `done` | 完了 |
| `error` | 失敗。`message`に理由を格納 |
| `cancelled` | 停止済み |

完了時は`metrics`、`image_url`、`metadata_url`が追加されます。prompt upsamplerを使ったときは`upsample_source`(使ったLLM)、展開できずに元のプロンプトで生成したときは`warning`が入ります。メタデータJSONには展開前のプロンプト(`original_prompt`)も記録されます。

```json
{
  "id": "a12b34c56d78",
  "status": "done",
  "step": 16,
  "steps": 16,
  "model": "flux3",
  "model_name": "FLUX 3 Image Klein",
  "seed": 42,
  "message": "生成が完了しました",
  "created_at": 1791948447.9,
  "reference_count": 0,
  "reference_max_side": null,
  "metrics": {
    "model": "flux3",
    "model_name": "FLUX 3 Image Klein",
    "seconds": 3.92,
    "total_seconds": 23.98,
    "peak_allocated_gib": 3.054,
    "peak_reserved_gib": 3.355,
    "allocator_budget_gib": 8.0,
    "gpu": "NVIDIA GPU",
    "steps": 16
  },
  "image_url": "/outputs/a12b34c56d78.png",
  "metadata_url": "/outputs/a12b34c56d78.json"
}
```

画像URLとメタデータURLはベースURLからの相対URLです。

## 停止とモデル解放

```bash
curl -X POST http://127.0.0.1:7864/api/jobs/JOB_ID/cancel
curl -X POST http://127.0.0.1:7864/api/unload
```

停止要求は生成中ならステップの区切りで反映されます(通常1〜2秒以内)。モデル読み込み中は即時停止できません。生成中に`/api/unload`を呼ぶと`409 Conflict`になります。

## ギャラリー(生成画像の一覧と削除)

```bash
curl 'http://127.0.0.1:7864/api/gallery?limit=60&offset=0'
curl -X DELETE http://127.0.0.1:7864/api/gallery/JOB_ID
```

一覧は`outputs/`のPNGを新しい順に返します(`total`と`items`。各要素は`id`、`image_url`、`metadata_url`、`created_at`、`width`、`height`、`seed`、`prompt`、`original_prompt`、`upsampler`、`upsample_source`、`reference_count`)。サーバーを再起動しても、ファイルが残っていれば一覧に出ます。
削除は、PNG・メタデータJSON・そのジョブの参照画像(`uploads/`)を消します。IDは32桁の16進数だけを受け付け、生成中のジョブは`409`になります。

## モデル・状態確認

```bash
curl http://127.0.0.1:7864/api/models
curl http://127.0.0.1:7864/api/health
```

`available`が`false`の場合は、公式配布ページでアクセス条件へ同意してから`download_models.py flux3 --acknowledge-flux3-license`で取得してください。詳細は[第三者ライセンスと注意事項](../THIRD_PARTY_NOTICES.md)を参照してください。`active_model`は現在メモリに読み込まれているモデルで、未読み込み時は`null`です。

## HTTPエラー

| 状態コード | 主な原因 |
|---:|---|
| `404` | 指定したジョブ・画像が存在しない |
| `409` | 別の生成を実行中、モデル未取得、生成中のモデル解放、生成中の画像の削除 |
| `413` | 参照画像が1枚10MBを超過 |
| `503` | `/api/enhance-prompt`で使えるLLMが無い(理由を`detail`に格納) |
| `422` | 入力範囲外、未対応モデル、画像形式または画像内容を読み込めない |

エラー本文は通常`{"detail":"説明"}`です。FastAPI自身の入力型検証では`detail`が配列になります。

## 保存と保持

- ジョブ状態は直近100件をメモリに保持します。
- サーバーを再起動するとジョブ状態は消えます。
- 生成PNGとJSONメタデータは`outputs/`へ保存します。
- 参照画像は`uploads/`へ保存します。
- 保存ファイルは自動削除されません。ギャラリー(画面または`DELETE /api/gallery/{id}`)で削除できます。
