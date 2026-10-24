# キャラクターシート API

日本語 | [English](CHARSHEET.en.md)

キャラクターの画像 1 枚から、8 方向(前・後ろ・左右・斜め 45 度 4 つ)の全身画像と、それを並べた 1 枚のシートを作ります。
各方向は参照画像からの Edit を 1 回ずつ行います(1 枚に複数の向きを並べて描かせる方式は、左右の横顔が同じ向きになることがあるため使いません)。

- 生成は他のジョブと共通で同時に 1 件だけです。実行中は `POST /api/generate` も `409` になります。
- 1 方向あたり、生成 7〜9 秒 + prompt upsampler の時間(LLM による)。8 方向で 1〜3 分程度です。
- 方向キー・ラベル・レスポンスの形は、一般的なキャラクターシートの API(Qwen-Image-Edit の角度 LoRA 版など)と揃えています。

## 流れ

1. `POST /api/charsheet/generate` に画像を送る(`202` と `job_id`)
2. `GET /api/charsheet/jobs/{job_id}` を 1〜2 秒間隔で、`status` が `done` / `error` / `cancelled` になるまで呼ぶ
   (完成した方向から `views[].url` が埋まる)
3. `sheet_url`(8 方向を並べた PNG)、`zip_url`(全画像の ZIP)を取得する

```bash
curl -F image=@character.png -F seed=42 http://127.0.0.1:7864/api/charsheet/generate
curl http://127.0.0.1:7864/api/charsheet/jobs/JOB_ID
curl -X POST http://127.0.0.1:7864/api/charsheet/jobs/JOB_ID/cancel
```

## 方向

| キー | ラベル | 向き |
|---|---|---|
| `front` | 前 / Front | 正面 |
| `front_left_45` | 左前45度 / Front-Left 45° | キャラクターが自分の左へ 45 度向く(顔は画像の右寄りを向く) |
| `left` | 左 / Left | 自分の左へ 90 度。横顔で、鼻が画像の**右**端を向く |
| `back_left_45` | 左後ろ45度 / Back-Left 45° | 自分の左へ 135 度。ほぼ背中、頭は画像の右寄りを向く |
| `back` | 後ろ / Back | 真後ろ |
| `back_right_45` | 右後ろ45度 / Back-Right 45° | 自分の右へ 135 度。ほぼ背中、頭は画像の左寄りを向く |
| `right` | 右 / Right | 自分の右へ 90 度。横顔で、鼻が画像の**左**端を向く |
| `front_right_45` | 右前45度 / Front-Right 45° | 自分の右へ 45 度(顔は画像の左寄りを向く) |

左右はキャラクター自身から見た向きです(カメラが回り込む向きとは逆)。Qwen-Image-Edit の角度 LoRA で作るシートと同じ向きになります。

シートの並び(4 列 × 2 行): 1 行目 `front` `front_left_45` `front_right_45` `left`、2 行目 `right` `back_left_45` `back_right_45` `back`。

## 事前変換(prep)

背景付き・上半身だけ・横向きなどの画像は、先に 1 回の Edit で「白背景・全身・正面」の 1 枚に変換し(prep)、各方向はその画像から作ります。

- `auto`(既定): 画像の外周の 85% 以上が白(明るく無彩色)なら変換しない。それ以外は変換する
- `on`: 常に変換する / `off`: 変換しない

変換した場合、その画像を `front` としてそのまま使い(`front` の生成は省略)、`prep_url` でも取得できます。
変換しない場合は `front` も入力画像から生成します(出力サイズと構図を他の方向に揃えるため)。
各方向の参照に使った画像は `reference_url` です(変換したら `prep_url`、しなければ `input_url` と同じ)。

「白背景・全身・正面の 1 枚」だけが欲しい場合は `views=front` と `prep=on` で、Edit 1 回で取得できます。

## `POST /api/charsheet/generate`

`multipart/form-data`。成功時 `202` で、下の[ジョブ状態](#get-apicharsheetjobsjob_id)と同じ形を返します。

| フィールド | 型 | 既定 | 説明 |
|---|---|---|---|
| `image` | ファイル | 必須 | キャラクターの画像 1 枚(10MB 以内、1600 万画素以内)。透過 PNG は白背景に合成。1MP を超える画像は縮小して参照する |
| `seed` | int | `-1` | 全方向で共通のシード。`-1` でランダム、または 0〜4294967295 |
| `views` | string | 8 方向 | 方向キーのカンマ区切り。指定した順に生成する |
| `width` / `height` | int | `768` / `1024` | 各方向の出力サイズ。256〜2688 の 32 の倍数、幅×高さは 2048×2048 以下(`/api/generate` と同じ) |
| `size` | int | なし | 正方形で指定する場合の一辺(`width`・`height` を省略したときだけ。32 の倍数に切り下げ) |
| `prep` | string | `auto` | `auto` / `on` / `off`(上記) |
| `upsample` | bool | `true` | 方向ごとに参照画像を見せて LLM でキャプションに展開する(`/api/generate` と同じ) |
| `upsample_strict` | bool | `false` | upsampler が使えないときに失敗にする(既定は元の文で生成し `warning` を付ける) |
| `upsampler` | string | 環境変数 | `auto` / `remote` / `local` / `pup-int8` / `pup-gguf`(`/api/generate` と同じ) |
| `shift` | float | 環境変数 | タイムステップのシフト(`/api/generate` と同じ) |

エラー: `409`(別の生成を実行中・モデル未取得)、`413`(10MB 超)、`422`(不正な方向キー・prep・サイズ・シード、画像を読み込めない)。

## `GET /api/charsheet/jobs/{job_id}`

```json
{
  "job_id": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
  "status": "running",
  "progress": 3,
  "total": 8,
  "views": [
    {"key": "front", "label_ja": "前", "label_en": "Front", "status": "done",
     "url": "/api/charsheet/jobs/0f1e…/images/front.png"},
    {"key": "back", "label_ja": "後ろ", "label_en": "Back", "status": "running", "url": null}
  ],
  "sheet_url": null,
  "zip_url": null,
  "error": null,
  "seed": 42, "width": 768, "height": 1024,
  "requested_views": ["front", "back", "left", "right", "front_left_45", "front_right_45", "back_left_45", "back_right_45"],
  "prep": "auto", "prep_applied": true, "prep_status": "done",
  "prep_url": "/api/charsheet/jobs/0f1e…/prep.png",
  "input_url": "/api/charsheet/jobs/0f1e…/input.png",
  "reference_url": "/api/charsheet/jobs/0f1e…/prep.png",
  "white_background_ratio": 0.017,
  "current": "back", "step": 9, "steps": 16, "message": "back: 生成中 9 / 16",
  "upsample": true, "upsampler": "auto", "warning": null,
  "metadata_url": "/api/charsheet/jobs/0f1e…/metadata.json",
  "created_at": 1792656173.05
}
```

| フィールド | 説明 |
|---|---|
| `status` | `queued` → `running` → `done` / `error` / `cancelled` |
| `progress` / `total` | 完成した方向の数 / 要求した方向の数(prep は数えない) |
| `views[].status` | `queued` / `running` / `done` / `error` / `cancelled` |
| `views[].url` | 完成した方向の PNG |
| `sheet_url` / `zip_url` | 全方向の完成後に設定。シートは完成した方向だけを並べる。ZIP は各方向・`sheet.png`・`input.png`・`prep.png`(あれば) |
| `error` | 失敗した段階と理由(例 `back: GPUメモリが不足しました…`) |
| `prep_applied` / `prep_status` | 変換したか / `skipped` / `queued` / `running` / `done` / `error` / `cancelled` |
| `white_background_ratio` | 入力画像の外周のうち白い画素の割合(`auto` の判定に使う、0.85 以上で白背景とみなす) |
| `current` / `step` / `steps` / `message` | 処理中の段階(`prep` または方向キー)と、その中のステップ |
| `warning` | upsampler が使えず元の文で生成した場合など |
| `metadata_url` | 方向ごとの実際のプロンプト(upsampler の展開後)・upsampler の出所・生成時間を含む JSON |

停止(`POST /api/charsheet/jobs/{job_id}/cancel`)はステップの区切りで反映され、完成済みの方向はそのまま残ります。

## ファイル

| パス | 内容 |
|---|---|
| `GET /api/charsheet/jobs/{job_id}/images/{key}.png` | 各方向 |
| `GET /api/charsheet/jobs/{job_id}/input.png` | 入力画像(読み込み・縮小後) |
| `GET /api/charsheet/jobs/{job_id}/prep.png` | 白背景・全身・正面に変換した画像(prep をした場合) |
| `GET /api/charsheet/jobs/{job_id}/sheet.png` | 8 方向を並べたシート |
| `GET /api/charsheet/jobs/{job_id}/download.zip` | 全画像の ZIP |
| `GET /api/charsheet/jobs/{job_id}/metadata.json` | 状態と方向ごとのプロンプト |

ファイルは `outputs/charsheet/{job_id}/` に保存され、自動削除されません。ジョブ状態も同じフォルダの `job.json` に書くので、
サーバーを再起動しても `GET /api/charsheet/jobs/{job_id}` で取得できます(実行中に止まったジョブは `error` になります)。
ギャラリー(`/api/gallery`)には出ません。

## プロンプト

各方向の文は `charsheet.py` の `BASE_PROMPT` + `VIEWS[].prompt` です。キャラクター固有の記述は入れず、
upsampler が参照画像を見て服装などを書き足します。工夫している点:

- 向きは「キャラクターが自分の左へ何度向いたか」と「鼻・顔が画像のどちらの端を向くか」の両方で書く。
- 斜め 45 度は「正面と横顔のちょうど中間、肩と腰が斜め、手前の肩が前に出て奥の腕は体に半分隠れる」と体の形で書く(回転が浅くなりやすいため)。
  目の見え方を書くと、アニメ調では前髪で目が隠れることがあったので書かない。
- 後ろ向きでは「前側だけにある柄(胸のプリント・前ポケット・胸当て・ボタン)は見えない」と書く(背中に前の柄が出やすいため)。
- 片側だけの意匠(片脚の靴下など)のために、キャラクターの左側が画像のどちら側に来るかを書く。
