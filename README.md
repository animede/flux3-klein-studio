# FLUX 3 Image Klein Reference Studio（評価版）

FLUX 3 Image Kleinをローカルで使う画像生成アプリです。T2I、I2I、最大8枚のRef2Iに対応します。

> [!IMPORTANT]
> アプリコードはApache-2.0ですが、FLUX 3モデルはFLUX Kommunity License v1.0です。モデル重みはこのリポジトリに含みません。詳細は[第三者ライセンスと注意事項](THIRD_PARTY_NOTICES.md)を確認してください。

## 起動

```bash
git clone git@github.com:animede/flux-klein-studio.git
cd flux-klein-studio
./setup.sh flux3 --acknowledge-flux3-license
CUDA_VISIBLE_DEVICES=0 ./run.sh
```

画面: http://127.0.0.1:7864/

API仕様書: [docs/API.md](docs/API.md)

## モデル

| 選択値 | 用途 | 画像モデル | テキスト側 | 既定step | ライセンス |
|---|---|---|---|---:|---|
| `flux3` | 最新世代の生成・編集 | 公式INT8 ConvRot distilled | Qwen3-VL-4B-Instruct（CPU BF16） | 16 | FLUX Kommunity License v1.0 |

モデルのダウンロードには、ライセンスを確認したことを示す明示オプションが必要です。

### FLUX 3通常版とBaseの違い

このアプリが追加するのは通常版の`black-forest-labs/flux-3-image-klein`です。

| | FLUX 3 Image Klein | FLUX 3 Image Klein Base |
|---|---|---|
| チェックポイント | 蒸留済み | 未蒸留 |
| 推論 | 16 step、CFGなし | 50 step、CFG 3.5 |
| 主用途 | 生成、編集、セルフホスト | fine-tuning、LoRA、研究 |
| 公式H200実測（1MP） | 約3秒 | 約12秒 |

Baseは「より高品質な通常利用版」ではなく、追加学習用の元モデルです。通常の画像生成アプリでは蒸留版が適しています。

### FLUX 3のメモリとプロンプト

FLUX 3は、公式Comfy配布のdistilled INT8 ConvRot TransformerをDiffusers ModularPipelineへ直接接続します。約5.43GiBのTransformerをGPUに置き、Qwen3-VLテキストエンコーダーはCPUで実行する段階式Pipelineです。既定のCUDA allocator上限は8GiBで、ComfyUIは使用しません。

公式は短い入力を詳細なscene captionへ展開するprompt upsamplerの利用を強く推奨しています。このアプリは大きな追加モデルであるupsamplerを自動取得・実行しないため、FLUX 3では構図、被写体、背景、照明、質感まで含む詳しいプロンプトを入力してください。

CUDA 13対応環境と`comfy-kitchen`のConvRotカーネルが必要です。

## セットアップ

Python 3.12、NVIDIA CUDA対応GPU、十分なRAMと空きディスクが必要です。`.venv`、`models`、`outputs`、`uploads`はGit管理されません。

### FLUX 3を準備

[通常版の公式配布ページ](https://huggingface.co/black-forest-labs/flux-3-image-klein)と[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)を読み、Hugging Face上でアクセス条件に同意してから実行します。

```bash
./setup.sh flux3 --acknowledge-flux3-license
```

依存ライブラリだけ準備する場合は`./setup.sh none`を使います。

取得後の画像生成はローカルファイルだけで動作します。「LLMで公式形式に強化」を使った場合のみ、プロンプトが`KLEIN_PROMPT_LLM_URL`の設定先へ送信されます。

## API

`POST /api/generate`のmultipartフォームでは`model=flux3`を指定できます。省略時もFLUX 3です。

```bash
curl -X POST http://127.0.0.1:7864/api/generate \
  -F 'model=flux3' \
  -F 'prompt=A detailed cinematic scene of a character reading by a cafe window, soft afternoon light...' \
  -F 'image=@/absolute/path/character.png' \
  -F 'width=1024' -F 'height=768' -F 'seed=42'
```

- `GET /api/health`: 利用可能モデル、現在読み込まれているモデル、稼働状態
- `GET /api/models`: 選択値、量子化、ライセンス表示
- `GET /api/jobs/JOB_ID`: 生成状況と使用モデル
- `POST /api/jobs/JOB_ID/cancel`: 停止
- `POST /api/unload`: 現在のモデルを解放

完全な仕様は[API仕様書](docs/API.md)を参照してください。起動中はSwagger UI（`/docs`）、ReDoc（`/redoc`）、OpenAPI JSON（`/openapi.json`）も利用できます。

## ライセンス

このリポジトリで作成したアプリケーションコードは[Apache License 2.0](LICENSE)で提供します。モデル、モデル重み、トークナイザー、依存ライブラリには提供元のライセンスが適用されます。

- [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)（比較対象。アプリでは未使用）

実際の利用ではリンク先の最新の原文が優先されます。
