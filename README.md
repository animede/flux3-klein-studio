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

公式はprompt upsamplerの使用を必須としています(短い入力は平凡な出力になり、1行の編集指示は構図まで変えてしまう)。このアプリは18.8GBの公式upsamplerの代わりに、BFL公式のシステムプロンプト(t2i / i2i / t2i_bbox / i2i_bbox)でLLMを動かします。画面の「LLMで公式形式に強化」、生成時の自動強化(`upsample=true`)、`POST /api/enhance-prompt`で使えます。編集では参照画像もLLMに見せます。

LLMは次の順に試します。どれも使えなければ理由を表示します(自動強化では元の文のまま生成し警告を出す)。

1. LLMサーバ: `KLEIN_PROMPT_LLM_URLS`(OpenAI互換のURLをカンマ区切りで、書いた順に試す。例: `http://127.0.0.1:8080/v1`。未設定ならLLMサーバは使わない)
2. ローカルのgemma-4-E4B(llama.cpp の llama-server、画像入力用のmmproj付き)
   - 空きVRAMが`KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB`(既定8)以上ならGPUで動かし、使い終わったらすぐ停止
   - そうでなければ、空きメモリが`KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB`(既定10)以上でCPUに余裕があるときCPUで動かす
   - CPUではCPU向けにビルドしたllama-server(`KLEIN_LOCAL_LLM_SERVER_CPU`、既定`~/llama.cpp/build-cpu/bin/llama-server`)を使う。
     GPU向けビルド(`GGML_NATIVE`/`AVX2`なし)をCPUで動かすと毎秒2トークン程度しか出ず、1回の強化に数分かかる。
     CPU向けは`cmake -B build-cpu -DGGML_NATIVE=ON -DGGML_CUDA=OFF && cmake --build build-cpu --target llama-server`で作れる
     (i5-12400Fで生成毎秒10トークン、1回20〜40秒)
   - 関連する設定: `KLEIN_LOCAL_LLM_SERVER`(GPU用)、`KLEIN_LOCAL_LLM_MODEL`、`KLEIN_LOCAL_LLM_MMPROJ`、`KLEIN_LOCAL_LLM_PORT`(既定7865)、`KLEIN_LOCAL_LLM=0`で無効

CUDA 13対応環境と`comfy-kitchen`のConvRotカーネルが必要です。

## セットアップ

Python 3.12、NVIDIA CUDA対応GPU、十分なRAMと空きディスクが必要です。`.venv`、`models`、`outputs`、`uploads`はGit管理されません。

### FLUX 3を準備

[通常版の公式配布ページ](https://huggingface.co/black-forest-labs/flux-3-image-klein)と[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)を読み、Hugging Face上でアクセス条件に同意してから実行します。

```bash
./setup.sh flux3 --acknowledge-flux3-license
```

依存ライブラリだけ準備する場合は`./setup.sh none`を使います。

取得後の画像生成はローカルファイルだけで動作します。プロンプト強化を使った場合のみ、プロンプトと(編集時は)参照画像が`KLEIN_PROMPT_LLM_URLS`の設定先へ送信されます。ローカルLLMにフォールバックした場合は外部へ送信しません。

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
