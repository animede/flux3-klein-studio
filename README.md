# FLUX 3 Image Klein Reference Studio（DiffusersサーバとGUIアプリ）

日本語 | [English](README.en.md)

FLUX 3 Image KleinをDiffusersで動かし、8GBのVRAMに収めることを目的に開発したプロジェクトです。API機能を確かめるためのGUIアプリも提供しています。T2I、I2I、最大8枚の参照画像を使うRef2Iに対応します。

> [!IMPORTANT]
> アプリコードはApache-2.0ですが、FLUX 3モデルはFLUX Kommunity License v1.0です。モデル重みはこのリポジトリに含みません。詳細は[第三者ライセンスと注意事項](THIRD_PARTY_NOTICES.md)を確認してください。

## 起動までの流れ

| 手順 | コマンド | すること |
|---|---|---|
| 1. セットアップ | `./setup.sh flux3 --acknowledge-flux3-license` | Python 仮想環境(`.venv`)の作成、依存ライブラリのインストール、モデルのダウンロード。**サーバーは起動しません** |
| 2. FastAPI サーバーの起動 | `./run.sh` | API サーバー(uvicorn)を起動します。Web アプリ(GUI)も同じサーバーが配信します |
| 3. Web アプリを開く | ブラウザで http://127.0.0.1:7864/ | 別に起動するものはありません |

```bash
git clone https://github.com/animede/flux3-klein-studio.git
cd flux3-klein-studio
./setup.sh flux3 --acknowledge-flux3-license   # 1. 初回だけ
CUDA_VISIBLE_DEVICES=0 ./run.sh                 # 2. サーバーを起動(Ctrl+C で停止)
# 3. ブラウザで http://127.0.0.1:7864/ を開く
```

API仕様書: [docs/API.md](docs/API.md)、APIリファレンス: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

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

公式はprompt upsamplerの使用を必須としています(短い入力は平凡な出力になり、1行の編集指示は構図まで変えてしまう)。このアプリは、BFL公式のシステムプロンプト(t2i / i2i / t2i_bbox / i2i_bbox)と公式の入力書式で、手元のLLMを動かします。GPUに余裕があれば、公式のupsampler(量子化版)も使えます(下の「公式のupsamplerを使う」)。**prompt upsamplerは標準で有効**で、生成のたびに入力文を完全なキャプションへ展開してから生成します。
比較のために使わずに生成する(バイパス)こともできます。画面では「prompt upsamplerを使う」のチェックを外し、APIでは`upsample=false`を指定します。
展開結果を生成前に確かめたいときは、画面の「prompt upsamplerで展開して確認」または`POST /api/enhance-prompt`を使います(確認したキャプションをそのまま生成すると、もう一度は展開しません)。編集では参照画像もLLMに見せます。

使うLLMは、画面の「prompt upsamplerのLLM」、APIの`upsampler`、起動時の環境変数`KLEIN_UPSAMPLER`で選びます。既定の`auto`は次の順に試します。どれも使えなければ、元の文のまま生成し、理由を警告として表示します(`upsample_strict=true`なら生成を止めます)。

1. LLMサーバ: `KLEIN_PROMPT_LLM_URLS`(OpenAI互換のURLをカンマ区切りで、書いた順に試す。例: `http://127.0.0.1:8080/v1`。未設定ならLLMサーバは使わない)
2. ローカルのgemma-4-E4B(llama.cpp の llama-server、画像入力用のmmproj付き)
   - 空きVRAMが`KLEIN_LOCAL_LLM_GPU_MIN_FREE_GIB`(既定8)以上ならGPUで動かし、使い終わったらすぐ停止
   - そうでなければ、空きメモリが`KLEIN_LOCAL_LLM_CPU_MIN_RAM_GIB`(既定10)以上でCPUに余裕があるときCPUで動かす
   - CPUではCPU向けにビルドしたllama-server(`KLEIN_LOCAL_LLM_SERVER_CPU`、既定`~/llama.cpp/build-cpu/bin/llama-server`)を使う。
     GPU向けビルド(`GGML_NATIVE`/`AVX2`なし)をCPUで動かすと毎秒2トークン程度しか出ず、1回の展開に数分かかる。
     CPU向けは`cmake -B build-cpu -DGGML_NATIVE=ON -DGGML_CUDA=OFF && cmake --build build-cpu --target llama-server`で作れる
     (i5-12400Fで生成毎秒10トークン、1回20〜40秒)
   - 関連する設定: `KLEIN_LOCAL_LLM_SERVER`(GPU用)、`KLEIN_LOCAL_LLM_MODEL`、`KLEIN_LOCAL_LLM_MMPROJ`、`KLEIN_LOCAL_LLM_PORT`(既定7865)、`KLEIN_LOCAL_LLM=0`で無効

| `upsampler` | 使うLLM |
|---|---|
| `auto`(既定) | LLMサーバ → ローカルのgemma-4-E4B |
| `remote` | LLMサーバだけ |
| `local` | ローカルのgemma-4-E4Bだけ |
| `pup-int8` | 公式upsamplerのINT8版(BFL公式) |
| `pup-gguf` | 公式upsamplerのGGUF版(Q4_K_M、コミュニティ製) |

#### 公式のupsamplerを使う

公式のupsampler([flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)、Qwen3.5-9Bベース、Apache-2.0)は、FLUX 3のために調整されたLLMです。
参照画像の構図を保たせる書き方や場面の作り込みはgemma-4より強い一方、依頼にない文字を書き足すことがあります。
bf16版は18.8GBあるため、このアプリでは量子化版を使います。どちらも**8GBの目標を超えるVRAMを使う前提**の機能です。

| | `pup-int8` | `pup-gguf` |
|---|---|---|
| ファイル | BFL公式の`qwen3.5_9b_flux3_prompt_upsampler_int8_convrot`(9.8GB、[flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)) | コミュニティ製のQ4_K_M+mmproj(6.5GB、[rekuen/flux-3-image-klein-pup-GGUF](https://huggingface.co/rekuen/flux-3-image-klein-pup-GGUF)) |
| 動かし方 | `pup_server.py`(Transformers+comfy-kitchenのINT8カーネル) | llama.cppのllama-server |
| 必要な空きVRAM(既定) | 12GiB(`KLEIN_PUP_INT8_MIN_FREE_GIB`) | 8GiB(`KLEIN_PUP_GGUF_MIN_FREE_GIB`) |
| 実測(RTX PRO 4000 24GB) | ピーク9.4GiB、起動込みで1回約19秒 | 約6.8GiB、起動込みで1回約8秒 |

- 使うたびにGPUで起動し、使い終わったらすぐ止めます(VRAMを画像モデルに返すため)。
- 空きVRAMが足りないときは、画像モデルを一時的に解放してから起動します(画像モデルは次の生成で読み込み直します)。
- 取得: `./setup.sh flux3 --acknowledge-flux3-license`の後に、`.venv/bin/python download_models.py flux3 --acknowledge-flux3-license --pup int8`(`gguf`または`all`も可)
- 関連する設定: `KLEIN_PUP_INT8_PORT`(既定7867)、`KLEIN_PUP_GGUF_PORT`(既定7866)。GGUF版はGPU向けのllama-server(`KLEIN_LOCAL_LLM_SERVER`)を使います。

CUDA 13対応環境と`comfy-kitchen`のConvRotカーネルが必要です。

## セットアップ

Python 3.12、NVIDIA CUDA対応GPU、十分なRAMと空きディスクが必要です。`.venv`、`models`、`outputs`、`uploads`はGit管理されません。

### 1. セットアップ(依存ライブラリとモデル)

[通常版の公式配布ページ](https://huggingface.co/black-forest-labs/flux-3-image-klein)と[FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)を読み、Hugging Face上でアクセス条件に同意します。
モデルはアクセス制限付きなので、`hf auth login`でログインするか、環境変数`HF_TOKEN`を設定してから実行します。

```bash
./setup.sh flux3 --acknowledge-flux3-license
```

`setup.sh`がすることは次の3つです。完了したら終了し、サーバーは起動しません。

1. `.venv`がなければ`python3.12 -m venv .venv`で作る
2. `requirements.txt`の依存ライブラリをインストールする
3. `download_models.py`でモデルを`models/`へダウンロードする(本体、INT8版、アウトペイントLoRA)

依存ライブラリだけ準備する場合は`./setup.sh none`を使います。2回目以降の起動ではセットアップは不要です。
公式のupsamplerも使う場合は、続けて`.venv/bin/python download_models.py flux3 --acknowledge-flux3-license --pup int8`(または`gguf`、`all`)で取得します。

### 2. FastAPI サーバーの起動

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh
```

`run.sh`は`.venv`のPythonで`uvicorn app:app`を起動します(ワーカー1つ)。`Uvicorn running on http://127.0.0.1:7864`と表示されたら起動完了で、`Ctrl+C`で停止します。
モデルは起動時ではなく、最初の生成のときに読み込みます(数秒)。

起動の確認:

```bash
curl http://127.0.0.1:7864/api/health
```

起動時に渡せる主な環境変数:

| 変数 | 既定値 | 内容 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | | 使うGPUの番号 |
| `KLEIN_HOST` | `127.0.0.1` | 待ち受けるアドレス |
| `KLEIN_PORT` | `7864` | ポート |
| `KLEIN_MEMORY_BUDGET_GIB` | `8.0` | PyTorchのGPU割り当て上限(GiB) |
| `KLEIN_PROMPT_LLM_URLS` | | prompt upsamplerに使うLLMサーバ(OpenAI互換、カンマ区切り) |
| `KLEIN_UPSAMPLER` | `auto` | 既定のupsampler(`auto` / `remote` / `local` / `pup-int8` / `pup-gguf`) |

```bash
KLEIN_PROMPT_LLM_URLS=http://127.0.0.1:8080/v1 CUDA_VISIBLE_DEVICES=0 ./run.sh
```

その他の環境変数は[APIリファレンス](docs/API_REFERENCE.md#環境変数)を参照してください。

### 3. Web アプリ(GUI)の起動

Webアプリは手順2のサーバーが同じポートで配信するので、別に起動するものはありません。ブラウザで次を開きます。

- Webアプリ: http://127.0.0.1:7864/
- APIの画面(Swagger UI): http://127.0.0.1:7864/docs

画面の左は入力(参照画像・依頼文・prompt upsamplerの設定)、右は「生成結果」と「ギャラリー」のタブで切り替えます(`http://127.0.0.1:7864/#gallery`で直接ギャラリーを開けます)。
左は常に表示されるので、生成結果やギャラリーの画像を左の参照画像へドラッグ&ドロップして、次の生成の参照に使えます(「参照に追加」ボタンでも可)。

- 入力: レイアウト指定(バウンディングボックス)は畳んであり、必要なときに開きます。
- 生成結果: 出力サイズの縦横比で表示し、スクロールしても画面上に残ります。画像はダウンロードできます。
- ギャラリー: これまでに生成した画像を新しい順に並べます。クリックで拡大し(プロンプト・シード・使ったupsamplerも表示、←→で前後へ)、ダウンロード・参照に追加・削除ができます。削除すると、設定のJSONと参照画像も消えます。

サーバーは既定で`127.0.0.1`だけで待ち受けるため、同じPCのブラウザからしか開けません。
別のPCから使う場合は、SSHのポート転送(`ssh -L 7864:127.0.0.1:7864 <サーバー>`)を使うか、`KLEIN_HOST=0.0.0.0`で起動します。
後者は認証がないので、信頼できるネットワークの中だけで使ってください。

### 外部への送信

画像生成はローカルファイルだけで動作します。prompt upsamplerは標準で有効なので、`KLEIN_PROMPT_LLM_URLS`を設定している場合は、生成のたびにプロンプトと(編集時は)参照画像がその送信先のLLMサーバへ送られます。
`KLEIN_PROMPT_LLM_URLS`を設定しなければLLMサーバは使わず、ローカルLLM(llama.cpp)だけで展開するので、外部へは送信しません。`local`・`pup-int8`・`pup-gguf`を選んだときと、バイパス(`upsample=false`)したときも送信しません。

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
- `GET /api/gallery` / `DELETE /api/gallery/{id}`: 生成画像の一覧と削除

使い方は[API仕様書](docs/API.md)、全エンドポイントの入出力とエラーは[APIリファレンス](docs/API_REFERENCE.md)を参照してください。起動中はSwagger UI（`/docs`）、ReDoc（`/redoc`）、OpenAPI JSON（`/openapi.json`）も利用できます。

## ライセンス

### このアプリ

Copyright 2026 YUZUKI

このリポジトリで作成したアプリケーションコードは [Apache License 2.0](LICENSE) で提供します。
著作権表示と第三者の表示は [NOTICE](NOTICE) にあります。再配布するときは、Apache License 2.0 の条件に従い
`LICENSE` と `NOTICE` を同梱してください。

```text
Copyright 2026 YUZUKI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### モデル(このリポジトリには含みません)

モデル、モデル重み、トークナイザー、依存ライブラリには、それぞれ提供元のライセンスが適用されます。
Apache License 2.0 はモデルには及びません。

| モデル | ライセンス |
|---|---|
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)(INT8 版は [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)) | [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)、[利用ポリシー](https://bfl.ai/legal/usage-policy) |
| [FLUX 3 Image Klein Outpaint LoRA](https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA)(コミュニティ製、任意) | FLUX Kommunity License |
| [flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)(既定は`system_prompts.json`のみ取得。`--pup int8`で設定ファイルも取得) | Apache License 2.0 |
| 公式upsamplerのINT8版(`--pup int8`、[flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)内) | 元のモデルはApache License 2.0。配布元のリポジトリはFLUX Kommunity License v1.0 |
| [rekuen/flux-3-image-klein-pup-GGUF](https://huggingface.co/rekuen/flux-3-image-klein-pup-GGUF)(`--pup gguf`、コミュニティ製) | Apache License 2.0 |
| [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)(比較対象。アプリでは未使用) | FLUX Kommunity License v1.0 |

FLUX 3 Image Klein の公式の表示(モデルリポジトリの `NOTICE.md`)をそのまま記載します。

> This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

このアプリは Black Forest Labs の公式製品ではなく、同社の承認を受けたものでもありません。
詳細は [第三者ライセンスと注意事項](THIRD_PARTY_NOTICES.md) を参照してください。実際の利用では、リンク先の最新の原文が優先されます。
