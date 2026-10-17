# テクニカルガイド

日本語 | [English](TECHNICAL_GUIDE.en.md)

FLUX 3 Image Klein Reference Studio が、FLUX 3 Image Klein を Diffusers で 8GiB の VRAM に収めて動かすために使っている手法をまとめます。
API の使い方は [API 仕様](API.md)、全項目は [API リファレンス](API_REFERENCE.md) を参照してください。

数値は、特記がなければ RTX PRO 4000 Blackwell(24GB)、Core i5-12400F、INT8、16 ステップ、タイムステップのシフト 16 での実測です(2026-10-18)。

## 目次

1. [全体の構成](#1-全体の構成)
2. [INT8 ConvRot の Transformer を Diffusers に載せる](#2-int8-convrot-の-transformer-を-diffusers-に載せる)
3. [段階式のパイプライン](#3-段階式のパイプライン)
4. [メモリの設計](#4-メモリの設計)
5. [サンプリングの設定](#5-サンプリングの設定)
6. [進捗の表示と停止](#6-進捗の表示と停止)
7. [アウトペイントと位置指定](#7-アウトペイントと位置指定)
8. [prompt upsampler](#8-prompt-upsampler)
9. [公式 upsampler の量子化版をローカルで動かす](#9-公式-upsampler-の量子化版をローカルで動かす)
10. [サーバーと GUI](#10-サーバーと-gui)
11. [制限と注意](#11-制限と注意)

---

## 1. 全体の構成

```text
依頼文(+ 参照画像)
   │
   ▼
prompt upsampler ── LLM サーバ / ローカル gemma-4-E4B / 公式 upsampler(INT8・GGUF)
   │  完全なシーンのキャプション
   ▼
テキストエンコーダ Qwen3-VL-4B(bf16、CPU メモリ → GPU で 1 層ずつ計算)
   │  prompt_embeds
   ▼
VAE エンコード(参照画像、GPU)
   │
   ▼
ノイズ除去 Transformer 5.6B(公式 INT8 ConvRot、GPU 常駐)× 16 ステップ
   │
   ▼
VAE デコード(GPU、1024px 超はタイル分割) → PNG
```

| 部品 | 形式 | 置き場所 |
|---|---|---|
| Transformer | BFL 公式の INT8 ConvRot(ComfyUI 向け配布、約 5.4GB) | GPU に常駐 |
| テキストエンコーダ | Qwen3-VL-4B、bf16(約 8GB) | CPU メモリ。計算時に 1 層ずつ GPU へ |
| VAE | FLUX.2 VAE、bf16 | GPU |
| prompt upsampler | 外部 LLM(OpenAI 互換 API)または公式 upsampler の量子化版 | 別プロセス |

FastAPI のサーバー 1 つが、API(`/api/*`)と GUI(`/`)を同じポートで配信します。生成は 1 件ずつ、ジョブ方式で実行します。

## 2. INT8 ConvRot の Transformer を Diffusers に載せる

BFL は FLUX 3 Image Klein の INT8 版を、ComfyUI 向けの 1 ファイル(`flux-3-image-klein-comfy` の
`diffusion_models/flux3-image-klein-int8-convrot.safetensors`)として配布しています。Diffusers は FLUX 3 に対応していますが、
この形式の読み込みには対応していないため、変換せずに直接つなぐ読み込み処理を書いています(`flux3_int8.py`)。

**ファイルの中身**: 量子化された全結合層ごとに、`weight`(int8)、`weight_scale`(float32、行ごと)、`comfy_quant`
(`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` を表すバイト列)が入っています。

**ConvRot**: 入力の次元をグループ(256 など)に分け、正規化したアダマール行列 H で回転してから量子化する方式です。
重みはオフラインで `W_rot = W · Hᵀ`(グループごと)にして保存され、推論時は活性化を同じ H で回してから INT8 の行列積を取ります。
外れ値が回転で均されるので、INT8 でも精度が保たれます。

**読み込みの手順**

1. `Flux3Transformer2DModel` を空の重み(`accelerate.init_empty_weights`)で作る。
2. `comfy_quant` を持つ層を、`comfy-kitchen` の `int8_linear(..., convrot=True)` を呼ぶ自作の `ConvRotInt8Linear` に差し替える。
   int8 の重みと float32 のスケールは、そのまま保持する(逆量子化しない)。
   ComfyUI の層名を Diffusers の層名に対応させる(`qkv` は `to_q` / `to_k` / `to_v` の 3 つに分割)。
3. 残りの浮動小数点の重みは、Diffusers の `convert_flux3_transformer_checkpoint_to_diffusers` で名前を合わせて読み込む。
4. 空の重み(meta テンソル)が残っていないことを確かめる。
5. `.to("cuda")` で移す(dtype は指定しない。float32 のスケールを落とさないため)。

## 3. 段階式のパイプライン

Diffusers の FLUX.3 は **ModularPipeline** で、テキストエンコード・VAE エンコード・ノイズ除去・デコードのブロックに分かれています。
このアプリはブロックごとに別のパイプラインを作り、部品ごとに置き場所を変えて、状態(`PipelineState` のテンソル)を受け渡しながら順に実行します。

```python
blocks = ModularPipeline.from_pretrained(model_dir)._blocks.sub_blocks
text_pipe = blocks["text_encoder"].init_pipeline(model_dir)
denoise_pipe = blocks["denoise"].init_pipeline(model_dir)
denoise_pipe.register_components(transformer=int8_transformer)   # 2 章の INT8 Transformer
vae_encode_pipe = blocks["vae_encoder"].init_pipeline(model_dir)
decode_pipe = blocks["decode"].init_pipeline(model_dir)

state = text_pipe(prompt=caption)
state = move_tensors(state, "cuda")
if references:
    state = vae_encode_pipe(state=state, image=references)
state = denoise_pipe(state=state, width=w, height=h, num_inference_steps=16, generator=g)
image = decode_pipe(state=state, output="images", output_type="pil")[0]
```

蒸留版は CFG を使わないので、1 ステップにつき Transformer を 1 回だけ呼びます。

## 4. メモリの設計

### 4.1 割り当ての上限

`torch.cuda.set_per_process_memory_fraction` で、PyTorch がこのプロセスで確保できる GPU メモリを制限しています。
既定は 8GiB(`KLEIN_MEMORY_BUDGET_GIB`)で、12GB 級のカードで動かすことを想定した値です。
上限があると、PyTorch はキャッシュを早めに手放すので、上限がない場合より実際のピークは少し下がります
(1024×1024 の T2I は、上限なしで 8.19GiB、上限 8GiB で 7.81GiB)。

### 4.2 テキストエンコーダを 1 層ずつ GPU で計算する

テキストエンコーダ(Qwen3-VL-4B、bf16、約 8GB)は、Transformer と同時に GPU に載せると 8GiB を超えます。
丸ごと GPU に載せると、それだけでピークが 8.6GiB です。

最初は CPU で計算していましたが、prompt upsampler が書く 200〜270 語のキャプションでは、エンコードに約 30 秒かかり、
1024×1024 の生成時間の 8 割を占めていました。そこで、accelerate の `cpu_offload` を使い、重みを CPU メモリに置いたまま、
計算のときに 1 層ずつ GPU へ送るようにしています(逐次オフロード)。

| | CPU で計算 | 1 層ずつ GPU で計算 |
|---|---|---|
| エンコード時間(約 400 トークン) | 29.6 秒 | 2.2〜2.6 秒 |
| 追加の VRAM | 0 | 約 0.95GiB |
| 結果の差(CPU の結果に対する相対誤差) | — | 0.7% |

`KLEIN_TEXT_ENCODER=cpu` で、CPU 計算に戻せます。

### 4.3 VAE のタイル分割

2048×2048(4MP)では、ノイズ除去のピークが約 8.2GiB なのに対して、最後の VAE デコードだけが約 15.3GiB 使います。
1024px を超える画像では VAE をタイル分割しています(`enable_tiling()`、1024px のタイル、重なり 25%)。

| 4MP のデコード | 分割なし | 分割あり |
|---|---|---|
| ピーク | 15.41GiB | 8.31GiB |
| 時間 | 1.43 秒 | 1.75 秒 |

分割の有無で画素は平均 8.4 ほど違いますが、差は輪郭に薄く分布していて、継ぎ目は見えません。
1024px 以下ではタイル分割は働かず、結果は変わりません。

### 4.4 2K 出力のときだけ上限を上げる

BFL は 2K(約 4MP)での利用を推奨しています。タイル分割をしても、4MP のノイズ除去は 8GiB をわずかに超えます。
そこで、出力が 1024×1024(1,048,576 画素)を超えるジョブだけ、上限を 10GiB(`KLEIN_HIRES_MEMORY_BUDGET_GIB`)に上げ、
次の 1MP 以下のジョブで 8GiB に戻します(戻すときに `torch.cuda.empty_cache()`)。
GPU の容量が上限より小さければ、生成を始める前にエラーにします。

### 4.5 参照画像の画素数

参照画像は、ノイズ除去のトークン数を増やします。BFL の推論コードは 1 枚あたり最大 1MP ですが、8 枚を 1MP ずつにすると
ピークが 11.2GiB になります。このアプリは、1 枚あたり最大 1MP、合計 2MP まで(`KLEIN_REFERENCE_TOTAL_MP`)に、
縦横比を保って縮小します(1〜2 枚は各 1MP、4 枚は各 0.5MP、8 枚は各 0.25MP)。

### 4.6 実測

| 処理 | 出力 | 生成時間 | ピーク(割り当て) | 上限 |
|---|---|---|---|---|
| T2I | 1024×1024 | 11.7 秒 | 7.81GiB | 8GiB |
| 参照 1 枚の編集 | 768×1024 | 18.1 秒 | 7.59GiB | 8GiB |
| 参照 2 枚 | 1024×768 | 29.8 秒 | 7.59GiB | 8GiB |
| 参照 8 枚(各 0.25MP) | 1024×768 | 28.2 秒 | 7.57GiB | 8GiB |
| T2I(2K) | 2048×2048 | 51.4 秒 | 8.22GiB | 10GiB |
| 参照 2 枚(2K) | 2304×1728 | 83.1 秒 | 9.50GiB | 10GiB |

(テキストエンコードを含み、prompt upsampler の時間は含みません。モデルの読み込みは数秒です)

## 5. サンプリングの設定

| 項目 | 値 |
|---|---|
| ステップ数 | 16(蒸留版) |
| CFG | なし |
| サンプラー | Flow Matching の Euler(`FlowMatchEulerDiscreteScheduler`) |
| タイムステップのシフト(alpha) | 16 |

**タイムステップのシフト**: シグマ σ を `σ' = α·σ / (1 + (α − 1)·σ)` に写す定数のシフトです。α を大きくすると、
ノイズの多い前半(全体の構図を決める段階)に多くのステップを使います。BFL は 2026-10-17 に、推論の α を 6.93 から 16 に変えました
(公式の推論コードと ComfyUI のワークフロー)。Diffusers の FLUX.3 は 6.93 のままなので、生成のたびに
`scheduler.set_shift(16.0)` で合わせています(`KLEIN_TIMESTEP_SHIFT`、API の `shift`)。

1024×1024 では、6.93 と 16 の画質の差ははっきりしませんでした(構図・内容は同じで、細部だけ変わる)。
シフトは高解像度で効く設定なので、2K では公式の値を使うのが安全です。また、ある編集の例では、6.93 で構図が破綻したキャプションが、
16 では破綻しませんでした(1 例だけの観察)。

## 6. 進捗の表示と停止

ブロックごとに実行すると、ノイズ除去のステップごとのコールバックがありません。そこで、Transformer に
`register_forward_pre_hook` を掛け、呼ばれた回数をステップ数として数えています。蒸留版は CFG なしなので、
1 ステップにつき 1 回呼ばれます(16 ステップでちょうど 16 回)。

停止の要求は、このフックの中で例外(`Cancelled`)を投げて受け付けます。生成はステップの区切り(1MP で約 0.5 秒ごと)で止まり、
停止したジョブはモデルを解放します。

## 7. アウトペイントと位置指定

### 7.1 アウトペイント

コミュニティ製のアウトペイント LoRA(`ausboss/FLUX-3-Image-Klein-Outpaint-LoRA`)を使います。

- **LoRA の適用**: INT8 の層は重みを書き換えられないので、`ConvRotLoRALinear` で、INT8 層の出力に bf16 の低ランク補正
  `x · Aᵀ · Bᵀ × 強さ(1.5)` を足します。LoRA を使うときは、LoRA 付きのモデルを読み込み直します。
- **入力**: 参照画像を出力サイズに収まるよう縮小し、灰色(128)の余白の中央に置いたものを参照として渡します。
- **出力**: 生成後、中央の元の画像を貼り戻します(元の画素を保つため)。

### 7.2 位置指定(バウンディングボックス)

モデルカードの書式どおり、キャプションの後に空白を 1 つ挟んで、要素の JSON 配列を付けます。座標は 0〜1000 の
`[top, left, bottom, right]`、各要素はキャプションの中で `<id>` の形で引用します。編集では `from`・`src_bbox`・`tgt_bbox`
(`null` で削除)を使います。prompt upsampler で展開した後に付けるので、upsampler がボックスを書き換えることはありません。

## 8. prompt upsampler

### 8.1 なぜ必須か

FLUX 3 Image Klein は、詳しいシーンのキャプションで学習されています。モデルカードによると、短いプロンプトは平凡な出力になり、
1 行の編集指示は構図まで変えてしまいます。参照画像は中身を固定しますが、構図までは固定しません。
そこで、短い依頼を完全なキャプションに書き直す LLM(prompt upsampler)を前段に置きます。このアプリでは**既定で有効**で、
比較のために通さずに生成する(バイパス、`upsample=false`)こともできます。

### 8.2 使う LLM

| `upsampler` | 使う LLM | システムプロンプト |
|---|---|---|
| `auto`(既定) | LLM サーバ(`KLEIN_PROMPT_LLM_URLS`)→ ローカルの gemma-4-E4B | 外部 LLM 向け |
| `remote` / `local` | どちらか一方 | 外部 LLM 向け |
| `pup-int8` / `pup-gguf` | 公式 upsampler の量子化版(9 章) | 公式 upsampler 用 |

### 8.3 公式 upsampler 用の入力(`pup-*`)

公式の推論コード(`src/flux3/pup.py`)と同じ形にしています。

- システムプロンプト: `flux-3-image-klein-pup` の `system_prompts.json`(t2i / i2i / t2i_bbox / i2i_bbox)
- 編集: 参照画像を先に、続けて 3 行の依頼文
  ```text
  Reference images follow in order (the first image, the second image, ...).
  User request: <依頼文>
  Output the caption.
  ```
- 参照画像: 白で透過を埋め、1 枚なら約 1MP、2 枚なら 724²、3 枚以上は 512² ずつに LANCZOS で縮小し、JPEG(品質 92)にする
- デコード: 貪欲法(temperature 0)、presence penalty は T2I で 1.5・編集で 0、最大 3584 トークン、思考(thinking)はオフ

### 8.4 外部の LLM 向けの入力(`auto` / `remote` / `local`)

BFL は 2026-10-17 に、公式 upsampler 以外の LLM 向けのシステムプロンプト(公式推論コードの
`src/flux3/pup_prompts_external_vlm.json`、Apache-2.0)を公開しました。公式 upsampler は作法を学習済みなので短いプロンプトで済みますが、
外部の LLM には作法を細かく文章で教え込む必要があり、T2I 用は約 7,400 語(約 1 万トークン)あります。

- 編集は、依頼文(`User request: <依頼文>`)を先に、参照画像を後ろに付けます(公式 upsampler 用とは逆)。
- T2I では、先頭に `TARGET CANVAS: 幅:高さ` の行を付けて出力の縦横比を伝えます(両方のシステムプロンプトが受け付ける形)。
- LLM のコンテキスト長は 16K 以上が必要です。llama.cpp のサーバは同じシステムプロンプトの処理結果を使い回すので、
  2 回目からは速くなります(gemma-4-31B で初回 20 秒 → 13〜16 秒、E4B で 8.7 秒 → 3.5〜4 秒)。
- `KLEIN_LLM_PROMPTS=pup` で、外部の LLM にも公式 upsampler 用のプロンプトを使う(2026-10-18 より前の動き)ようにできます。

### 8.5 失敗時の扱いとキャッシュ

- どの LLM も使えなければ、元の文のまま生成し、ジョブに `warning` を付けます。`upsample_strict=true` なら失敗扱いにします。
- 結果は、upsampler・モード・依頼文・縦横・参照画像のハッシュで、直近 128 件をキャッシュします。
- GUI の「展開して確認」で作ったキャプションを書き換えずに生成した場合は、二重に展開しないよう `upsample=false` で送ります。

### 8.6 比べてみて分かったこと

6 課題 × 2 seed での比較では、画質そのものの差は小さく、違いは次の点に出ました。

- 公式 upsampler は、参照画像を「変えない土台」と書き、窓やスクリーンまで列挙するので、参照の構図をよく保ちます。
- 公式 upsampler も、外部 LLM 向けプロンプトを使った gemma-4 も、依頼にない文字(会場名や時刻)を書き足すことがあります。
  文字を正確に入れたい用途では注意が要ります。

## 9. 公式 upsampler の量子化版をローカルで動かす

公式 upsampler(`flux-3-image-klein-pup`、Qwen3.5-9B ベース)は bf16 で 18.8GB あり、FLUX 3 と同時には載りません。
量子化版を、使うときだけ GPU で起動し、使い終わったらすぐ止めて VRAM を画像モデルに返します。
空き VRAM が足りなければ、画像モデルを一時的に解放してから起動します(生成ジョブの中ではいつでも、`/api/enhance-prompt` では生成中でないときだけ)。

| | `pup-int8` | `pup-gguf` |
|---|---|---|
| ファイル | BFL 公式の `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot`(9.8GB) | コミュニティ製の Q4_K_M + mmproj(6.5GB) |
| 実行 | `pup_server.py`(Transformers + comfy-kitchen) | llama.cpp の llama-server |
| VRAM | 約 9.4GiB | 約 6.8GiB |
| 1 回の時間(起動込み) | 約 19 秒 | 約 8 秒 |
| bf16 版との語の一致度 | 0.28〜0.75 | 0.28〜0.66 |

(一致度は 6 課題のキャプションの語の並びの一致率。gemma-4-31B と bf16 版の間は 0.14〜0.23)

### 9.1 INT8 版の読み込み(`pup_int8.py`)

ComfyUI 向けのファイルですが、テンソル名は Transformers の `Qwen3_5ForConditionalGeneration` と同じ並びです。

- 全結合層: 2 章と同じ `ConvRotInt8Linear` に差し替えます。画像入力側の層はバイアスを持ち、回転のグループサイズも 16 / 64 / 256 と
  層によって違うので、それぞれ設定から読みます。
- 単語の埋め込み層: これも INT8 で、しかも回転を掛けた形(`W_rot = W · Hᵀ`)で保存されています。正規化したアダマール行列は
  対称かつ直交なので、`W = W_rot · H` で元に戻せます。表全体(248,320 行)を戻すと大きいので、引いた行だけをその場で
  逆量子化・逆回転する `ConvRotInt8Embedding` にしています。
- 投機的デコード用の MTP ヘッド(`mtp.*`)は使いません。

### 9.2 別プロセスにする理由

画像モデルのプロセスには 8GiB の割り当て上限が掛かっていて(4.1 節)、10GiB 級の upsampler はその中に載りません。
`pup_server.py` は、OpenAI 互換の `/v1/chat/completions` だけを持つ小さなサーバーで、upsampler.py が子プロセスとして起動します。
presence penalty は Transformers にないので、vLLM と同じ「出力済みのトークンから一度だけ引く」処理を自作しています。

### 9.3 子プロセスの後始末

llama-server と `pup_server.py` は、起動時に `prctl(PR_SET_PDEATHSIG, SIGTERM)` を設定し、スタジオが落ちても孤児として残らないようにしています。
ローカルの gemma-4-E4B は、空き VRAM が足りなければ CPU で動かし、CPU のときは数分使われなければ止めます。
CPU では、GPU 向けにビルドした llama-server(`GGML_NATIVE` / `AVX2` なし)だと極端に遅いので、CPU 向けのビルドを別に使います。

## 10. サーバーと GUI

- **ジョブ**: `POST /api/generate` はジョブを作ってすぐ返し、生成は 1 スレッドのワーカーで 1 件ずつ実行します。
  状態は直近 100 件をメモリに保持し、画像とメタデータ JSON は `outputs/` に保存します。
- **メタデータ**: 実際にモデルに渡したプロンプト、元の依頼文、使った upsampler と LLM、シード、シフト、
  テキストエンコーダの計算場所、時間とピーク VRAM を記録します。
- **ギャラリー**: `GET /api/gallery` は `outputs/` の画像を新しい順に返し、`DELETE /api/gallery/{id}` は PNG・メタデータ・参照画像を消します
  (ID は 32 桁の 16 進数だけを受け付け、生成中のジョブは消さない)。
- **GUI**: 左の入力は常に表示し、右を「生成結果」と「ギャラリー」で切り替えます。生成結果やギャラリーの画像を左の参照画像へ
  ドラッグ&ドロップすると、サーバーから画像を取得して参照に追加します。GUI のファイルは `Cache-Control: no-cache` で配信し、
  更新後に古い画面が残らないようにしています。

## 11. 制限と注意

- Blackwell 世代(RTX PRO 4000 / 5000)でだけ確かめています。INT8 の計算は comfy-kitchen の ConvRot カーネルに頼っているので、
  それより前の世代で動くかどうかは、このカーネルの対応次第です。CUDA 13 に対応した環境が必要です。
- 12GB のカードでの動作(1MP は 8GiB、2K は 10GiB の上限)は計算上のもので、実機では試していません。
- テキストエンコーダを置くため、PC のメモリは 32GB 程度必要です。
- 参照画像は 2MP まで(4.5 節)に縮小されるので、細かい部分が描き直されることがあります。

## 参考

- FLUX 3 Image Klein: https://huggingface.co/black-forest-labs/flux-3-image-klein
- ComfyUI 向けの配布(INT8 版): https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
- 公式 upsampler: https://huggingface.co/black-forest-labs/flux-3-image-klein-pup
- 公式の推論コード(`pup.py`、外部 LLM 向けシステムプロンプト): https://github.com/black-forest-labs/flux-3-image-klein
- アウトペイント LoRA: https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA
