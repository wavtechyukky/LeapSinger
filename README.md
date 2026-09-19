# LeapSinger

**English**: [README.en.md](README.en.md)

LeapSinger は、CPUでも極めて高速かつ、安定した周期性成分の生成を行える歌声合成のためのDiffusion系の音響モデルです。音素・その長さ（持続長）・音高（F0）を受け取り、メルスペクトログラムを生成します。学習には音素タイミングを記した音素ラベルと音声のみを必要とし、言語による依存はありません（ただし、サンプルには全て日本語のデータを用いています）。

![LeapSinger の概要](doc/fig/leapsinger_overview.png)

**▶ デモを聴く： https://wavtechyukky.github.io/LeapSinger/demo/**

## LeapSinger とは

多くの拡散モデルは、ランダムなノイズから出発して、何ステップもかけて少しずつmelを描いていきますが、LeapSinger は**F0 から作った「擬似mel」（インパルス波形にホワイトノイズを足したもの）を出発点にして、rectified flow で1ステップでmelを仕上げます。** （綺麗な周期的成分の生成方法の学習に時間をかけることなく、一足飛びに高品質なmelを生成するためLeapSingerと命名）

擬似melはv/uv対応モデルと非対応モデルで異なっており、v/uv対応モデルは意図的に無声区間を作ることができます。

![v/uv あり・なしの擬似mel](doc/fig/pseudo_mel_vuv.png)

非対応モデル（上）は全フレームに倍音を敷きますが、対応モデル（下）は無声フレームで倍音をゲートします（暗い縦帯が無声区間）。

多くの検討の結果、高品質な歌声合成用のニューラルボコーダーはmelの質感に敏感であり、かつ、音響モデルは周期的成分を綺麗に生成することが課題であることが分かりました。LeapSingerは出発点がノイズではなく、すでに音高の形を持った擬似melなので、最も困難な周期的成分の描写の学習を避けることができます。この設計から、次の効果が得られます。

- **高品質な周期的成分** — ノイズの少ない高品質な周期的成分を安定して描画することができます。これは、ニューラルボコーダーによる合成結果の質感や話者の再現性に良い結果をもたらします。
- **速さ** — Reverse stepは1回のみであり、1コアのCPUで生成を行ってもRTFは0.03を切ります。なお、ステップ数を増やすことはできますが、1step以上行うとかえってGround truthから離れていきます。

このほかの機能:

- **多話者対応** — 話者IDで声を切り替えられます。例として、日本語3話者（御丹宮くるみ/夏目悠李/波音リツ）のモデルを配布します。
- **スタイル変換** — 同じ話者で歌い方（スタイル）を切り替えられます。

## デモ

**https://wavtechyukky.github.io/LeapSinger/demo/**

- 3話者を学習したモデルで、GT（本物の録音）と生成結果を並べて比較しています。なお、単一の話者で学習させるとわずかにmelの話者の再現性が上がりますが、合成品質に大きな差は見られませんでした。
- スタイル変換のデモを載せています。

## 性能

CPUで計測したRTF（Real-Time Factor。小さいほど速く、1未満なら実時間より速い）です。

音響モデル1本あたりのRTFを、Python nativeとONNXでコア数ごとに比較したものです。

| コア数 | Python native | ONNX |
|:--:|--:|--:|
| 1 | 0.023 | 0.066 |
| 2 | 0.021 | 0.047 |
| 4 | 0.020 | 0.042 |
| 8 | 0.021 | 0.060 |
| 10 | 0.021 | 0.062 |

- **Python native** は1ステップなのでコア数にほぼ依存せず、1コアでも RTF 0.023（実時間の約43倍速）です。
- **ONNX** は onnxruntime のオーバヘッドで native より数倍遅くなりますが、それでも実時間の15倍以上の速さです。4コアが最速で、8コア以上ではかえって遅くなります。励起の倍音和を逐次加算にしてメモリを入力長から切り離しているため、コア数を増やしてもここが並列化されないためです。
- **NHVSing ボコーダー** は CPU で RTF 0.1 未満です（詳細は NHVSing のリポジトリを参照）。

（計測条件：Apple Silicon 10コア・onnxruntime CPU・実データの7.0秒フレーズ・9回の中央値。ONNX は `export/cli.py` の既定設定（`--variant diffsinger --hop 512`）で書き出したものです。機種によって変わります。）

フレーム設定は44.1kHz・hop size256です。hop size512 には、隣り合う2フレームの平均を取ることで対応します。

## 構成

上の図の流れは、次のとおりです。

1. **入力** — 音素・持続長・F0（必要なら話者ID）。
2. **Encoder** — 音素を埋め込み、持続長にしたがってフレーム長へ引き伸ばし、F0 と話者を足してconditionを作ります。
3. **励起（Harmonic + Noise Excitation）** — F0から、インパルス波形にホワイトノイズを足した「擬似mel」を作ります。これがflowの出発点です。
4. **Rectified Flow（1ステップ）** — 擬似メルを出発点に、conditionを条件にして、1ステップで本物らしいmelへ変換します。
5. **NHVSing** — melを音声（波形）に変換する、CPUでRTF0.1を切り、高音質なニューラルボコーダーです。 https://github.com/wavtechyukky/NHVSing/

擬似melは、倍音の減衰・本数・ホワイトノイズの強さを調整できます。しかしながら、検討を重ねた結果、インパルス波形はナイキスト周波数まで重ね切った方が品質に貢献します。

## 使い方

### 環境構築

Python 3.10 以上。リポジトリを clone して editable install します。

    git clone https://github.com/wavtechyukky/LeapSinger
    cd LeapSinger
    pip install -e .                  # 推論・再合成・ノートブック
    pip install -e ".[train]"         # 学習（TensorBoard を追加）
    pip install -e ".[export]"        # ONNX 書き出し
    pip install -e ".[train,export]"  # 全部入り

- **PyTorch** は環境に合わせて入れるのが確実です（CPU版 / CUDA版）。GPU で学習するなら [PyTorch 公式](https://pytorch.org/) の手順で CUDA 版を入れてください。
- **F0抽出（RMVPE）** のウェイトは初回実行時に自動ダウンロードされます（HuggingFace → `preprocess/algorithms/rmvpe.pt`）。
- **ボコーダー（NHVSing）** は `checkpoints/` に ONNX 同梱済みで、追加ダウンロード不要です。
- 学習・配布用の**音響モデル本体は Release で配布**しています（リポには含みません）。

### config設定

学習や書き出しの設定は yaml で行います（`configs/` に例があります）。yaml は `mel` / `model` / `excitation` / `train` / `gan` / `data` の6つのセクションに分かれています。主な項目は次のとおりです。

- `model` — `spk_dim`（0より大きいと多話者）、`n_speakers`、`n_styles`（0より大きいとスタイル機能を使う）、`use_uv`（v/uvを条件に使うか）
- `excitation` — `n_harm`（倍音の本数）、`harm_decay`（倍音の減衰）、`noise_ratio`（ホワイトノイズの強さ）
- `train` — `lr`、`max_updates`、`num_steps`（推論のステップ数。**1** を推奨）、`balance_speakers`（話者を均等にサンプリング）など
- `gan` — 質感を鮮明化する GAN の設定。`enabled`（`false` で flow損失＋mel損失のみの学習）、`gan_start_step`（GAN を入れ始める step）、`gan_strength`（敵対的損失の強さ）など
- `data` — `spk_map` / `style_map`（データセットのフォルダ名 → 話者ID / スタイルID の対応）

### 辞書の作成

日本語の音素の一覧は `dict/ja.phonemes` にあります（1行に1音素、並び順がそのままID、先頭の `pau` が ID 0、`#` から先はコメント）。日本語以外や独自の音素を使いたいときは、同じ形式のファイルを用意し、各コマンドに `--phonemes <ファイル>` で渡します（省略時は日本語の `dict/ja.phonemes` を使います）。

### preprocess

データセット1つにつき、yaml（`configs/recipes/<db>.yaml`）を用意します。1曲あたり、音声 `wav` と音素タイミングの `.lab`（＋譜面）を使います。次のコマンドで `data/<db>/` に前処理済みのデータができます。

    python -m preprocess.run --recipe configs/recipes/<db>.yaml

例に使った3つのデータベースは、次のスクリプトでダウンロードできます（各DBの規約に従ってご利用ください）。

    python preprocess/download_scripts/download_oniku.py
    python preprocess/download_scripts/download_natsume.py
    python preprocess/download_scripts/download_ritsu.py

F0の抽出にはRMVPEを使います（RMVPEはマルチプロセスで動かさないようご注意ください）。

### training

学習は2段構成です。前半は flow損失＋mel損失で土台の声を学習し、後半で GAN を入れて質感を鮮明化します（切り替えは config の `gan` セクションで設定。`gan.enabled: false` なら flow損失＋mel損失のみ）。

    python -m train --config configs/<name>.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

同じコマンドをもう一度実行すると、途中から自動で再開します。

### export

    python -m export.cli \
      --ckpt log/<run>/ckpt_050000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 256 --speaker bake --spk-id 0

話者の指定（bake / embed / なし）など詳しい説明は、下の「ONNX への書き出し」を参照してください。

ノートブックで、書き出しから利用まで一通り試すことができます。必要なモデルは Release からダウンロードして `notebooks/sample_data/` に置いてください（詳しくは同フォルダの `place_model_here.txt`）。

    notebooks/export_and_use_onnx.ipynb

ノートブックでは、音響モデルを単一のONNXに書き出し、一通り動かします（音素＋持続長＋F0 → mel → 音声）。

### ONNX への書き出し

`export/` は、チェックポイントを自己完結した ONNXグラフに変換します。励起と1ステップのflowはグラフに焼き込まれるので、使う側は音素・持続長・F0を渡すだけで済みます。話者の扱いは、次から選べます。

- **焼き込み（bake）** — 1つの声を固定します。グラフがいちばん単純になります（話者入力なし）。
- **埋め込み（embed）** — 話者ベクトルを入力にします。1つのグラフでどの声にも切り替えられます。
- **なし（none）** — 単一話者モデル向けです。話者の入力も焼き込みもしません（話者の概念がないグラフ）。多話者モデルではbakeかembedを使います。

### ボコーダー

`checkpoints/` に NHVSing ボコーダーを2つ同梱しています。

- `nhv_v3_2.onnx` — hop size256のmelとF0を受け取ります。
- `nhv_v3_2x.onnx` — hop size512のmelとF0を受け取ります。

同梱しているのは **V3.2** です（高音域でフレーム単位に波形が急激に弱まる現象を、LTV フィルタの重ね合わせを Hann 窓化して解消した最新のウェイト。入出力の仕様は V3.1 と同一なので差し替えるだけで使えます。詳細は [NHVSing](https://github.com/wavtechyukky/NHVSing/) を参照）。

## オプション

- **辞書** — 任意の音素辞書を指定できます。日本語以外にも対応できる設計です（多言語対応そのものは今後の課題です）。
- **v/uv の扱い** — 有声・無声（v/uv）を条件に使うモードと、隙間を線形補間で埋めた連続F0だけを使うモードを選べます。
- **励起の調整** — 倍音の減衰・本数・ホワイトノイズの強さを変えられます。
- **学習レシピ** — 学習の条件は yaml で指定します（使用するデータセット、話者ID、話者ごとのスタイルやデータなど）。

## 今後の課題

- **多言語対応** — 現状は日本語データで検証していますが、設計自体は言語に依存しません。他言語の音素辞書・データへの対応を行い、1つのモデルで複数の言語を扱えることを目標とします。
- **さらなる品質向上** — 擬似melの生成方法やパラメータの調整などで話者再現性に改善の余地があると考えています。

## ライセンス

コードは MIT です（`LICENSE`）。ただし次のものは MIT の対象外で、それぞれのライセンス・規約に従います。対応表と詳細は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) にまとめてあります。

- **`preprocess/algorithms/base.py` と `preprocess/algorithms/rmvpe.py`** — どちらも [pitch-benchmark](https://github.com/lars76/pitch-benchmark)（MIT, Copyright (c) 2025 Lars Nieradzik）からそのまま取り込んだファイルです。著作権表示は `LICENSES/pitch-benchmark-MIT.txt` にあり、再配布時はこれを保持してください。`base.py` は本リポジトリと同じ MIT です。
  `rmvpe.py` は違います。中の RMVPE モデルは [上流](https://github.com/Dream-High/RMVPE)が **Apache-2.0** なので、このファイルだけ Apache-2.0 で配布します（全文 `LICENSES/Apache-2.0.txt`）。`to_local_average_cents()` は [CREPE](https://github.com/marl/crepe) 由来で、その MIT 表示は `LICENSES/crepe-MIT.txt` にあります。ウェイト `rmvpe.pt` は初回実行時にダウンロードするもので、本リポジトリには含みません。
- **デモ・サンプル音声**（`demo/audio/*_gt.ogg`、`notebooks/sample_data/*.wav`） — 合成音ではなく歌声データベースの実録音の抜粋です。各データベースの規約に従います。
- **同梱のボコーダー ONNX**（`checkpoints/nhv_v3_2*.onnx`） — [NHVSing](https://github.com/wavtechyukky/NHVSing/) の成果物です。
- **Release で配布する学習済みモデル**とその学習に使った歌声データベース — モデル配布物の `CREDITS.txt` を参照してください。

## 謝辞

本モデルの学習に使わせていただいたデータセットと、関連するプロジェクトに感謝します。

- 御丹宮くるみ歌声データベース（御丹宮くるみ） — https://onikuru.info/db-download/
- 夏目悠李（歌声DB制作: アマノケイ／音声提供者: 霧野蒼太） — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- 波音リツ — https://www.canon-voice.com/voicebanks/
- Neural Homomorphic Vocoder — https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html
- dsp（zjlww） — https://github.com/zjlww/dsp
- pitch-benchmark（Lars Nieradzik。`preprocess/algorithms/` の 2 ファイルはここからの取り込みです） — https://github.com/lars76/pitch-benchmark
- RMVPE（F0 抽出モデル本体） — https://github.com/Dream-High/RMVPE
- CREPE（RMVPE 経由で `to_local_average_cents()` を利用） — https://github.com/marl/crepe
- DiffGAN-TTS（JCU 判別器の設計を参考にしました。コードは自前実装です） — https://github.com/keonlee9420/DiffGAN-TTS

配布する多話者モデルには、各データベースの規約に従って上記のクレジットを表示します。夏目悠李については **歌声DB制作: アマノケイ／音声提供者: 霧野蒼太** を表示し、「夏目悠李の出力音声に関する利用規約」をモデル配布物に同梱します。
