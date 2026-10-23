# 音量正規化チューナー（loudness tuner）

学習目標の mel に掛ける「遅い自動ゲイン」（`leapsinger/loudness.py`、config の `data.loudness_agc`）のパラメータを、
前処理済みのデータセットで目視・試聴しながら決めるためのローカル Web アプリです。学習（`dataset.py`）と同じ関数で
計算するので、ここで見た通りの目標 mel でモデルが学習します。

**English:** a local web app for tuning `data.loudness_agc` (the slow loudness AGC applied to the target mel) on a
preprocessed dataset, with per-phrase envelope/gain plots, before/after mels and A/B playback. Run
`python tools/loudness_tuner/app.py --data data/<db>` and open http://127.0.0.1:8018. The YAML box shows the
current values in the form that goes into your config.

## 起動

```bash
pip install -e ".[tuner]"                            # fastapi・uvicorn（1 回だけ）
python tools/loudness_tuner/app.py --data data/ritsu   # → http://127.0.0.1:8018
```

`--data` にはデータセットのフォルダ（`shard.npz` と `metadata.json` があるフォルダ）か、それらを含む親フォルダ
（例: `data`）を渡します。複数渡せます。省略しても、画面の欄にフォルダを打ち込んで Scan できます。
ほかのオプション: `--port`、`--host`、`--cache_dir`（DB 全体の統計の置き場。既定 `~/.cache/leapsinger/loudness_tuner`）。

## 画面

1. **データ** — フォルダを Scan すると、見つかったデータセットが並びます（フレーズ数・時間・hop・shard の大きさ・
   試聴できる音声の本数）。1 つ選ぶと計算が走ります。
2. **パラメータ** — σ（これより遅い音量変化を取り除く）、target（揃える先のフレーム音量 dB）、強さ k、
   ゲインの上限/下限。「計算」で再計算します（入力しただけでは走りません）。下の **YAML** の枠が今の値を
   config に貼る形で出すので、「コピー」して `data.loudness_agc` に貼ります。
3. **カード** — フレーズごとに、フレーム音量 E[t]（灰）、正規化後の E（薄橙）、遅い包絡の前（青）と後（橙）、
   target（点線）、ゲイン c[t]（紫破線、右軸）、pau 区間の網掛け、音素境界。見出しに平均音量・ゲイン範囲・
   遅い包絡 std の前後。音声があるフレーズは原音と正規化音を再生でき、「このページを連続再生」でフレーズ間の
   レベル差を聴けます。正規化音のピークが 0.99 を超える場合は**両方**を同じ倍率で下げてバッジで知らせます
   （相対レベルは変えない）。「mel を表示」で正規化前後の mel 画像（同じ色スケール）。
4. **DB 全体の分布** — 全フレーズを現在のパラメータで走査し、フレーズ間のレベル差 std、フレーズ内の遅い包絡 std、
   macro/meso/micro の std、|ゲイン| の分位、clamp に当たった割合を前後で表示します。結果はキャッシュされ、
   同じ設定なら即時に出ます。

## 試聴できる音声

再生音はボコーダを通さず、元の波形に同じゲイン曲線を掛けたものです。波形は次の順に探します。

- `<データセット>/shard_wav.npz` — 前処理のレシピで `save_wav: true` にすると作られます（推奨）。
- `<親>/phrase/<データセット名>/<フレーズ>.wav`（`phrase_demo`、`phrase_demo_cand` も）— 古いレイアウト。

どちらも無いフレーズは図だけ出ます（「音声があるフレーズだけ」のチェックを外すと表示されます）。

## ファイル

- `app.py` — FastAPI。`GET /config`, `POST /scan`, `POST /compute`, `GET /audio`, `GET /audio_info`, `GET /mel.png`,
  `POST /stats_start`, `GET /stats_progress`, `POST /stats_cancel`。
- `static/index.html` — UI（vanilla JS、Canvas 2D）。
- 正規化の実装は `leapsinger/loudness.py`（`slow_agc_gain(mel, sung, frame_rate, **params)` が dataloader の入口。
  既定値は `AGC_DEFAULTS` だけが持つ）。テストは `tests/test_loudness.py` と `tests/test_loudness_tuner.py`。
