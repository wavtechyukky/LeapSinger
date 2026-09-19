# Third-party notices

LeapSinger is released under the MIT license (see [`LICENSE`](LICENSE)). A few things
shipped in, or fetched by, this repository are **not** ours and carry different terms.
They are listed here. If you redistribute LeapSinger, carry these notices with it.

| What | Where | Terms |
|---|---|---|
| Pitch-algorithm base classes | `preprocess/algorithms/base.py` | MIT (pitch-benchmark) |
| RMVPE pitch estimator | `preprocess/algorithms/rmvpe.py` | Apache-2.0, plus the MIT notices below |
| RMVPE weights `rmvpe.pt` | downloaded at run time, **not** in this repo | see §3 |
| Neural vocoder | `checkpoints/nhv_v3_2.onnx`, `checkpoints/nhv_v3_2x.onnx` | NHVSing project |
| Demo and sample audio | `demo/audio/*_gt.ogg`, `notebooks/sample_data/*.wav` | each singing database's terms of use |
| Acoustic-model checkpoints | GitHub Releases, **not** in this repo | each singing database's terms of use |

Everything else in this repository is our own work and is MIT.

---

## 1. pitch-benchmark — `preprocess/algorithms/base.py` and `rmvpe.py` (MIT)

Both files were taken **verbatim** from

> **pitch-benchmark** — <https://github.com/lars76/pitch-benchmark>
> The MIT License (MIT), Copyright (c) 2025 Lars Nieradzik

- `preprocess/algorithms/base.py` is their `algorithms/base.py`: `PitchAlgorithm`,
  `ContinuousPitchAlgorithm` and `ThresholdPitchAlgorithm`.
- `preprocess/algorithms/rmvpe.py` is their `algorithms/rmvpe.py`: `MelSpectrogram`,
  `ConvBlockRes`, `ResEncoderBlock`, `ResDecoderBlock`, `Encoder`, `Intermediate`,
  `Decoder`, `DeepUnet0`, `BiGRU`, `E2E0`, `to_local_average_cents`, `get_model_path`
  and `RMVPEPitchAlgorithm`.

Their MIT text and copyright notice are in
[`LICENSES/pitch-benchmark-MIT.txt`](LICENSES/pitch-benchmark-MIT.txt), and MIT requires
that notice to travel with the code.

**Our modifications**

- `base.py`: `PitchAlgorithm` no longer clips the estimated pitch to `[fmin, fmax]`.
  RMVPE never receives those bounds, so clipping pinned out-of-range frames to the
  boundary and invented flat sustained notes rather than rejecting them.
- `rmvpe.py`: the notice header was added. The code is unchanged.

## 2. RMVPE and CREPE — inside `rmvpe.py` (Apache-2.0, MIT)

The model that pitch-benchmark packaged in that file comes from

> **RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music**
> Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen
> <https://arxiv.org/abs/2306.15412> — code at <https://github.com/Dream-High/RMVPE>

which is **Apache-2.0**, and `to_local_average_cents()` originates one step further back
in

> **CREPE** — <https://github.com/marl/crepe>
> The MIT License (MIT), Copyright (c) 2018 Jong Wook Kim

Because the RMVPE code is Apache-2.0 upstream, `preprocess/algorithms/rmvpe.py` is **not**
covered by this repository's MIT license. We distribute it under the Apache License,
Version 2.0 ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)), whose conditions are a
superset of the MIT conditions; keeping the Apache notice together with the two MIT
copyright notices satisfies all three upstreams. CREPE's text is in
[`LICENSES/crepe-MIT.txt`](LICENSES/crepe-MIT.txt).

## 3. RMVPE weights (`rmvpe.pt`)

**Not redistributed here.** `get_model_path()` downloads the file on first use from

    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

which is a third-party mirror (`lj1995` is the RVC author). The weights originate with
the RMVPE authors. We do not restate terms for them, because the mirror states none — if
you redistribute the weights yourself, check with the RMVPE authors first.

## 4. Neural vocoder ONNX

`checkpoints/nhv_v3_2.onnx` and `checkpoints/nhv_v3_2x.onnx` are built artifacts of the
[NHVSing](https://github.com/wavtechyukky/NHVSing/) project (also ours, MIT) and are
governed by that project's license and by the terms of the data it was trained on.

## 5. Demo and sample audio

`demo/audio/*_gt.ogg` and `notebooks/sample_data/ritsu_flashblack_0000.wav` are excerpts
of real recordings from the singing databases below, not synthesis output. They are
included so the demo page can compare against ground truth. They remain governed by each
database's own terms of use, **not** by MIT.

- Oniku Kurumi singing database — <https://onikuru.info/db-download/>
- Natsume Yuuri (database production: アマノケイ / voice provider: 霧野蒼太) —
  <https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs>
- Namine Ritsu — <https://www.canon-voice.com/voicebanks/>

The same applies to the trained acoustic-model checkpoints distributed via GitHub
Releases; see the `CREDITS.txt` bundled with each release.

---

## Not third-party: our own implementations

For the avoidance of doubt, the following are **our own code**, written from the
published descriptions, and were checked against the upstreams named below by comparing
syntax trees rather than by eye. They are MIT like the rest of the repository. We name
the work that the design follows because we think credit is due, not because any of
their code is present here.

| Our file | Design follows |
|---|---|
| `leapsinger/modules/discriminators/jcu.py` | the JCU discriminator of [DiffGAN-TTS](https://github.com/keonlee9420/DiffGAN-TTS) |
| `leapsinger/modules/flow/*`, `leapsinger/modules/backbones/dilated_conv.py` | rectified flow, and the DiffWave-style dilated-convolution backbone |
