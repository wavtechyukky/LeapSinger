# Third-party notices

LeapSinger is released under the MIT license (see [`LICENSE`](LICENSE)). A few things
shipped in, or fetched by, this repository are **not** ours and carry different terms.
They are listed here. If you redistribute LeapSinger, carry these notices with it.

| What | Where | Terms |
|---|---|---|
| RMVPE pitch estimator | `preprocess/algorithms/rmvpe.py` | Apache-2.0 — [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| `to_local_average_cents()` inside that file | same file | MIT (CREPE) — [`LICENSES/crepe-MIT.txt`](LICENSES/crepe-MIT.txt) |
| RMVPE weights `rmvpe.pt` | downloaded at run time, **not** in this repo | see below |
| Neural vocoder | `checkpoints/nhv_v3_2.onnx`, `checkpoints/nhv_v3_2x.onnx` | NHVSing project |
| Demo and sample audio | `demo/audio/*_gt.ogg`, `notebooks/sample_data/*.wav` | each singing database's terms of use |
| Acoustic-model checkpoints | GitHub Releases, **not** in this repo | each singing database's terms of use |

Everything else in this repository is our own work and is MIT.

---

## 1. RMVPE — `preprocess/algorithms/rmvpe.py` (Apache-2.0)

This one file is **not** MIT. It is licensed under the Apache License, Version 2.0,
whose full text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

The model comes from RMVPE:

> **RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music**
> Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen
> <https://arxiv.org/abs/2306.15412> — code at <https://github.com/Dream-High/RMVPE>

`ConvBlockRes`, `ResEncoderBlock`, `ResDecoderBlock`, `Encoder`, `Intermediate`,
`Decoder`, `DeepUnet0`, `BiGRU`, `E2E0` and `to_local_average_cents` are from there.
Upstream spreads them over `src/spec.py`, `src/deepunet.py`, `src/seq.py`,
`src/model.py` and `src/utils.py`; we flattened them into one module.

Two pieces do not come from that repository. They match the RMVPE integrations the
singing- and voice-conversion community maintains:

- **`MelSpectrogram`** — its `keyshift` / `speed` arguments do not exist upstream.
  This version is identical to the one in [yxlllc/DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
  (MIT) and in [openvpi/DiffSinger](https://github.com/openvpi/DiffSinger) and
  [openvpi/SOME](https://github.com/openvpi/SOME) (Apache-2.0).
- **the pad-to-a-multiple-of-32-frames step in `E2E0.forward`** — this matches
  [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT),
  where it sits in `RMVPE.mel2hidden` rather than in the module.

All of those are Apache-2.0 or MIT. Nothing here comes from the AGPL-3.0 variant in
so-vits-svc: that fork rewrote `to_local_average_cents`, and the copy in this file is
Dream-High's, not theirs.

### Modifications (Apache-2.0 section 4(b))

Modified in 2026 by wavtechyukky:

- flattened the upstream modules into one file, and dropped what inference does not
  need (`STFT`, `TimbreFilter`, `DeepUnet`, `E2E`, `BiLSTM`, training and evaluation code)
- `ResDecoderBlock.__init__` picks `output_padding` for strides `(2, 2)` and `(2, 1)`
  as well, not only `(1, 2)`
- added `DEFAULT_MODEL_URL` and `get_model_path()`, which fetch the weights on first use
- added `RMVPEPitchAlgorithm`, which adapts the model to this repository's
  `ContinuousPitchAlgorithm` interface

## 2. CREPE — `to_local_average_cents()` (MIT)

RMVPE took that function from CREPE almost verbatim, so its MIT notice applies too:

> The MIT License (MIT) — Copyright (c) 2018 Jong Wook Kim
> <https://github.com/marl/crepe>

Full text: [`LICENSES/crepe-MIT.txt`](LICENSES/crepe-MIT.txt).

## 3. RMVPE weights (`rmvpe.pt`)

**Not redistributed here.** `get_model_path()` downloads the file on first use from

    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

which is a third-party mirror (`lj1995` is the RVC author). The weights originate with
the RMVPE authors. We do not restate terms for them, because the mirror states none —
if you redistribute the weights yourself, check with the RMVPE authors first.

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
published descriptions. They are MIT like the rest of the repository. We name the work
that the design follows because we think credit is due, not because any of their code is
present here.

| Our file | Design follows |
|---|---|
| `leapsinger/modules/discriminators/jcu.py` | the JCU discriminator of [DiffGAN-TTS](https://github.com/keonlee9420/DiffGAN-TTS) |
| `leapsinger/modules/flow/*`, `leapsinger/modules/backbones/dilated_conv.py` | rectified flow, and the DiffWave-style dilated-convolution backbone |
