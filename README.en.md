# LeapSinger

**日本語**: [README.md](README.md)

LeapSinger is a diffusion-style **acoustic model** for singing voice. It runs very fast — even on a CPU — and produces stable, clean periodic (harmonic) content. It takes phonemes, their durations, and pitch (F0), and generates a mel-spectrogram. Training needs only audio plus phoneme labels with timing; it does not depend on any particular language. (All of our examples use Japanese data, though.)

![LeapSinger overview](doc/fig/leapsinger_overview.png)

**▶ Listen to the demo: https://wavtechyukky.github.io/LeapSinger/demo/**

## What is LeapSinger?

Most diffusion models start from random noise and paint the mel little by little over many steps. LeapSinger instead starts from a **"pseudo-mel" built from F0** (an impulse waveform plus white noise), and finishes the mel in **a single step** with a rectified flow. It skips the slow work of learning how to draw clean periodic content, and jumps straight to a high-quality mel — hence the name *LeapSinger*.

The pseudo-mel differs between the v/uv model and the non-v/uv model. The v/uv model can deliberately produce unvoiced regions.

![Pseudo-mel with and without v/uv](doc/fig/pseudo_mel_vuv.png)

The non-v/uv model (top) lays harmonics across every frame. The v/uv model (bottom) gates the harmonics off in unvoiced frames (the dark vertical bands are the unvoiced regions).

Through a lot of experiments we found two things: a high-quality neural vocoder for singing is sensitive to the *texture* of the mel, and the hard part for an acoustic model is drawing clean periodic content. LeapSinger starts not from noise but from a pseudo-mel that already has the shape of the pitch, so it avoids the hardest part — learning to draw the periodic content. This design gives:

- **High-quality periodic content** — it draws clean, low-noise periodic content stably. This helps both the texture of the vocoder output and how well each speaker's voice is reproduced.
- **Speed** — there is only one reverse step, so even on a single CPU core the RTF is under 0.03. You *can* use more steps, but going beyond one step actually moves the result *away* from the ground truth.

Other features:

- **Multi-speaker** — switch voices by speaker ID. As an example, we distribute a model with three Japanese singers (Oniku Kurumi / Natsume Yuuri / Namine Ritsu).
- **Style control** — switch singing styles within the same speaker.

## Demo

**https://wavtechyukky.github.io/LeapSinger/demo/**

- A three-speaker model, comparing the generated results side by side with GT (the real recordings). Training on a single speaker slightly improves the speaker fidelity of the mel, but we saw no large difference in synthesis quality.
- A style-control demo.

## Performance

RTF (Real-Time Factor) measured on CPU. Smaller is faster; below 1 means faster than real time.

RTF per acoustic model, comparing Python native vs. ONNX across core counts:

| Cores | Python native | ONNX |
|:--:|--:|--:|
| 1 | 0.023 | 0.066 |
| 2 | 0.021 | 0.047 |
| 4 | 0.020 | 0.042 |
| 8 | 0.021 | 0.060 |
| 10 | 0.021 | 0.062 |

- **Python native** is a single step, so it barely depends on core count. Even on one core the RTF is 0.023 (about 43× faster than real time).
- **ONNX** is a few times slower than native because of onnxruntime overhead, but it is still more than 15× faster than real time. 4 cores are fastest; 8 or more is actually slower, because the excitation sums its harmonics one at a time — which is what keeps memory independent of the input length — and that sum does not parallelise across cores.
- **The NHVSing vocoder** runs at RTF under 0.1 on CPU (see the NHVSing repository for details).

(Measured on Apple Silicon, 10 cores, onnxruntime CPU, a real 7.0-second phrase, median of 9 runs. The ONNX build is `export/cli.py` with its defaults, `--variant diffsinger --hop 512`. Results vary by machine.)

The frame settings are 44.1 kHz and hop size 256. Hop size 512 is handled by averaging each pair of adjacent frames.

## Architecture

The flow in the figure above is:

1. **Input** — phonemes, durations, F0 (plus speaker ID if needed).
2. **Encoder** — embed the phonemes, stretch them to frame length according to the durations, and add F0 and speaker to form the condition.
3. **Harmonic + Noise Excitation** — build a "pseudo-mel" from F0 (an impulse waveform plus white noise). This is the starting point of the flow.
4. **Rectified Flow (1 step)** — starting from the pseudo-mel and conditioned on the condition, transform it into a realistic mel in a single step.
5. **NHVSing** — a high-quality neural vocoder that turns the mel into audio (a waveform), at RTF under 0.1 on CPU. https://github.com/wavtechyukky/NHVSing/

The pseudo-mel lets you tune the harmonic decay, the number of harmonics, and the strength of the white noise. That said, after much testing, stacking the impulse's harmonics all the way up to the Nyquist frequency gives the best quality.

## Usage

### Setup

Python 3.10 or later. Clone the repository and do an editable install.

    git clone https://github.com/wavtechyukky/LeapSinger
    cd LeapSinger
    pip install -e .                  # inference, re-synthesis, notebooks
    pip install -e ".[train]"         # training (adds TensorBoard)
    pip install -e ".[export]"        # ONNX export
    pip install -e ".[train,export]"  # everything

- **PyTorch** is best installed to match your environment (CPU or CUDA build). To train on GPU, install the CUDA build following the [PyTorch site](https://pytorch.org/).
- **F0 extraction (RMVPE)** weights download automatically on first run (HuggingFace → `preprocess/algorithms/rmvpe.pt`).
- **The vocoder (NHVSing)** is bundled as ONNX under `checkpoints/`; no extra download is needed.
- **The acoustic model itself is distributed via Releases** (not included in the repo).

### Configuration

Training and export are driven by YAML (there are examples in `configs/`). A YAML file has six sections: `mel` / `model` / `excitation` / `train` / `gan` / `data`. The main keys are:

- `model` — `spk_dim` (greater than 0 means multi-speaker), `n_speakers`, `n_styles` (greater than 0 enables styles), `use_uv` (whether to feed v/uv as a condition), `pos_cond` (the relative position inside each phoneme, added to the condition: `phoneme` switches it, on by default; `encoding: sinusoidal` is DiffSinger's stretch embedding, round(1000·i/n) → sinusoidal embedding → MLP; `raised_cosine` is the old code of the 2026-10-16 A/B, kept only to load old checkpoints, and `unit`, the position inside each nucleus unit, works only with it and is not used; a resumed run keeps its checkpoint's setting), `dur_cond` (log(1 + phoneme frames) embedding; off by default). `--set model.pos_cond.phoneme=false` overrides the config for A/B runs without editing it.
- `excitation` — `n_harm` (number of harmonics), `harm_decay` (harmonic decay, 1.5 by default), `noise_ratio` (white-noise strength), `x0_sigma` (standard deviation of Gaussian noise added to the flow's starting point; above 0 the start becomes stochastic and the model becomes a multi-step reflow teacher; 0 by default).
- `train` — `lr`, `max_updates`, `num_steps` (inference steps; **1** recommended), `balance_speakers` (sample speakers equally), `min_batch_items` (above 0: pack longest-first and split the final leftover with the previous batch, so the few longest phrases never form a tiny batch of their own), and so on. Training logs the pre-clip gradient norm (max and median per interval), the smallest batch and the largest loss per interval to TensorBoard, and writes every step whose gradient norm exceeds 5× the recent median, with the batch's contents, to `<run>/spikes.log`. `amp` (`fp32` | `bf16`; `bf16` runs the encoder and the velocity network under autocast, faster on an A100). `lr_decay` / `lr_decay_steps` (below 1, the learning rate and the discriminator's learning rate fall smoothly every step, by a factor of lr_decay every lr_decay_steps; 0.8 / 5000 is DiffSinger's pace; the default 1.0 keeps it constant). Checkpoints also record the training settings (batch, learning rate, precision and so on), and a resumed run reports any setting that differs. Each step's time is broken down as well: `time/data_wait_ms` (waiting for data), `time/h2d_ms` (copy to the GPU), `time/step_ms` (compute), `time/steps_per_sec`, `time/eval_s` and `time/save_s`, and the GPU time inside the compute, `time/gpu_{aug,enc,exc,net,gan,bwd,opt}_ms` (`exc` is the pseudo-mel, `net` the velocity network and the losses).
- `gan` — settings for the GAN that sharpens texture. `enabled` (`false` = flow loss + mel loss only), `gan_start_step` (the step at which the GAN turns on), `gan_strength` (strength of the adversarial loss), and so on.
- `data` — `spk_map` / `style_map` (mapping from dataset folder name → speaker ID / style ID), `max_sec` (drop phrases longer than this; the dropped names are printed at start-up), `loudness_agc` (slow automatic gain on the target mel: loudness changes slower than `sigma_sec` — phrase-to-phrase level, section-level crescendi — are removed, note-level and phonetic dynamics are kept, so the model sings at a constant level whatever the score; `null` turns it off).

### Building a dictionary

The Japanese phoneme list is in `dict/ja.phonemes` (one phoneme per line, the order is the ID, `pau` at the top is ID 0, and anything after `#` is a comment). To use a different language or your own phonemes, make a file in the same format and pass it to each command with `--phonemes <file>` (if omitted, the Japanese `dict/ja.phonemes` is used).

Phoneme classes that the list cannot express go in a language pack, `dict/<language>.yaml` (for example `dict/ja.yaml` and `dict/en.yaml`). It points to the phoneme list with `phonemes` and defines `vowels` and `silence`; every other phoneme is a consonant. For the evaluation tool it can also list `devoiced` (devoiced vowels), `voiceless` and `voiced` consonants, and `steady_vowels` (vowels whose long middles are checked for wobble); these are optional. Training and evaluation look in `dict/` for the language pack whose phoneme list matches the model's or the data's. The consonant loss weight (`train.cons_weight`) needs a language pack.

### Preprocessing

Prepare one YAML per dataset (`configs/recipes/<db>.yaml`). Each song uses an audio `wav` and a `.lab` with phoneme timing (plus a score). The following command produces preprocessed data under `data/<db>/`.

    python -m preprocess.run --recipe configs/recipes/<db>.yaml

The three databases used in the examples can be downloaded with these scripts (please follow each database's terms of use).

    python preprocess/download_scripts/download_oniku.py
    python preprocess/download_scripts/download_natsume.py
    python preprocess/download_scripts/download_ritsu.py

F0 is extracted with RMVPE. (Please do not run RMVPE across multiple processes.)

- A recipe can name its phoneme list with `phonemes: dict/<list>.phonemes` (`--phonemes` takes precedence).
- `--check` only checks the labels, without processing audio. It reports symbols missing from the phoneme list (after `phon_norm` and the silence map), missing wav or label files, and labels that run past the end of the audio. It writes nothing.
- `layout: from_lab` reads only the songs that have a label file (the `lab` template may be an absolute path).
- Labels that only hold phoneme durations (Opencpop's `transcriptions.txt`, DiffSinger's `transcriptions.csv`) are converted to per-segment labs with `tools/convert_labels.py` first.

### Training

Training has two stages. The first stage learns the base voice with a flow loss plus a mel loss; the second stage turns on a GAN to sharpen texture. (This is set in the config's `gan` section; `gan.enabled: false` gives the flow loss plus mel loss only.)

    python -m train --config configs/<name>.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

Running the same command again automatically resumes from where it stopped.

### Export

    python -m export.cli \
      --ckpt log/<run>/ckpt_050000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 256 --speaker bake --spk-id 0

For speaker handling (bake / embed / none) and other details, see "Export to ONNX" below.

You can try the whole flow — from export to use — in a notebook. Download the model from the Release and place it in `notebooks/sample_data/` (see `place_model_here.txt` in that folder).

    notebooks/export_and_use_onnx.ipynb

The notebook exports the acoustic model to a single ONNX and runs it end to end (phonemes + duration + F0 → mel → audio).

### Export to ONNX

`export/` converts a checkpoint into a self-contained ONNX graph. The excitation and the single-step flow are baked into the graph, so the caller only needs to pass phonemes, durations, and F0. For speaker handling, you can choose:

- **bake** — fix a single voice. This makes the simplest graph (no speaker input).
- **embed** — take a speaker vector as input. One graph can switch to any voice.
- **none** — for single-speaker models. No speaker input and no baking (a graph with no notion of speaker). For multi-speaker models, use bake or embed.

A style-bearing model (`n_styles` > 0) is exported with `--variant full`, and the style to use is baked into the graph with `--style-id` (default 0) — one ONNX per style.

As the last post-export pass, the excitation's harmonic sum (a straight-line chain of 256 blocks by default) is folded into a single ONNX `Loop` (`export/harmonic_loop.py`; `--no-harmonic-loop` keeps the unrolled chain). Left unrolled, onnxruntime's default execution order keeps 256 sample-rate arrays alive at once and peak memory reaches 0.28 MB per frame (about 1.5 GB for a 30-second phrase). With the Loop, the 30-second peak drops from +1469 MB to +287 MB, inference is 10–30% faster and session creation is 10× faster. The formula and the order of operations are unchanged, so the output is bit-identical (`--verify` compares against the unrolled form). `Loop` has existed since opset 1 and runs on every runtime, including OpenUTAU's onnxruntime. It can also be applied to an already exported ONNX: `python -m export.harmonic_loop <in.onnx> <out.onnx> --check`

### Vocoder

Two NHVSing vocoders are bundled under `checkpoints/`.

- `nhv_v3_2.onnx` — takes a hop-size-256 mel and F0.
- `nhv_v3_2x.onnx` — takes a hop-size-512 mel and F0.

The bundled version is **V3.2** — the latest weights. It fixes an abrupt per-frame weakening of the waveform at high pitch by switching the LTV filter's overlap-add to a Hann window. The input/output contract is identical to V3.1, so it is a drop-in replacement (see [NHVSing](https://github.com/wavtechyukky/NHVSing/) for details).

## Options

- **Dictionary** — you can specify any phoneme dictionary. The design can handle languages other than Japanese (multilingual support itself is future work).
- **v/uv handling** — choose between a mode that feeds voiced/unvoiced (v/uv) as a condition, and a mode that uses only a continuous F0 with the gaps filled by linear interpolation.
- **Excitation tuning** — change the harmonic decay, the number of harmonics, and the white-noise strength.
- **Training recipes** — training conditions are set in YAML (which datasets to use, speaker IDs, per-speaker styles and data, and so on).

## Future work

- **Multilingual support** — so far we validate with Japanese data, but the design itself is language-independent. We plan to support other languages with their own phoneme dictionaries and data, aiming for a single model that can handle multiple languages.
- **Higher quality** — we think there is still room to improve speaker fidelity, for example by refining how the pseudo-mel is generated and tuning its parameters.

## License

The code is MIT (`LICENSE`). The following are **not** covered by MIT and follow their own licenses and terms of use; [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) has the details.

- **`preprocess/algorithms/base.py` and `preprocess/algorithms/rmvpe.py`** — both taken verbatim from [pitch-benchmark](https://github.com/lars76/pitch-benchmark) (MIT, Copyright (c) 2025 Lars Nieradzik). Their notice is in `LICENSES/pitch-benchmark-MIT.txt` and must be kept when redistributing. `base.py` is MIT like the rest of this repository.
  `rmvpe.py` is not: the RMVPE model inside it is [Apache-2.0 upstream](https://github.com/Dream-High/RMVPE), so that one file is distributed under Apache-2.0 (full text in `LICENSES/Apache-2.0.txt`). Its `to_local_average_cents()` originates in [CREPE](https://github.com/marl/crepe); that MIT notice is in `LICENSES/crepe-MIT.txt`. The `rmvpe.pt` weights are downloaded at run time and are not distributed here.
- **Demo and sample audio** (`demo/audio/*_gt.ogg`, `notebooks/sample_data/*.wav`) — excerpts of real recordings from the singing databases, not synthesis output. Each database's terms of use apply.
- **The bundled vocoder ONNX files** (`checkpoints/nhv_v3_2*.onnx`) — artifacts of [NHVSing](https://github.com/wavtechyukky/NHVSing/).
- **The trained models distributed via Releases** and the singing databases used to train them — see `CREDITS.txt` in the model release.

## Acknowledgments

Thanks to the datasets used to train this model, and to the related projects.

- Oniku Kurumi singing database (Oniku Kurumi) — https://onikuru.info/db-download/
- Natsume Yuuri (database production: アマノケイ / voice provider: 霧野蒼太) — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- Namine Ritsu — https://www.canon-voice.com/voicebanks/
- Neural Homomorphic Vocoder — https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html
- dsp (zjlww) — https://github.com/zjlww/dsp
- pitch-benchmark (Lars Nieradzik; the two files in `preprocess/algorithms/` come from it) — https://github.com/lars76/pitch-benchmark
- RMVPE (the F0 model itself) — https://github.com/Dream-High/RMVPE
- CREPE (`to_local_average_cents()`, reached via RMVPE) — https://github.com/marl/crepe
- DiffGAN-TTS (the JCU discriminator design; our code is our own) — https://github.com/keonlee9420/DiffGAN-TTS

The distributed multi-speaker models display the credits above, following each database's terms. For Natsume Yuuri, we display **database production: アマノケイ / voice provider: 霧野蒼太**, and we bundle the "Terms of use for Natsume Yuuri's output audio" with the model distribution.
