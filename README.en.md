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
| 1 | 0.027 | 0.090 |
| 2 | 0.026 | 0.063 |
| 4 | 0.027 | 0.058 |
| 8 | 0.026 | 0.054 |
| 10 | 0.024 | 0.065 |

- **Python native** is a single step, so it barely depends on core count. Even on one core the RTF is 0.027 (about 37× faster than real time).
- **ONNX** is a few times slower than native because of onnxruntime overhead, but it is still more than 10× faster than real time. 4–8 cores are fastest; using all cores (10) is actually slower.
- **The NHVSing vocoder** runs at RTF under 0.1 on CPU (see the NHVSing repository for details).

(Measured on Apple Silicon, 10 cores, onnxruntime CPU, a ~7-second phrase, median. Results vary by machine.)

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

- `model` — `spk_dim` (greater than 0 means multi-speaker), `n_speakers`, `n_styles` (greater than 0 enables styles), `use_uv` (whether to feed v/uv as a condition).
- `excitation` — `n_harm` (number of harmonics), `harm_decay` (harmonic decay), `noise_ratio` (white-noise strength).
- `train` — `lr`, `max_updates`, `num_steps` (inference steps; **1** recommended), `balance_speakers` (sample speakers equally), and so on.
- `gan` — settings for the GAN that sharpens texture. `enabled` (`false` = flow loss + mel loss only), `gan_start_step` (the step at which the GAN turns on), `gan_strength` (strength of the adversarial loss), and so on.
- `data` — `spk_map` / `style_map` (mapping from dataset folder name → speaker ID / style ID).

### Building a dictionary

The Japanese phoneme list is in `dict/ja.phonemes` (one phoneme per line, the order is the ID, `pau` at the top is ID 0, and anything after `#` is a comment). To use a different language or your own phonemes, make a file in the same format and pass it to each command with `--phonemes <file>` (if omitted, the Japanese `dict/ja.phonemes` is used).

### Preprocessing

Prepare one YAML per dataset (`configs/recipes/<db>.yaml`). Each song uses an audio `wav` and a `.lab` with phoneme timing (plus a score). The following command produces preprocessed data under `data/<db>/`.

    python -m preprocess.run --recipe configs/recipes/<db>.yaml

The three databases used in the examples can be downloaded with these scripts (please follow each database's terms of use).

    python preprocess/download_scripts/download_oniku.py
    python preprocess/download_scripts/download_natsume.py
    python preprocess/download_scripts/download_ritsu.py

F0 is extracted with RMVPE. (Please do not run RMVPE across multiple processes.)

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

- **`preprocess/algorithms/rmvpe.py`** — a vendored and modified copy of [RMVPE](https://github.com/Dream-High/RMVPE). That one file is **Apache-2.0** (full text in `LICENSES/Apache-2.0.txt`). The `to_local_average_cents()` function inside it originates in [CREPE](https://github.com/marl/crepe); its MIT notice is in `LICENSES/crepe-MIT.txt`. The `rmvpe.pt` weights are downloaded at run time and are not distributed here.
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
- RMVPE (F0 extraction; vendored and modified here) — https://github.com/Dream-High/RMVPE
- CREPE (`to_local_average_cents()`, reached via RMVPE) — https://github.com/marl/crepe
- DiffGAN-TTS (the JCU discriminator design; our code is our own) — https://github.com/keonlee9420/DiffGAN-TTS

The distributed multi-speaker models display the credits above, following each database's terms. For Natsume Yuuri, we display **database production: アマノケイ / voice provider: 霧野蒼太**, and we bundle the "Terms of use for Natsume Yuuri's output audio" with the model distribution.
