# LeapSinger

**日本語**: [README.md](README.md)

LeapSinger is a diffusion-style **acoustic model** for singing voice. It runs very fast — even on a CPU — and produces stable, clean periodic (harmonic) content. It takes phonemes, their durations, and pitch (F0), and generates a mel-spectrogram. Training needs only audio plus phoneme labels with timing; it does not depend on any particular language. (The released models are trained on Japanese data.)

![LeapSinger overview](doc/fig/leapsinger_overview.png)

**▶ Listen to the demo: https://wavtechyukky.github.io/LeapSinger/demo/**

### What changed in V2 (2.0.0)

- **No GAN.** A teacher solves the flow in 16 steps; a 1-step student is then distilled from the teacher's start/output pairs (a second rectified flow, "reflow"). Inference is still one step, the harmonics are cleaner than the GAN model's, and the spectra of voiceless consonants are closer to the recordings. The released model (`3singer_v2_student.pt`) is this student.
- **A "loose" loudness correction of the training mel** (`data.loudness_agc`). It removes phrase-to-phrase level differences and slow crescendi from the training target while keeping the singing expression as far as possible, so the model sings at a constant level whatever the score. The values are chosen with the bundled [loudness-correction tuner](#loudness-correction-tuner).
- **Relative position inside each phoneme** as a condition (`model.pos_cond`), as used in DiffSinger. It gave no significant quality gain on the Japanese singers, but there was no reason to drop it either, so it stays.
- **Language packs** (`dict/<language>.yaml`) and `tools/convert_labels.py` for other label formats. Dictionaries, recipes and configs for English, Korean and Mandarin are included (models: Japanese only).
- **ONNX export** folds the excitation's harmonic sum into one `Loop`, cutting peak memory on long phrases by about 5×. `--speaker embed` also writes the speaker-vector table (`<name>.speakers.json`).
- **Evaluation tools** (`tools/eval_ab.py`, `tools/render_ab.py`) and `tests/`.
- v0.1.0 checkpoints and preprocessed data still work ([Compatibility](#compatibility)).

#### V1 vs V2 on the 71 held-out phrases

An excerpt of `tools/eval_ab.py` on the v0.1.0 model (V1) and the V2 student, over 71 phrases from songs never seen in training (synthesised in one step from the true phoneme durations and F0). Ratios are generated / recording, so 1 means "like the recording"; per-band values are 68 Hz–945 Hz / 973 Hz–2.4 kHz / 2.5–6.1 kHz / 6.3–15.5 kHz. Bold = closer to the recording (ratios closer to 1, offsets closer to 0, error lower).

| Metric | V1 | V2 |
|---|---|---|
| Fine time texture (per band) | 0.86 / 0.91 / 0.94 / 0.92 | **0.92 / 0.95 / 0.99 / 1.01** |
| Spectral-shape peaks and valleys, consonants (per band) | 0.95 / 0.86 / 0.73 / 0.72 | **1.04 / 0.96 / 0.88 / 0.93** |
| Spectral-shape peaks and valleys, vowels (per band) | **0.99** / 0.97 / 0.86 / 0.78 | 1.02 / **1.01** / **0.90** / **0.87** |
| Motion at the phoneme rate, 30–150 ms, vowels (per band) | 0.93 / 0.95 / 0.92 / 0.87 | **0.98 / 0.99 / 0.99 / 0.98** |
| Harmonic peak roughness, < 1 kHz / 1–3 kHz (above 1 = jagged) | 1.57 / 1.34 | **1.20 / 1.12** |
| Harmonic valley level vs recording (+ = valleys filled in) | +0.7 / +0.7 dB | **−0.6 / −0.5 dB** |
| Regular (GAN-like) pattern, largest excess (per band) | 2.8 / 2.4 / 2.1 / 6.3 dB | **0.7 / 1.4 / 1.7 / 2.3 dB** |
| Steadiness inside long vowels | 0.91 | **1.03** |
| Mel error against each model's own training target (non-silent frames) | **5.08 dB** | 5.32 dB |
| The same after smoothing both by 21 frames × 21 bins (the coarse shape only) | 2.44 dB | **2.32 dB** |
| Phrase-level level offset | 0.95 dB | **0.64 dB** |

V2 keeps the detail that averaging tends to lose — fine time texture, the spectral shape of consonants, phoneme-rate motion — its harmonic lines are cleaner, and the regular pattern the GAN left is gone. It is not better everywhere: the plain mel error is 0.24 dB higher (the excess is random detail that does not coincide with the recording — smoothing both sides reverses the order, and the phrase-level level is closer for V2); the first and last 20 ms of phonemes are 0.13 dB worse; and the (rare) devoiced vowels are clearly worse (error 7.8 → 11.2 dB, spectral centroid 1.3 kHz low). One checkpoint each; V1's target is the raw mel and V2's the loudness-corrected mel (the ratio metrics do not depend on the target's level).

## What is LeapSinger?

Most diffusion models start from random noise and paint the mel little by little over many steps. LeapSinger instead starts from a **"pseudo-mel" built from F0** (an impulse waveform plus white noise), and finishes the mel in **a single step** with a rectified flow. It skips the slow work of learning how to draw clean periodic content, and jumps straight to a high-quality mel — hence the name *LeapSinger*.

The pseudo-mel differs between the v/uv model and the non-v/uv model. The v/uv model can deliberately produce unvoiced regions.

![Pseudo-mel with and without v/uv](doc/fig/pseudo_mel_vuv.png)

The non-v/uv model (top) lays harmonics across every frame. The v/uv model (bottom) gates the harmonics off in unvoiced frames (the dark vertical bands are the unvoiced regions).

Through a lot of experiments we found two things: a high-quality neural vocoder for singing is sensitive to the *texture* of the mel, and the hard part for an acoustic model is drawing clean periodic content. LeapSinger starts not from noise but from a pseudo-mel that already has the shape of the pitch, so it avoids the hardest part — learning to draw the periodic content. This design gives:

- **High-quality periodic content** — it draws clean, low-noise periodic content stably. This helps both the texture of the vocoder output and how well each speaker's voice is reproduced.
- **Speed** — there is only one reverse step; on 4 CPU cores the RTF is about 0.02 (about 50× faster than real time). In V2 the 16-step teacher exists only during training; its knowledge is moved into a 1-step student (the student is not significantly different from the teacher, and inference stays one step).

Other features:

- **Multi-speaker** — switch voices by speaker ID. As an example, we distribute a model with three Japanese singers (Oniku Kurumi / Natsume Yuuri / Namine Ritsu).
- **Style control** — switch singing styles within the same speaker (`n_styles` > 0; the v0.1.0 Namine Ritsu three-style model is the example. The V2 release model has no styles).

## Demo

**https://wavtechyukky.github.io/LeapSinger/demo/**

- A three-speaker model, comparing the generated results side by side with GT (the real recordings), for **V1** (v0.1.0, GAN) and **V2** (2.0.0, the reflow student). Training on a single speaker slightly improves the speaker fidelity of the mel, but we saw no large difference in synthesis quality.
- Songs sung from scores the model never saw, V1 next to V2.
- A style-control demo (the v0.1.0 style-bearing model).

## Performance

RTF (Real-Time Factor) measured on CPU. Smaller is faster; below 1 means faster than real time.

RTF of the V2 release model (`3singer_v2_student.pt`), Python native vs. ONNX across core counts:

| Cores | Python native | ONNX (hop 512) | ONNX (hop 256) |
|:--:|--:|--:|--:|
| 1 | 0.057 | 0.050 | 0.051 |
| 2 | 0.033 | 0.033 | 0.033 |
| 4 | 0.020 | 0.025 | 0.026 |
| 8 | 0.018 | 0.036 | 0.035 |
| 10 | 0.017 | 0.036 | 0.035 |

- **Python native** reaches RTF 0.020 on 4 cores (about 50× faster than real time) and 0.017 on 10. The one-step velocity network is light; most of the time is the excitation's harmonic sum (a `torch.compile` fused kernel), which parallelises across cores.
- **ONNX** is fastest on 4 cores, RTF 0.025 (40× faster than real time); 8 or more is actually slower, because the harmonic sum is folded into one sequential `Loop` — which is what keeps memory independent of the input length — and that loop does not parallelise across cores. Hop 512 and 256 make no difference.
- **The NHVSing vocoder** runs at RTF under 0.1 on CPU (see the NHVSing repository for details).

(Measured on an Apple M4 (10 cores = 4P + 6E), 16 GB, PyTorch 2.11, onnxruntime 1.23, a real 7.0-second phrase, median of 9 runs. The ONNX builds are `export/cli.py` with `--variant diffsinger --speaker embed` and `--hop 512` / `--hop 256`. Native is `infer.infer_mel`, excluding the first-call compile. The v0.1.0 model measured under the same conditions gives the same numbers (0.021 on 4 cores), so V2 is not slower. Results vary by machine.)

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
    pip install -e ".[tuner]"         # the loudness-correction tuner (fastapi, uvicorn)
    pip install -e ".[dev]"           # tests/ (pytest, httpx)
    pip install -e ".[train,export,tuner,dev]"  # everything

- **PyTorch** is best installed to match your environment (CPU or CUDA build). To train on GPU, install the CUDA build following the [PyTorch site](https://pytorch.org/).
- **F0 extraction (RMVPE)** weights download automatically on first run (HuggingFace → `preprocess/algorithms/rmvpe.pt`).
- **The vocoder (NHVSing)** is bundled as ONNX under `checkpoints/`; no extra download is needed.
- **The acoustic model itself is distributed via Releases** (not included in the repo).

### Configuration

Training and export are driven by YAML (there are examples in `configs/`). A YAML file has six sections: `mel` / `model` / `excitation` / `train` / `gan` / `data`. The main keys are:

- `model` — `spk_dim` (greater than 0 means multi-speaker), `n_speakers`, `n_styles` (greater than 0 enables styles), `use_uv` (whether to feed v/uv as a condition), `pos_cond` (the relative position inside each phoneme, added to the condition: `phoneme` switches it, on by default; `encoding: sinusoidal` is DiffSinger's stretch embedding, round(1000·i/n) → sinusoidal embedding → MLP; `raised_cosine` is the old code of the 2026-10-16 A/B, kept only to load old checkpoints, and `unit`, the position inside each nucleus unit, works only with it and is not used; a resumed run keeps its checkpoint's setting), `dur_cond` (log(1 + phoneme frames) embedding; off by default). `--set model.pos_cond.phoneme=false` overrides the config for A/B runs without editing it.
- `excitation` — `n_harm` (number of harmonics), `harm_decay` (harmonic decay, 1.5 by default), `noise_ratio` (white-noise strength), `x0_sigma` (standard deviation of Gaussian noise added to the flow's starting point; above 0 the start becomes stochastic and the model becomes a multi-step reflow teacher; 0 by default).
- `train` — `lr`, `max_updates`, `num_steps` (inference steps: **1** for a 1-step model or a student, 16 for a reflow teacher), `reflow_teacher` (path of the teacher checkpoint; setting it makes the run a reflow student. `reflow_teacher_steps` (steps the teacher uses to solve the pairs, 8 by default), `reflow_draws` (pairs per phrase, 4), `reflow_algorithm` (euler | rk2 | rk4), `reflow_t` (zero = t = 0 only, a 1-step model; uniform = any t), `reflow_seed`. The student starts from the teacher's weights and refuses a config that differs from the teacher's checkpoint), `cons_weight` / `cons_weight_next_ms` (loss weight on consonant frames and the head of the phoneme after each; 1.0 = off), `balance_speakers` (sample speakers equally), `min_batch_items` (above 0: pack longest-first and split the final leftover with the previous batch, so the few longest phrases never form a tiny batch of their own), and so on. Training logs the pre-clip gradient norm (max and median per interval), the smallest batch and the largest loss per interval to TensorBoard, and writes every step whose gradient norm exceeds 5× the recent median, with the batch's contents, to `<run>/spikes.log`. `amp` (`fp32` | `bf16`; `bf16` runs the encoder and the velocity network under autocast, faster on an A100). `lr_decay` / `lr_decay_steps` (below 1, the learning rate and the discriminator's learning rate fall smoothly every step, by a factor of lr_decay every lr_decay_steps; 0.8 / 5000 is DiffSinger's pace; the default 1.0 keeps it constant). Checkpoints also record the training settings (batch, learning rate, precision and so on), and a resumed run reports any setting that differs. Each step's time is broken down as well: `time/data_wait_ms` (waiting for data), `time/h2d_ms` (copy to the GPU), `time/step_ms` (compute), `time/steps_per_sec`, `time/eval_s` and `time/save_s`, and the GPU time inside the compute, `time/gpu_{aug,enc,exc,net,gan,bwd,opt}_ms` (`exc` is the pseudo-mel, `net` the velocity network and the losses).
- `gan` — settings for the GAN that sharpens texture. `enabled` (`false` = flow loss + mel loss only), `gan_start_step` (the step at which the GAN turns on), `gan_strength` (strength of the adversarial loss), and so on.
- `data` — `spk_map` / `style_map` (mapping from dataset folder name → speaker ID / style ID), `eval_songs` (songs held out per dataset for evaluation), `eval_dbs` (restrict the held-out songs to these datasets; all of them if absent), `max_sec` (drop phrases longer than this; the dropped names are printed at start-up), `loudness_agc` (slow automatic gain on the target mel: loudness changes slower than `sigma_sec` — phrase-to-phrase level, section-level crescendi — are removed, note-level and phonetic dynamics are kept, so the model sings at a constant level whatever the score; `null` turns it off; choose the values with the [loudness-correction tuner](#loudness-correction-tuner)).

Any key can be overridden on the command line without editing the config: `--set key=value` (for example `--set train.num_steps=1 --set model.pos_cond.phoneme=false`; repeatable).

### Building a dictionary

The Japanese phoneme list is in `dict/ja.phonemes` (one phoneme per line, the order is the ID, `pau` at the top is ID 0, and anything after `#` is a comment). To use a different language or your own phonemes, make a file in the same format and pass it to each command with `--phonemes <file>` (if omitted, the Japanese `dict/ja.phonemes` is used).

Phoneme classes that the list cannot express go in a language pack, `dict/<language>.yaml` (for example `dict/ja.yaml` and `dict/en.yaml`). It points to the phoneme list with `phonemes` and defines `vowels` and `silence`; every other phoneme is a consonant. For the evaluation tool it can also list `devoiced` (devoiced vowels), `voiceless` and `voiced` consonants, and `steady_vowels` (vowels whose long middles are checked for wobble); these are optional. Training and evaluation look in `dict/` for the language pack whose phoneme list matches the model's or the data's. The consonant loss weight (`train.cons_weight`) needs a language pack.

Bundled dictionaries and language packs: Japanese `ja` (the released models), English `en` (CMUdict's 39 phonemes + 6) and `en_full` (54, with the symbols common in singing labels), Korean `ko_coda` (Team CODA's romanisation, 37 phonemes), Mandarin `zh_opencpop` (pinyin initials and finals, 61 phonemes).

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
- Labels that only hold phoneme durations (Opencpop's `transcriptions.txt`, DiffSinger's `transcriptions.csv`), HTK labs that need tidying (`lab`, `labdir`) and GTSinger's IPA labels are converted to per-segment labs with `tools/convert_labels.py` first.
- `save_wav: true` in a recipe also keeps each phrase's waveform in `shard_wav.npz` (needed for playback in the loudness-correction tuner and for `train.pitch_aug`).

### Training

**The V2 recipe (recommended)**: no GAN; train a 16-step teacher, then distil a 1-step student. The Japanese three-singer config is `configs/3singer_v2.yaml`, whose header lists the commands.

    # 1. Teacher (50k steps)
    python -m train --config configs/3singer_v2.yaml \
      --data_dirs data/oniku data/natsume data/ritsu \
      --run_name 3singer_v2_teacher --out_root log --device cuda
    # 2. Student (20k steps from the teacher's weights; the pairs are built automatically and are not saved)
    python -m train --config configs/3singer_v2.yaml \
      --data_dirs data/oniku data/natsume data/ritsu \
      --run_name 3singer_v2_student --out_root log --device cuda --max_updates 20000 \
      --set train.reflow_teacher=log/3singer_v2_teacher/ckpt_050000.pt --set train.reflow_teacher_steps=16 \
      --set train.num_steps=1 --set train.lr=2.0e-4

On one A100 the teacher takes about 3.3 hours, the pairs about 10 minutes and the student about 1.5 hours. The pairs (four per phrase) are held in RAM: about 8.5 GB for the three singers, plus the harmonic cache of about 4.3 GB. The student's config must match the teacher's checkpoint (only `recon_weight`, `num_steps` and `flow_loss` may differ).

**The V1 recipe (GAN)**: one run; the first stage learns the base voice with a flow loss plus a mel loss, the second turns on a GAN to sharpen texture (`configs/3speaker_gan2d.yaml`; the `gan` section controls it, `gan.enabled: false` gives the flow loss plus mel loss only). This is how the v0.1.0 models were made, and it still works.

    python -m train --config configs/3speaker_gan2d.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

In both cases, running the same command again automatically resumes from where it stopped. `--init_from <ckpt>` starts a new run from another checkpoint's weights (the model structure follows that checkpoint, so a v0.1.0 checkpoint works too).

### Loudness-correction tuner

A local web app for choosing `data.loudness_agc` by eye and by ear on a preprocessed dataset (`tools/loudness_tuner/`; it computes with the same `leapsinger/loudness.py` the training uses).

    pip install -e ".[tuner]"
    python tools/loudness_tuner/app.py --data data/ritsu     # → http://127.0.0.1:8018

Point it at a dataset folder (the one holding `shard.npz`) or at a parent folder that holds several. For every phrase it plots the frame loudness, the slow envelope before and after, the gain and the pauses; with data preprocessed with `save_wav: true` you can A/B the original and the normalised audio. It also reports the distribution over the whole dataset (phrase-to-phrase level differences and so on). The YAML box shows the current values ready to paste into `data.loudness_agc`. Details: `tools/loudness_tuner/README.md`.

### Evaluation tools

- `tools/eval_ab.py` — compares trained runs on the held-out phrases (synthesised from the true phoneme durations and F0). Besides the overall error it reports the error at phoneme edges vs. middles, per-band fine-texture ratios, roughness of harmonic peaks and valleys, per-consonant-class spectral shape and centroid shifts, and GAN-like regular patterns, as a table over runs × checkpoints × step counts.
- `tools/render_ab.py` — renders the same phrases TensorBoard shows with several models, as wav files and mel figures (for listening tests).

### Export

    python -m export.cli \
      --ckpt log/<run>/ckpt_020000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 256 --speaker embed --verify

For speaker handling (bake / embed / none) and other details, see "Export to ONNX" below. With `--speaker embed`, `<name>.speakers.json` (every speaker's vector and the phoneme list) is written alongside.

You can try the whole flow — from export to use — in a notebook. Download the model from the Release and place it in `notebooks/sample_data/` (see `place_model_here.txt` in that folder).

    notebooks/export_and_use_onnx.ipynb

The notebook exports the acoustic model to a single ONNX and runs it end to end (phonemes + duration + F0 → mel → audio).

### Export to ONNX

`export/` converts a checkpoint into a self-contained ONNX graph. The excitation and the single-step flow are baked into the graph, so the caller only needs to pass phonemes, durations, and F0. For speaker handling, you can choose:

- **bake** — fix a single voice. This makes the simplest graph (no speaker input).
- **embed** — take a speaker vector as input. One graph can switch to any voice. Each speaker's vector (what to feed as `spk_embed`) and the phoneme list are in the `<name>.speakers.json` written next to the graph.
- **none** — for single-speaker models. No speaker input and no baking (a graph with no notion of speaker). For multi-speaker models, use bake or embed.

A style-bearing model (`n_styles` > 0) is exported with `--variant full`, and the style to use is baked into the graph with `--style-id` (default 0) — one ONNX per style.

As the last post-export pass, the excitation's harmonic sum (a straight-line chain of 256 blocks by default) is folded into a single ONNX `Loop` (`export/harmonic_loop.py`; `--no-harmonic-loop` keeps the unrolled chain). Left unrolled, onnxruntime's default execution order keeps 256 sample-rate arrays alive at once and peak memory reaches 0.28 MB per frame (about 1.5 GB for a 30-second phrase). With the Loop, the 30-second peak drops from +1469 MB to +287 MB, inference is 10–30% faster and session creation is 10× faster. The formula and the order of operations are unchanged, so the output is bit-identical (`--verify` compares against the unrolled form). `Loop` has existed since opset 1 and runs on every runtime, including OpenUTAU's onnxruntime. It can also be applied to an already exported ONNX: `python -m export.harmonic_loop <in.onnx> <out.onnx> --check`

### Vocoder

NHVSing vocoders are bundled under `checkpoints/`.

- `nhv_v3_2_1.onnx` — takes a hop-size-256 mel and F0 (V3.2.1; used for the training previews and the demo).
- `nhv_v3_2_1x.onnx` — the hop-size-512 build.
- `nhv_v3_2.onnx` / `nhv_v3_2x.onnx` — V3.2 (the v0.1.0 vocoders).

V3.2 fixes an abrupt per-frame weakening of the waveform at high pitch by switching the LTV filter's overlap-add to a Hann window; V3.2.1 is V3.2 trained for more epochs. The input/output contract is the same for all of them, so they are drop-in replacements (see [NHVSing](https://github.com/wavtechyukky/NHVSing/) for details).

### Compatibility

- The v0.1.0 models (`3speaker_gan2d.pth`, `3singer_ritsu3style_uv_gan2d.pth`) still run in `infer.py`, export with `export/cli.py` and can seed a run with `--init_from` (they are loaded in their own structure, without the position condition).
- The preprocessed data format (`shard.npz` + `metadata.json`) is unchanged.
- The bundled configs' defaults were changed for V2 training (`harm_decay` 1.5, batch sizes, the position condition, the slow AGC and so on). The exact file the v0.1.0 models were trained with is `git show v0.1.0:configs/3speaker_gan2d.yaml`.
- V2 checkpoints do not load in the v0.1.0 code (they carry the position condition's weights).

## Options

- **Dictionary** — you can specify any phoneme dictionary. Besides Japanese, dictionaries and language packs for English, Korean and Mandarin are bundled, with recipes for the corpora (`configs/recipes/`) and teacher configs (`configs/{en,ko,zh}_teacher.yaml` and others). No models are released for them (English was trained but its quality is not good enough yet; Korean and Mandarin are configs only because of the datasets' terms).
- **v/uv handling** — choose between a mode that feeds voiced/unvoiced (v/uv) as a condition, and a mode that uses only a continuous F0 with the gaps filled by linear interpolation.
- **Excitation tuning** — change the harmonic decay, the number of harmonics, and the white-noise strength.
- **Training recipes** — training conditions are set in YAML (which datasets to use, speaker IDs, per-speaker styles and data, and so on).

## Future work

- **Multilingual support** — the dictionaries, language packs, label conversion and recipes are in place, but only Japanese models are released. The goal is models for other languages from data we may publish, and one model that handles several languages.
- **Consonants** — consonants seen in training are reproduced about as accurately as vowels, but in unseen sequences the timing of plosive and affricate transitions drifts. More data and the consonant loss weight (`train.cons_weight`) are the levers.

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
