"""Database-agnostic lab -> per-phrase training-data pipeline.

Input per song: a phoneme `.lab` label file (`start end phoneme`) aligned to a `.wav`.
Output: per-phrase `.npz` files with phoneme ids, durations, F0 (RMVPE), voiced flag, and a
mel-spectrogram — everything the acoustic model (and the separately-published F0 / duration
models) trains on. Phoneme timing comes straight from the lab; there is no musical score.

Frame / mel settings come from a `MelSpec` (default hop = 256, DiffSinger otherwise)
so the frame size is fully configurable. F0 uses RMVPE only. The mel and F0 keep the **raw**
audio here (breath included) — whether to silence pau/breath is an opinion that belongs to the
data loader (`LeapSingerDataset(silence=...)`), not the data.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from leapsinger.config import MelSpec
from .vocab import DEFAULT_VOCAB
from .phrase_cut import find_phrase_spans, cut_phrases
from .f0_rmvpe import extract_f0_rmvpe
from leapsinger.mel import wav_to_mel_nhv

# Non-canonical silence tokens -> `pau` (breath included: the example DBs are human singers and
# breath is silenced). Edge / GlottalStop stay (they are canonical phonemes).
DEFAULT_SIL_MAP = {"sil": "pau", "br": "pau", "sp": "pau", "sli": "pau", "vf": "pau"}


def read_lab(path: str, unit: str = "100ns"):
    """Read a phoneme `.lab` -> list of (start_sec, end_sec, phoneme).

    unit='100ns' (HTS integer) or 'sec' (decimal seconds).
    """
    scale = 1e-7 if unit == "100ns" else 1.0
    segs = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if len(t) >= 3:
            s, e = float(t[0]) * scale, float(t[1]) * scale
            if e > s:
                segs.append((s, e, t[2]))
    return segs


def load_lab(lab_path: str, *, lab_unit: str = "100ns", sil_map: dict | None = None,
             phon_norm: dict | None = None, lab_fix=None, strict: bool = True, vocab=None):
    """Read + clean a `.lab` into a gap-free `(start_sec, end_sec, phoneme)` timeline:
    normalise phonemes (phon_norm, then silence map -> `pau`), insert `pau` rows over gaps
    (continuity), and merge adjacent `pau`. No score, no note — timing is the lab's own.

    lab_fix: optional per-singer `(intervals)->intervals` hook (a database recipe correction).
    strict=True raises on any phoneme outside the vocab (no silent fallback).
    vocab: the phoneme set to validate against (default: the bundled Japanese vocab).
    """
    vocab = vocab or DEFAULT_VOCAB
    sil_map = {**DEFAULT_SIL_MAP, **(sil_map or {})}
    phon_norm = phon_norm or {}
    segs = read_lab(lab_path, lab_unit)
    if lab_fix is not None:
        segs = lab_fix(segs)
    rows, prev_e, unknown = [], 0.0, set()
    for s, e, ph in segs:
        ph = phon_norm.get(ph, sil_map.get(ph, ph))
        if ph not in vocab:
            unknown.add(ph)
            continue
        if s > prev_e + 1e-6:                                # gap -> pau row (continuity)
            rows.append((prev_e, s, "pau"))
        rows.append((s, e, ph))
        prev_e = e
    merged = []                                              # merge adjacent pau rows
    for r in rows:
        if merged and r[2] == "pau" and merged[-1][2] == "pau" and abs(merged[-1][1] - r[0]) < 1e-9:
            merged[-1] = (merged[-1][0], r[1], "pau")
        else:
            merged.append(r)
    if strict and unknown:
        raise ValueError(f"{lab_path}: phonemes not in canonical vocab: {sorted(unknown)} "
                         f"(add them to phon_norm/sil_map or the vocab)")
    return merged


def process_file(
    name: str,
    lab_path: Path,
    wav_path: Path,
    out_dir: Path,
    *,
    mel: MelSpec | None = None,
    f0_device: str = "cpu",
    lufs_target: float = -23.0,
    save_wav: bool = False,              # also store the per-phrase clip wav (for pitch_aug)
    exclude_spans: list | None = None,   # [[s_sec, e_sec], ...] drop overlapping phrases
    quiet: bool = False,
    lab_unit: str = "100ns",
    sil_map: dict | None = None,
    phon_norm: dict | None = None,
    lab_fix=None,
    vocab=None,                          # phoneme set (default: bundled Japanese vocab)
) -> bool:
    """Turn one (lab, wav) pair into per-phrase npz files under `out_dir/phrases/`."""
    import librosa
    import pyloudnorm as pyln

    mel = mel or MelSpec()
    vocab = vocab or DEFAULT_VOCAB
    sr, hop, frame_rate = mel.sr, mel.hop, mel.frame_rate

    phrase_dir = Path(out_dir) / "phrases"
    if list(phrase_dir.glob(f"{name}_*.npz")):
        if not quiet:
            print("  [SKIP] already done")
        return True

    # ── read + clean the .lab (phoneme timeline; no score, no note) ─────────────
    lab_rows = load_lab(str(lab_path), lab_unit=lab_unit, sil_map=sil_map,
                        phon_norm=phon_norm, lab_fix=lab_fix, vocab=vocab)
    ph_list, dur_sec_l, lab_segs = [], [], []
    for s, e, ph in lab_rows:
        ph_list.append(ph)
        dur_sec_l.append(e - s)
        lab_segs.append((s, e, ph))
    if not ph_list:
        if not quiet:
            print("  [SKIP] empty phoneme sequence")
        return False

    # ── load audio + LUFS normalise (whole song once) ───────────────────────────
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    meter = pyln.Meter(sr)
    cur_lufs = meter.integrated_loudness(audio.astype(float))
    gain = (10.0 ** ((lufs_target - cur_lufs) / 20.0)
            if np.isfinite(cur_lufs) and cur_lufs > -70.0 else 1.0)
    audio = (audio * gain).astype(np.float32)

    # ── phrase spans (cut on long silence) ──────────────────────────────────────
    T_frames  = int(len(audio) // hop)
    total_sec = T_frames / frame_rate
    spans = find_phrase_spans(lab_segs, total_sec, hop_size=hop, sr=sr)
    if not spans:
        spans = [(0, T_frames, 0, 0)]

    if exclude_spans:
        def _hits(sp):
            s_sec, e_sec = sp[0] / frame_rate, sp[1] / frame_rate
            return any(not (e_sec <= xs or s_sec >= xe) for xs, xe in exclude_spans)
        n0 = len(spans)
        spans = [sp for sp in spans if not _hits(sp)]
        if not quiet and len(spans) < n0:
            print(f"  [EXCLUDE] {n0 - len(spans)} phrase(s) dropped")
        if not spans:
            return True

    def _span_data(s_fr: int, e_fr: int):
        n = e_fr - s_fr
        clip = audio[s_fr * hop: e_fr * hop].copy()   # raw audio (no silencing — see module docstring)
        # mel (matches the vocoder's mel exactly: center=False + reflect pad, ln, clamp)
        mel_clip = wav_to_mel_nhv(clip, sr, mel.n_fft, hop, mel.win,
                                  mel.n_mels, mel.fmin, mel.fmax)[:, :n]
        if mel_clip.shape[1] < n:
            mel_clip = np.pad(mel_clip, ((0, 0), (0, n - mel_clip.shape[1])),
                              constant_values=math.log(1e-5))
        # F0 (RMVPE only; raw 0-in-unvoiced — cut_phrases fills gaps per phrase)
        f0_clip, uv_clip = extract_f0_rmvpe(
            np.clip(clip, -1.0, 1.0), sr, hop,
            device=f0_device, interpolate=False)
        f0_clip = (f0_clip[:n] if len(f0_clip) >= n
                   else np.pad(f0_clip, (0, n - len(f0_clip)))).astype(np.float32)
        uv_clip = (uv_clip[:n] if len(uv_clip) >= n
                   else np.pad(uv_clip, (0, n - len(uv_clip)))).astype(np.float32)
        if save_wav:                                     # per-phrase clip wav (for pitch_aug)
            return mel_clip, f0_clip, uv_clip, {}, clip.astype(np.float32)
        return mel_clip, f0_clip, uv_clip, {}

    ph_ids_arr   = np.array([vocab.get(p, 0) for p in ph_list], dtype=np.int16)
    dur_sec_arr  = np.array(dur_sec_l,  dtype=np.float32)

    if not quiet:
        print(f"  {len(spans)} spans (per-phrase mel+F0 @{sr}/{hop}) ...")
    n_written = cut_phrases(
        name=name, lab_segs=lab_segs, ph_ids=ph_ids_arr, dur_sec=dur_sec_arr,
        f0_raw=None, uv=None, mel=None,
        out_dir=phrase_dir, span_data=_span_data, total_frames=T_frames,
        hop_size=hop, sr=sr,
        pau_id=vocab.pau_id, spans=spans, verbose=True,
    )
    if not quiet:
        print(f"  -> {n_written} phrases  ({len(ph_list)} phonemes, {T_frames} frames)")
    return True
