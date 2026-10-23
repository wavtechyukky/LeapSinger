"""生成した mel のポストフィルタ: 周波数方向の粗いメリハリ（フォルマントの山と谷）を広げる（試聴の実験用）。

生成した mel は「ありうる声の平均」に寄って、bin 方向 5〜21 本の尺度の山と谷が正解より縮む（tools/eval_ab.py の
平均化の指標 fcon_coarse。2026-10-18 の decay_15 では子音の 1〜15 kHz で正解の 0.75〜0.84 倍）。mel を bin 方向の
移動平均で 3 つに分け（21 本の平均 = 大まかな傾き、5 本と 21 本の平均の差 = フォルマント、5 本の平均からの
はみ出し = 倍音）、フォルマントの成分だけを α 倍する。α は帯域ごと・母音／子音ごとに決めて、帯域の中心の間は
直線でつなぐ。時間方向は母音と子音の切り替わりで急に変わらないよう移動平均でならす。無音（pau）は触らない。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d

FINE, COARSE = 5, 21                                   # tools/eval_ab.py の FREQ_FINE / FREQ_COARSE
MEL_FLOOR = math.log(1e-5)                             # ln-mel の下限（preprocess の mel の clamp と同じ）


def frame_classes(ph_ids, ph_durs, id2ph: dict, n_frames: int, vowels) -> np.ndarray:
    """フレームごとの種類 [T]: 0 = 母音、1 = 子音など、-1 = 無音（pau）。vowels は言語パックの母音
    （leapsinger/phonemes.py の LangPack.vowels。tools/eval_ab.py と同じ分け方）。"""
    cls = np.full(n_frames, -1, dtype=np.int64)
    fb = np.concatenate([[0], np.cumsum(np.asarray(ph_durs, dtype=np.int64))]).clip(max=n_frames)
    for k, p in enumerate(ph_ids):
        ph = id2ph.get(int(p), str(int(p)))
        if ph != "pau":
            cls[int(fb[k]):int(fb[k + 1])] = 0 if ph in vowels else 1
    return cls


def ratios_from_summary(summary: dict, label: str, key: str = "fcon_coarse") -> dict:
    """eval_ab の summary.json から、モデル label の帯域ごとの比 {"vowel": [...], "cons": [...]}。"""
    m, nb = summary["metrics"], len(summary["band_names"])
    return {g: [float(m[f"{key}:{g}:{b}"]["values"][label]) for b in range(nb)] for g in ("vowel", "cons")}


def alphas_from_ratios(ratios: dict, strength: float = 1.0) -> dict:
    """比 → α = (1 / 比) ** strength。strength 1 で正解並みに戻す、2 でその 2 倍（対数で）広げる。"""
    return {g: [float((1.0 / r) ** strength) for r in v] for g, v in ratios.items()}


def _bin_curve(band_alpha, n_mels: int) -> np.ndarray:
    """帯域ごとの α を、帯域の中心の bin の間で直線につないだ bin ごとの α [n_mels]（両端は外側の帯域の値）。"""
    nb = len(band_alpha)
    w = n_mels // nb
    centers = np.array([b * w + (w - 1) / 2.0 for b in range(nb)])
    return np.interp(np.arange(n_mels), centers, np.asarray(band_alpha, dtype=np.float64))


def contrast_postfilter(mel, cls, alpha: dict, smooth_frames: int = 5) -> np.ndarray:
    """mel [n_mels, T]（ln-mel）のフォルマントの成分を α 倍した mel（float32）を返す。
    cls: frame_classes の [T]。alpha: {"vowel": [帯域ごと], "cons": [...]}。無音のフレームは α = 1。"""
    mel = np.asarray(mel, dtype=np.float64)
    n, T = mel.shape
    cls = np.asarray(cls)[:T]
    A = np.ones((n, T))
    for c, g in ((0, "vowel"), (1, "cons")):
        A[:, cls == c] = _bin_curve(alpha[g], n)[:, None]
    if smooth_frames > 1:
        A = uniform_filter1d(A, smooth_frames, axis=1, mode="nearest")
    formant = uniform_filter1d(mel, FINE, axis=0, mode="nearest") - uniform_filter1d(mel, COARSE, axis=0, mode="nearest")
    return np.maximum(mel + (A - 1.0) * formant, MEL_FLOOR).astype(np.float32)
