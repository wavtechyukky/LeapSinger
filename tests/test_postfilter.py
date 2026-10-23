"""leapsinger/postfilter.py（フォルマントの山と谷を広げるポストフィルタ）: α = 1 なら何もしない、広げる向きと量、
無音は触らない、フレームの種類、α の決め方。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_ab as ev  # noqa: E402
from leapsinger import postfilter as pf  # noqa: E402
from leapsinger.phonemes import load_lang  # noqa: E402

ID2PH = {0: "pau", 1: "a", 2: "k"}
IDS, DURS = [0, 1, 2, 1, 0], [60, 150, 60, 150, 60]          # pau / a / k / a / pau = 480 フレーム
JA = load_lang("ja")
VOWELS = JA.vowels


def _mel(seed=0):
    """フォルマントくらいの尺度（bin 方向 24 本周期）の山と谷 + 細かいばらつき（倍音の代わり）。"""
    rng = np.random.default_rng(seed)
    env = 2.0 * np.sin(2 * np.pi * np.arange(128) / 24.0)[:, None] + rng.normal(0, 0.5, (128, 1))
    return (env + rng.normal(0, 0.3, (128, 480)) - 5.0).astype(np.float32)


def _value(st, key):
    return ev.metric_value([st], next(m for m in ev.build_metrics() if m["key"] == key))


def test_frame_classes():
    cls = pf.frame_classes(IDS, DURS, ID2PH, 480, VOWELS)
    assert (cls[:60] == -1).all() and (cls[60:210] == 0).all() and (cls[210:270] == 1).all()
    assert (cls[270:420] == 0).all() and (cls[420:] == -1).all()


def test_alpha_one_changes_nothing():
    mel = _mel()
    out = pf.contrast_postfilter(mel, pf.frame_classes(IDS, DURS, ID2PH, 480, VOWELS),
                                 {"vowel": [1.0] * 4, "cons": [1.0] * 4})
    assert np.allclose(out, mel, atol=1e-5)


def test_postfilter_widens_only_the_formant_contrast():
    gt = _mel()
    out = pf.contrast_postfilter(gt, pf.frame_classes(IDS, DURS, ID2PH, 480, VOWELS),
                                 {"vowel": [1.2] * 4, "cons": [1.5] * 4})
    st = ev.phrase_stats(out, gt, IDS, DURS, ID2PH, 172.27, lang=JA)
    for b in range(1, 3):                                  # 内側の帯域（端の bin は移動平均の折り返しがある）
        # 移動平均で測るので、広げた分の 8 割ほどが指標に出る（α 1.2 → 1.16 くらい、1.5 → 1.4 くらい）
        assert 1.1 < _value(st, f"fcon_coarse:vowel:{b}") < 1.22
        assert 1.3 < _value(st, f"fcon_coarse:cons:{b}") < 1.52
        assert abs(_value(st, f"fcon_fine:vowel:{b}") - 1.0) < 0.05     # 倍音の細かいメリハリはほぼそのまま
    assert np.array_equal(out[:, :55], gt[:, :55])          # 無音は触らない（境目の数フレームのならしを除く）
    assert np.array_equal(out[:, 425:], gt[:, 425:])


def test_alphas_from_the_eval_summary():
    summary = {"band_names": ["b0", "b1"], "metrics": {
        f"fcon_coarse:{g}:{b}": {"values": {"decay_15@k1": v}}
        for g, vals in (("vowel", (0.98, 0.8)), ("cons", (0.95, 0.75))) for b, v in enumerate(vals)}}
    r = pf.ratios_from_summary(summary, "decay_15@k1")
    assert r == {"vowel": [0.98, 0.8], "cons": [0.95, 0.75]}
    a1, a2 = pf.alphas_from_ratios(r, 1.0), pf.alphas_from_ratios(r, 2.0)
    assert np.allclose(a1["cons"], [1 / 0.95, 1 / 0.75]) and np.allclose(a2["vowel"], [1 / 0.98 ** 2, 1 / 0.8 ** 2])
