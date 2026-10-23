"""leapsinger/loudness.py — the slow AGC applied to the target mel (`data.loudness_agc`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leapsinger import loudness as L  # noqa: E402

FR = 44100 / 256


def test_constant_mel_is_moved_to_target():
    mel = np.full((128, 400), -5.0, np.float32)                       # E = DB * -5 ≈ -43.4 dB
    out = L.normalize_mel(mel, np.ones(400, bool), FR, sigma_sec=1.0, target_db=-30.0, k=1.0,
                          max_gain_db=40, max_cut_db=40)
    assert np.allclose(L.frame_db(out), -30.0, atol=1e-3)


def test_clamp_limits_gain():
    mel = np.full((128, 300), -8.0, np.float32)
    c = L.slow_agc_gain(mel, np.ones(300, bool), FR, target_db=-30.0, max_gain_db=12.0, max_cut_db=6.0)
    assert np.allclose(c, 12.0)
    c = L.slow_agc_gain(np.zeros((128, 300), np.float32), np.ones(300, bool), FR, target_db=-30.0,
                        max_gain_db=12.0, max_cut_db=6.0)
    assert np.allclose(c, -6.0)


def test_pau_frames_do_not_influence_the_envelope():
    T = 1000
    sung = np.ones(T, bool); sung[400:480] = False
    mel = np.full((128, T), -5.0, np.float32)
    mel2 = mel.copy(); mel2[:, 400:480] = 1.5                          # loud breath inside the pau
    a = L.slow_envelope(L.frame_db(mel), sung, FR, 1.0)
    b = L.slow_envelope(L.frame_db(mel2), sung, FR, 1.0)
    assert np.allclose(a, b)


def test_slow_removed_fast_kept():
    T = 4000
    t = np.arange(T) / FR
    slow = 4.0 * np.sin(2 * np.pi * t / 20.0)
    fast = 3.0 * np.sin(2 * np.pi * t * 4.0)
    E = -30.0 + slow + fast
    mel = np.repeat((E / L.DB)[None, :], 128, 0).astype(np.float32)
    out = L.normalize_mel(mel, np.ones(T, bool), FR, sigma_sec=1.0, target_db=-30.0, k=1.0,
                          max_gain_db=40, max_cut_db=40)
    E2 = L.frame_db(out)
    mid = slice(400, T - 400)
    assert np.abs(E2[mid] - (-30.0 + fast[mid])).max() < 0.6
    assert np.all(out >= np.float32(L.MEL_FLOOR))


def test_wav_gain_matches_mel_gain():
    T, hop = 50, 256
    wav = np.random.default_rng(0).standard_normal(T * hop).astype(np.float32)
    c = np.linspace(-6, 6, T)
    w2 = L.apply_gain_wav(wav, c, hop)
    assert w2.shape == wav.shape
    assert np.isclose(w2[0] / wav[0], 10 ** (-6 / 20), atol=1e-4)
    assert np.isclose(w2[-1] / wav[-1], 10 ** (6 / 20), atol=1e-4)


def test_mel_gain_stops_at_the_floor():
    T = 50
    mel = np.full((128, T), L.MEL_FLOOR, np.float32)
    m2 = L.apply_gain_mel(mel, np.full(T, -10.0))
    assert np.all(m2 >= np.float32(L.MEL_FLOOR))                       # 床より下へは行かない
    m3 = L.apply_gain_mel(mel, np.full(T, 10.0))
    assert np.allclose(L.frame_db(m3) - L.frame_db(mel), 10.0, atol=1e-3)


def test_decompose_splits_the_envelope_into_bands():
    T = 6000
    t = np.arange(T) / FR
    macro = 2.0 * np.sin(2 * np.pi * t / 20.0)           # σ_long=1 s でほぼ全部 macro に入る周期
    micro = 5.0 * np.sin(2 * np.pi * t * 6.0)
    E = -30.0 + macro + micro
    d = L.decompose(E, np.ones(T, bool), FR, sigma_long=1.0, sigma_short=0.3)
    assert abs(d["mean_db"] + 30.0) < 0.2
    assert abs(d["macro_std"] - macro.std()) < 0.3
    assert abs(d["micro_std"] - micro.std()) < 0.6
    assert d["meso_std"] < 0.5


def test_pau_mask_follows_the_phoneme_bounds():
    ph = np.array([0, 5, 6, 0])
    dur = np.array([0.5, 0.3, 0.2, 0.4])
    T = int(round(dur.sum() * FR))
    m = L.pau_mask(ph, dur, T, FR, pau_id=0)
    fb = L.phoneme_bounds(dur, T, FR)
    assert m[:fb[1]].all() and not m[fb[1]:fb[3]].any() and m[fb[3]:].all()
    assert fb[-1] == T
