"""RMVPE-only F0 extraction for LeapSinger preprocessing.

The reproducible public pipeline uses **RMVPE alone** as the pitch estimator. A single
neural estimator occasionally emits an isolated, badly-wrong single frame; to guard
against that we apply a light **single-frame outlier despike** (100-cent threshold):
a voiced frame whose pitch differs from *both* voiced neighbours by more than 100 cents
is treated as a misdetection and dropped (then filled by interpolation). Lone voiced
frames (surrounded by unvoiced) are dropped too. This recovers the isolated-outlier
robustness that a multi-estimator consensus would otherwise provide, without any of it.

Output convention (matches the dataset loader):
    f0_interp : float32 [T]  gap-less F0 in Hz (unvoiced regions linearly interpolated)
    uv        : float32 [T]  1.0 = voiced, 0.0 = unvoiced
"""
from __future__ import annotations

import numpy as np

from .algorithms.rmvpe import RMVPEPitchAlgorithm

# Isolated single-frame jump threshold. Same magnitude used by robust multi-estimator
# pitch pipelines for per-frame outlier rejection.
OUTLIER_CENTS = 100.0

_ALGO_CACHE: dict = {}


def _cents(f0: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(np.maximum(f0, 1e-5))


# RMVPE の出力は 360 ビン固定でこの範囲しか取り得ない（実測）。推定レンジは
# モデルに渡らないので、ここは「値域の宣言」であって調整ノブではない。
RMVPE_FMIN, RMVPE_FMAX = 31.7, 2005.5


def _get_algo(sample_rate: int, hop_size: int, device: str = "cpu"):
    """Cache one RMVPE model per (sr, hop, device) — it loads a neural net."""
    key = (int(sample_rate), int(hop_size), str(device))
    if key not in _ALGO_CACHE:
        _ALGO_CACHE[key] = RMVPEPitchAlgorithm(
            sample_rate=sample_rate, hop_size=hop_size,
            fmin=RMVPE_FMIN, fmax=RMVPE_FMAX, device=device,
        )
    return _ALGO_CACHE[key]


def despike_isolated(f0: np.ndarray, voiced: np.ndarray, cents: float = OUTLIER_CENTS):
    """Drop isolated single-frame pitch outliers (mark unvoiced). Returns (f0, voiced).

    A voiced frame is dropped when it is either (a) a lone voiced frame surrounded by
    unvoiced, or (b) differs from every present voiced neighbour by > `cents`.
    """
    f0 = np.asarray(f0, np.float32).copy()
    voiced = np.asarray(voiced, bool).copy()
    c = _cents(f0)
    n = len(f0)
    for t in range(n):
        if not voiced[t]:
            continue
        lv = bool(voiced[t - 1]) if t - 1 >= 0 else False
        rv = bool(voiced[t + 1]) if t + 1 < n else False
        if not lv and not rv:
            voiced[t] = False                                   # lone voiced frame
            continue
        dl = abs(c[t] - c[t - 1]) if lv else None
        dr = abs(c[t] - c[t + 1]) if rv else None
        jumps = [d for d in (dl, dr) if d is not None]
        if jumps and all(d > cents for d in jumps):             # differs from all neighbours
            voiced[t] = False
    f0[~voiced] = 0.0
    return f0, voiced


def octave_correct(f0: np.ndarray, voiced: np.ndarray,
                   max_run: int = 3, win: int = 15, tol: float = 0.20) -> np.ndarray:
    """Fix short octave-doubling/halving errors that the neighbour despike misses.

    A single neural estimator sometimes locks onto the 2nd harmonic (or sub-harmonic) for
    a few consecutive frames — an octave error that is smooth in its own sequence, so a
    neighbour-difference test never fires. Here each voiced frame's pitch is compared to
    the local median (voiced frames within +-win); frames sitting ~1 octave from it are
    grouped into runs, and only SHORT runs (<= max_run frames) are snapped to the near
    octave. Sustained octave jumps (real melodic leaps) form long runs and are left alone.
    """
    f0 = np.asarray(f0, np.float32).copy()
    v = np.asarray(voiced, bool)
    n = len(f0)
    lf = np.log2(np.maximum(f0, 1e-5))
    off = np.zeros(n, np.int8)                       # -1 / 0 / +1 octaves from local median
    for t in range(n):
        if not v[t]:
            continue
        a, b = max(0, t - win), min(n, t + win + 1)
        w = v[a:b].copy(); w[t - a] = False
        if w.sum() < 3:
            continue
        d = lf[t] - float(np.median(lf[a:b][w]))
        if abs(d - 1.0) < tol:
            off[t] = 1
        elif abs(d + 1.0) < tol:
            off[t] = -1
    t = 0
    while t < n:
        if off[t] != 0:
            j = t
            while j < n and off[j] == off[t]:
                j += 1
            if (j - t) <= max_run:                   # short run = error, snap toward median
                f0[t:j] = f0[t:j] / 2.0 if off[t] == 1 else f0[t:j] * 2.0
            t = j
        else:
            t += 1
    return f0


def interp_f0(f0: np.ndarray) -> np.ndarray:
    """Linearly interpolate F0 (Hz) over unvoiced (0) frames → gap-less contour."""
    f0 = np.asarray(f0, np.float32)
    voiced = f0 > 0
    if not voiced.any():
        return np.zeros_like(f0, dtype=np.float32)
    idx = np.arange(len(f0))
    return np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)


def extract_f0_rmvpe(
    wav: np.ndarray,
    sample_rate: int,
    hop_size: int,
    device: str = "cpu",
    despike: bool = True,
    octave_fix: bool = False,      # opt-in: RMVPE was verified clean on the example DBs
    interpolate: bool = True,
):
    """wav → (f0, uv[1=voiced/0=unvoiced]) at frame rate sr/hop.

    RMVPE only + 100-cent single-frame despike. The audio is peak-normalised to [-1, 1]
    first (RMVPE is amplitude sensitive), matching the estimator's expected input.

    interpolate=True  → f0 is gap-less (unvoiced filled by linear interp) — for direct use.
    interpolate=False → f0 is 0 in unvoiced frames (raw) — the phrase-cutter fills gaps
                        per phrase (holding F0 flat at phrase edges), so pass raw there.
    """
    w = np.asarray(wav, np.float32)
    peak = np.abs(w).max()
    if peak > 1e-6:
        w = np.clip(w / peak, -1.0, 1.0)
    algo = _get_algo(sample_rate, hop_size, device)
    f0_raw, voiced_flag, _ = algo.extract_pitch(w)
    voiced = np.asarray(voiced_flag, bool)
    f0_raw = np.asarray(f0_raw, np.float32)
    f0_raw[~voiced] = 0.0
    if despike:
        f0_raw, voiced = despike_isolated(f0_raw, voiced)
    if octave_fix:
        f0_raw = octave_correct(f0_raw, voiced)
    uv = voiced.astype(np.float32)
    f0_out = interp_f0(f0_raw) if interpolate else f0_raw
    return f0_out, uv
