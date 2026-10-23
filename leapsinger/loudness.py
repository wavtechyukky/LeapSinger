"""遅い自動ゲイン（slow AGC）— 学習目標 mel から「1 秒程度より遅い音量変化」だけを取り除く。

config の `data.loudness_agc` で有効になり、dataset.py（と train.py のピッチ拡張）が目標 mel に掛ける。
フレーズ間のレベル差・区間レベル・ゆっくりしたクレッシェンドは消え、ノート単位の立ち上がり・減衰・
アクセントと音素由来の速い変化は残る。モデルは巨視的な音量を学ばないので、推論時にどんな譜面を
与えても出力レベルが一定になる（巨視的な強弱はユーザー側で付ける）。

    E[t]    = 20·log10( mean_bins exp(mel[:, t]) )          フレーム音量 [dB]（ボコーダが見る量）
    slow[t] = gsmooth(E, σ·frame_rate, valid=sung)          pau を除いた有効フレームで正規化したガウス平滑
    c[t]    = clip( k·(target_db − slow[t]), −max_cut_db, +max_gain_db )   ゲイン [dB]
    mel'    = mel + c[None, :] / DB                         全 bin へ定数加算（床 MEL_FLOOR で止める）
    wav'    = wav · 10^(c_interp / 20)                      同じゲインを波形へ（試聴用）

numpy と scipy だけに依存する。research/loudness_tuner（調整用 Web アプリ）はこのモジュールを
そのまま import するので、既定値は AGC_DEFAULTS 以外に書かない。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

DB = 20.0 / np.log(10.0)              # ln → dB
MEL_FLOOR = float(np.log(1e-5))       # 前処理の mel クランプ下限（≈ −11.51）

AGC_DEFAULTS = dict(
    sigma_sec=1.0,      # これより遅い音量変化を取り除く（ガウス平滑の σ）
    target_db=-30.0,    # 揃える先のフレーム音量（ritsu / synthv の実測中央値が −28〜−33 dB）
    k=1.0,              # 強さ。1 = 遅い成分を全部取る、0 = 何もしない
    max_gain_db=12.0,   # 持ち上げの上限（ささやき・歌い出しの極端な増幅を防ぐ）
    max_cut_db=12.0,    # 下げの上限
)

# 調整用 UI がフォームを組み立てるための仕様（既定値は AGC_DEFAULTS を参照する）
AGC_SPEC = [
    dict(key="sigma_sec", label="σ [s]", min=0.05, max=5.0, step=0.05,
         title="ガウス平滑の時定数。これより遅い音量変化が取り除かれる。1.0 で 3 秒のクレッシェンドはほぼ消え、"
               "ノート単位（0.3〜1 s）の起伏は残る。2.0 にするとフレーズの弧が一部残る"),
    dict(key="target_db", label="target [dB]", min=-60.0, max=0.0, step=0.5,
         title="揃える先のフレーム音量 20·log10(mean_bins exp(mel))。フレーズ平均がこの値になる"),
    dict(key="k", label="強さ k", min=0.0, max=1.0, step=0.05,
         title="1 = 遅い成分を全部取り除く（推奨）。0.5 なら半分。0 で無効"),
    dict(key="max_gain_db", label="最大持ち上げ [dB]", min=0.0, max=40.0, step=1.0,
         title="ゲインの上限。小さな声の区間が極端に増幅されるのを防ぐ"),
    dict(key="max_cut_db", label="最大下げ [dB]", min=0.0, max=40.0, step=1.0,
         title="ゲインの下限（下げ側）"),
]


def frame_db(mel: np.ndarray) -> np.ndarray:
    """ln-mel [n_mels, T] → フレーム音量 [T] dB。"""
    return DB * np.log(np.exp(np.asarray(mel, dtype=np.float64)).mean(0) + 1e-12)


def gsmooth(x: np.ndarray, sigma_frames: float, valid=None) -> np.ndarray:
    """有効フレームだけで正規化したガウス平滑。端の値を外側へ複製しないので、σ がフレーズ長に
    近くても端（静かな歌い出し・語尾）に引っ張られない。無効フレームは近傍の有効値の加重平均。
    有効フレームから遠すぎて重みがほぼ 0 の所は、最寄りの値で線形補間する。"""
    x = np.asarray(x, dtype=np.float64)
    w = np.ones_like(x) if valid is None else np.asarray(valid, dtype=np.float64)
    if sigma_frames <= 0:
        return x.copy()
    num = gaussian_filter1d(x * w, sigma_frames, mode="constant", cval=0.0)
    den = gaussian_filter1d(w, sigma_frames, mode="constant", cval=0.0)
    out = num / np.maximum(den, 1e-12)
    ok = den > 1e-3 * max(float(den.max()), 1e-12)
    if ok.any() and not ok.all():
        idx = np.arange(len(x))
        out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
    return out


def phoneme_bounds(dur_sec, T: int, frame_rate: float) -> np.ndarray:
    """音素の累積長 [s] → フレーム境界 fb[L+1]（fb[-1] = T）。dataset.py と同じ丸め。"""
    fb = np.round(np.concatenate([[0.0], np.cumsum(np.asarray(dur_sec, dtype=np.float64))]) * frame_rate).astype(int)
    fb = np.clip(fb, 0, T)
    fb[-1] = T
    return fb


def pau_mask(ph_ids, dur_sec, T: int, frame_rate: float, pau_id: int = 0) -> np.ndarray:
    """lab 由来の pau フレーム bool[T]（True = pau）。"""
    fb = phoneme_bounds(dur_sec, T, frame_rate)
    m = np.zeros(T, dtype=bool)
    for k in range(len(ph_ids)):
        if int(ph_ids[k]) == pau_id:
            m[fb[k]:fb[k + 1]] = True
    return m


def _valid(sung, T: int) -> np.ndarray:
    if sung is None:
        return np.ones(T, dtype=bool)
    sung = np.asarray(sung, dtype=bool)
    return sung if sung.any() else np.ones(T, dtype=bool)


def slow_envelope(E: np.ndarray, sung, frame_rate: float, sigma_sec: float) -> np.ndarray:
    """フレーム音量 E [T] dB の遅い包絡（pau を除いて平滑）。"""
    return gsmooth(E, sigma_sec * frame_rate, _valid(sung, len(E)))


def slow_agc_gain(mel: np.ndarray, sung, frame_rate: float, *,
                  sigma_sec: float = AGC_DEFAULTS["sigma_sec"],
                  target_db: float = AGC_DEFAULTS["target_db"],
                  k: float = AGC_DEFAULTS["k"],
                  max_gain_db: float = AGC_DEFAULTS["max_gain_db"],
                  max_cut_db: float = AGC_DEFAULTS["max_cut_db"]) -> np.ndarray:
    """mel [n_mels, T] と pau 以外のマスク sung [T] → ゲイン c [T] dB。"""
    E = frame_db(mel)
    slow = slow_envelope(E, sung, frame_rate, sigma_sec)
    c = k * (target_db - slow)
    return np.clip(c, -abs(max_cut_db), abs(max_gain_db))


def apply_gain_mel(mel: np.ndarray, c_db: np.ndarray) -> np.ndarray:
    """mel + c/DB（全 bin へ定数加算）。前処理の床より下には下げない。"""
    out = np.asarray(mel, dtype=np.float32) + (np.asarray(c_db, dtype=np.float64) / DB)[None, :].astype(np.float32)
    return np.maximum(out, np.float32(MEL_FLOOR))


def normalize_mel(mel: np.ndarray, sung, frame_rate: float, **params) -> np.ndarray:
    """dataloader の入口: 目標 mel [n_mels, T] に遅い AGC を掛けて返す。params は AGC_DEFAULTS のキー。"""
    return apply_gain_mel(mel, slow_agc_gain(mel, sung, frame_rate, **{**AGC_DEFAULTS, **params}))


def apply_gain_wav(wav: np.ndarray, c_db: np.ndarray, hop: int) -> np.ndarray:
    """波形 [n] に同じゲインを掛ける。c はフレーム中心 (t+0.5)·hop に置き、サンプルへ線形補間（端は保持）。"""
    wav = np.asarray(wav, dtype=np.float64)
    T = len(c_db)
    centers = (np.arange(T) + 0.5) * hop
    g = np.interp(np.arange(len(wav)), centers, np.asarray(c_db, dtype=np.float64))
    return (wav * np.power(10.0, g / 20.0)).astype(np.float32)


def agc_curves(mel: np.ndarray, sung, frame_rate: float, **params) -> dict:
    """表示・統計用に途中の量も返す。E / slow_before / c_db / clamped / E_after / slow_after。
    slow_after は適用後の E を同じ σ で再平滑したもの（clamp で非線形になるため式ではなく実測）。"""
    p = {**AGC_DEFAULTS, **params}
    E = frame_db(mel)
    slow_before = slow_envelope(E, sung, frame_rate, p["sigma_sec"])
    c_raw = p["k"] * (p["target_db"] - slow_before)
    c = np.clip(c_raw, -abs(p["max_cut_db"]), abs(p["max_gain_db"]))
    E_after = E + c
    return dict(E=E, slow_before=slow_before, c_db=c, clamped=(c != c_raw), E_after=E_after,
                slow_after=slow_envelope(E_after, sung, frame_rate, p["sigma_sec"]))


def decompose(E: np.ndarray, sung, frame_rate: float, sigma_long: float = 1.0,
              sigma_short: float = 0.3) -> dict:
    """音量包絡の時間スケール分解。有効フレーム上の std [dB]。
    macro = G(σ_long) − 平均、meso = G(σ_short) − G(σ_long)、micro = E − G(σ_short)。"""
    v = _valid(sung, len(E))
    s_short = gsmooth(E, sigma_short * frame_rate, v)
    s_long = gsmooth(E, sigma_long * frame_rate, v)
    m = float(E[v].mean())
    return dict(mean_db=m,
                macro_std=float((s_long - m)[v].std()),
                meso_std=float((s_short - s_long)[v].std()),
                micro_std=float((E - s_short)[v].std()),
                slow_std=float((s_short - m)[v].std()))
