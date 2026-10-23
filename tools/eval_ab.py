"""A/B 評価: 学習した run どうしを評価用の句で比べる（位置情報・擬似 mel のノイズ量など）。

    python tools/eval_ab.py --config configs/3speaker_gan2d.yaml \
        --data_dirs /content/data/oniku /content/data/natsume /content/data/ritsu \
        --runs log/base_v2 log/noise_020 \
        --steps 31000 32000 33000 34000 35000 \
        --out log/eval_noise_gan --device cuda

（旧名 tools/eval_position.py でも同じものが動く）

reflow の教師のように複数 step で解くモデルは、--num_steps 1 4 8 16 のように step 数を並べると、run ごとに
「名前@k8」のような step 数別のモデルとして比べられる（既定はチェックポイントに記録された step 数）。
run ごとに step 数を決めるときは --runs log/decay_15@1 log/teacher_s015@16 のように書く（--num_steps より優先）。
使うチェックポイントも run ごとに決めるときは log/student@1:16000,17000,18000,19000,20000 のように書く（--steps より
優先）。基準と step が重ならない run は、チェックポイントを並び順（学習の進んだ方から）で組にして比べる。

考え方
------
評価用の句（学習に使っていない曲）を、正解の音素長と F0 を与えて合成すると、生成 mel は正解と
フレーム単位で揃う。そこで「どこで・どんな差が出ているか」を分けて測る:

  全体  * 無音以外の全フレームの誤差
  位置  * 端 − 中ほど = 音素の端（頭・終わり 0–20 ms）の誤差 − 中ほど（相対位置 0.4–0.6）の誤差
            … 位置情報が効いたかの決め手。モデル全体が一律に良い・悪いズレには左右されにくい
        * 音素の中の相対位置（0〜1 を 10 区間）・頭から / 終わりまでの距離・音素の長さごとの誤差
        * 境界のつなぎの鋭さ = 境界付近の |Δmel| の 生成/正解 比（1 に近いほど良い。1 未満はぼやけ）
        * 長い母音（300 ms 以上）の中ほどの揺れ = |Δmel| の 生成/正解 比
  帯域  mel の bin を 4 等分した帯域ごとに（擬似 mel のノイズ量・傾き・高域の判別器などの A/B 用）
        * 微細テクスチャ = mel から時間方向 5 フレームの移動平均を引いた残り（細かい揺らぎ）の RMS の
          生成/正解 比（議論メモ §1.2 と同じ定義）。1 未満はのっぺり、1 を超えるとざらつき過多
        * 誤差、レベルのずれ（生成 − 正解）
        細部が正しく出るようになっても、ランダムな細部は正解と一致しないので、誤差はむしろ増える
        ことがある。テクスチャの A/B は誤差ではなくテクスチャ比と試聴で見る。
  平均化  生成が「ありうる声の平均」に寄って縮んでいないかを、尺度ごとに 生成/正解 の RMS 比で見る
        （1 未満 = 縮んでいる）。帯域ごと
        * 周波数方向のメリハリ: 細かい = bin 方向 5 本の移動平均からのはみ出し（母音では倍音の山と谷）、
          粗い = 5 本と 21 本の移動平均の差（スペクトルの形の山と谷。母音ではフォルマント、無声子音では
          狭めの前の空間の共鳴でできる雑音の山）。母音・子音のフレーム別
        * 時間方向の動き: 移動平均の幅の差で切り出した 約 30〜150 ms（音素の速さ、母音・子音別）と
          約 150〜600 ms（音節・音符の速さ）の変化。約 30 ms 以下は上の微細テクスチャ
        * GV: bin ごとの、無音以外のフレームでの分散（動く幅全体）
  子音の種類別  「子音など」を 無声化した母音（A I U E O）・無声子音（k s sh t ts ch h f p と拗音）・
        有声子音（g z j d n b m y r w v と拗音）・その他（cl・息など）に分けて（音素で分ける）
        * 誤差、周波数方向のメリハリ（粗い・細かい、帯域ごと）
        * スペクトルの重心のずれ（生成 − 正解、Hz）: 雑音の山の位置（s は高く sh は低いなど、子音を聞き分ける
          手がかり）がずれていないか。重心 = mel の振幅で重みを付けた mel フィルタの中心周波数の平均
        * 子音ごとの表: フレーム数が足りる子音ごとに、誤差・973 Hz 以上の粗いメリハリ・重心のずれ
  倍音  有声のフレームで、正解の F0 から倍音（k·F0）と倍音の間の谷（(k+0.5)·F0）の位置を決めて mel を読む（bin の
        間隔が F0/3 以下で 3 kHz 以下の所だけ。1 kHz 未満と 1〜3 kHz に分ける）
        * 山と谷の深さ（生成/正解）、山・谷それぞれのレベルのずれ（生成 − 正解）
        * 山・谷それぞれのざらつき = 3 フレームでの 2 階差分の大きさ（生成/正解）。ゆっくりした動き（ビブラートなど）は
          打ち消し、フレームごとのギザギザだけを見る
        谷だけが持ち上がっていれば波形のノイズのような埋まり方、山までざらついていれば対数の mel で足したノイズの残り
  模様  GAN に起因するような規則的なパターンが入っていないか。帯域ごとに mel を 64 フレーム × 32 bin のタイルに切り、
        2 次元のパワースペクトルを 生成/正解 の比（dB）で比べて、周りの成分より突出している分の最大を見る。
        平均化でなだらかに減るのは数えない

誤差は mel の各 bin の |生成 − 正解| の平均を dB で表したもの（無音 pau は除く）。

学習の揺れ: 学習率が一定で EMA も無いので、同じ学習の中でもチェックポイントごとに良し悪しが揺れる
（2026-10-17 の評価では base_agc の 18000 だけ全体誤差が他より 0.8 dB 悪く、それだけで「改善」と
出ていた。直前の step が 1 本だけの極小バッチだった）。そこで
  * 指標はチェックポイントごとに計算し、チェックポイント間の中央値で代表させる
  * 比較は同じ step 同士の差（モデル − 基準）をチェックポイントごとに出し、その中央値に、句を復元抽出
    するブートストラップで 95% 区間を付ける
  * 「改善」「悪化」は、区間が 0 をまたがず、しかも全部のチェックポイントで差の向きが同じときだけ
チェックポイントは 5 つ以上を推奨。GAN の段階（20k 以降）と flow だけの段階は別々に評価する。
--runs の最初が基準。出力: <out>/report.md（日本語の要約）、summary.json、図 5 枚。

ローカルで回す場合は torch 1 プロセス・スレッド 2 本で（--threads）。Colab の GPU なら数分。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dataset import LeapSingerDataset  # noqa: E402
from infer import infer_mel, load_acoustic  # noqa: E402
from leapsinger.phonemes import CLASSES, LangPack, find_lang  # noqa: E402  音素の分け方は言語パック（dict/<lang>.yaml）

DB = 20.0 / np.log(10.0)                                    # ln → dB
NO_LANG = LangPack(name="", phonemes=(), vowels=frozenset(), silence=frozenset({"pau"}))   # 言語パックが無いとき
GROUPS = ("vowel", "cons")
GROUP_JA = {"vowel": "母音", "cons": "子音など"}
GROUP_EN = {"vowel": "vowels", "cons": "consonants etc."}         # 図のラベル（Colab に日本語フォントが無いため）
GROUP_SHORT = {"vowel": "vowel", "cons": "cons."}
POS_BINS = 10
DIST_EDGES_MS = [0.0, 20.0, 50.0, 100.0, np.inf]
LEN_EDGES_MS = [0.0, 50.0, 150.0, 300.0, np.inf]
STEADY_MIN_MS, STEADY_GUARD_MS = 300.0, 30.0
BANDS = 4                                                   # mel の bin を等分した帯域（128 本なら 32 本ずつ）
TEX_WIN = 5                                                 # 微細テクスチャ = mel − 時間方向 5 フレーム移動平均
FREQ_FINE, FREQ_COARSE = 5, 21                              # 周波数方向の移動平均の幅（bin）: 倍音 / フォルマント
MOD_PH, MOD_SY = (5, 25), (25, 101)                         # 時間方向の移動平均の幅の組（フレーム）: 約 30〜150 ms / 150〜600 ms
CLASS_JA = {"devoiced": "無声化した母音", "ucons": "無声子音", "vcons": "有声子音", "other": "その他"}  # 「子音など」の内訳
CLASS_EN = {"devoiced": "devoiced vowels", "ucons": "voiceless consonants", "vcons": "voiced consonants",
            "other": "other"}
PH_MIN_FRAMES = 300                                                        # 子音ごとの表に載せる最小のフレーム数（全句）
PAT_FRAMES, PAT_HOP = 64, 32       # 模様: タイルの長さ（フレーム、約 0.37 秒）とずらし幅。周波数方向は帯域（32 bin）ごと
PAT_EXCLUDE = 1                    # 模様の突出から除く中心付近の低い成分（|kf| ≤ 1 かつ kt ≤ 1 = タイルの中の大まかな傾き）
PAT_LOCAL = 5                      # 突出 = 比（dB）から周りの 5 × 5 の中央値を引いた分（なだらかな減り方は数えない）
HARM_FMAX, HARM_SPLIT, HARM_RES = 3000.0, 1000.0, 3.0   # 倍音: 谷が 3 kHz 以下・1 kHz で分ける・bin の間隔 ≤ F0/3
HARM_RANGES = ("lo", "mid")
HARM_RANGE_JA = {"lo": "1 kHz 未満", "mid": "1〜3 kHz"}
ALL = dict(key="all", kind="mean", keys=["all"])
MIN_STEPS_ADVISED = 5
SECTIONS = (("all", "全体"), ("position", "音素の中の位置"), ("band", "帯域（擬似 mel のノイズ量などの A/B 用）"),
            ("unvoiced", "無声の区間（子音など）"),
            ("averaging", "平均化（メリハリと動きの幅。生成/正解、1 未満 = 縮んでいる）"),
            ("consonants", "子音の種類別（無声化した母音・無声子音・有声子音・その他）"),
            ("harmonics", "倍音の構造（有声のフレーム。正解の F0 から倍音と谷の位置を決める）"))


_MEL_HZ: dict = {}


def mel_hz(n_mels: int = 128, fmin: float = 40.0, fmax: float = 16000.0) -> np.ndarray:
    """mel フィルタの中心周波数 [n_mels]（Hz。band_labels と同じ librosa の Slaney mel）。"""
    key = (int(n_mels), float(fmin), float(fmax))
    if key not in _MEL_HZ:
        import librosa
        _MEL_HZ[key] = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
    return _MEL_HZ[key]


def _bin_label(edges, j):
    lo, hi = edges[j], edges[j + 1]
    return f"{lo:.0f}-{hi:.0f} ms" if np.isfinite(hi) else f">={lo:.0f} ms"


def _k(step: int) -> str:
    return f"{step / 1000:g}k"


def _hz(x: float) -> str:
    return f"{x / 1000:.1f} kHz" if x >= 1000 else f"{x:.0f} Hz"


def band_labels(n_mels: int = 128, fmin: float = 40.0, fmax: float = 16000.0, bands: int = BANDS):
    """各帯域に入る mel フィルタの中心周波数の範囲（librosa の Slaney mel = leapsinger/mel.py と同じ）。"""
    import librosa
    f = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]
    nb = n_mels // bands
    return [f"{_hz(f[b * nb])}–{_hz(f[(n_mels if b == bands - 1 else (b + 1) * nb) - 1])}" for b in range(bands)]


# ── 1 句ぶんの集計 ─────────────────────────────────────────────────────────────

def _window_ok(mask: np.ndarray, half: int) -> np.ndarray:
    """mask の中で、前後 half フレームも mask に入っているフレーム（移動平均の窓が外にかからない）。"""
    T = len(mask)
    ok = mask.copy()
    for d in range(1, half + 1):
        ok[d:] &= mask[:-d]
        ok[:-d] &= mask[d:]
    ok[:half] = False
    ok[max(T - half, 0):] = False
    return ok


def _harmonic_stats(pred, gt, f0_logf0, voiced, hz, add) -> None:
    """倍音の構造（phrase_stats の一部）。有声のフレームで、正解の F0 から倍音 k·F0 と谷 (k+0.5)·F0 の位置を決め、
    mel をその周波数で読む（bin の中心周波数の間を直線で補間、dB）。谷が HARM_FMAX 以下で、そこの bin の間隔が
    F0 / HARM_RES 以下の倍音だけ。山と谷の深さ・レベルのずれ・ざらつき（3 フレームでの 2 階差分の大きさ。周波数は
    真ん中のフレームの倍音の位置）を、倍音が HARM_SPLIT 未満か以上かで分けて足す。"""
    n, T = gt.shape
    if T < 3 or not voiced.any():
        return
    f0 = np.power(2.0, np.asarray(f0_logf0, dtype=np.float64)[:T])
    kmax = int(np.ceil(HARM_FMAX / max(float(f0[voiced].min()), 50.0)))
    k = np.arange(1, kmax + 1, dtype=np.float64)
    Hk = f0[:, None] * k[None, :]                                           # [T, K] 倍音
    Vk = Hk + 0.5 * f0[:, None]                                             # その上の谷
    spacing = np.interp(Vk, (hz[:-1] + hz[1:]) / 2.0, np.diff(hz))          # 谷のあたりの bin の間隔
    ok = voiced[:, None] & (Vk <= HARM_FMAX) & (spacing <= f0[:, None] / HARM_RES)
    if not ok.any():
        return
    bins = np.arange(n, dtype=np.float64)

    def read(m, freq, frames):                                              # m を周波数 freq [T', K]・フレーム [T'] で読む
        bi = np.interp(freq, hz, bins)
        i0 = np.clip(np.floor(bi).astype(np.int64), 0, n - 2)
        w = bi - i0
        cols = np.broadcast_to(np.asarray(frames)[:, None], freq.shape)
        return (m[i0, cols] * (1.0 - w) + m[i0 + 1, cols] * w) * DB

    tt = np.arange(T)
    lv = {who: (read(m, Hk, tt), read(m, Vk, tt)) for who, m in (("pred", pred), ("gt", gt))}
    gr = {who: [np.abs(read(m, F[1:-1], tt[:-2]) - 2.0 * read(m, F[1:-1], tt[1:-1]) + read(m, F[1:-1], tt[2:]))
                for F in (Hk, Vk)] for who, m in (("pred", pred), ("gt", gt))}      # [T-2, K]
    inner = voiced[:-2] & voiced[1:-1] & voiced[2:]                         # 前後のフレームも有声
    (Pp, Vp), (Pg, Vg) = lv["pred"], lv["gt"]
    for gi, g in enumerate(HARM_RANGES):
        sel = ok & ((Hk >= HARM_SPLIT) == bool(gi))
        if not sel.any():
            continue
        add(f"harm_depth:{g}:pred", (Pp - Vp)[sel])
        add(f"harm_depth:{g}:gt", (Pg - Vg)[sel])
        add(f"harm_peak_bias:{g}", (Pp - Pg)[sel])
        add(f"harm_valley_bias:{g}", (Vp - Vg)[sel])
        si = sel[1:-1] & inner[:, None]
        if si.any():
            for j, part in enumerate(("peak", "valley")):
                add(f"harm_grain_{part}:{g}:pred", gr["pred"][j][si])
                add(f"harm_grain_{part}:{g}:gt", gr["gt"][j][si])


def phrase_stats(pred: np.ndarray, gt: np.ndarray, ph_ids, ph_durs, id2ph: dict, fr: float,
                 uv=None, mel_centers=None, f0_logf0=None, lang: LangPack | None = None) -> dict:
    """pred/gt [n_mels, T] ln-mel → {キー: [合計, 個数]}。キーは下の build_metrics と図が使う。
    uv（正解の有声フラグ [T]、1 = 有声）を渡すと、無声の区間だけの帯域の指標も出す。
    mel_centers: mel フィルタの中心周波数 [n_mels]（スペクトルの重心用。既定は mel_hz(n_mels) = 40 Hz〜16 kHz）。
    f0_logf0: 正解の F0（log2 Hz、[T]）。uv と一緒に渡すと倍音の構造の指標も出す。
    lang: 母音と子音の分け方（言語パック、leapsinger/phonemes.py）。None なら母音が無い扱い（pau 以外は全部
    「子音など」・子音の種類は「その他」）で、母音と子音で分ける指標は意味を持たない（ほかの指標は変わらない）。"""
    lang = lang or NO_LANG
    T = gt.shape[1]
    pred = pred[:, :T]
    err = np.abs(pred - gt).mean(0) * DB                                   # [T] dB
    d_gt = np.abs(np.diff(gt, axis=1)).mean(0) * DB                         # [T-1]
    d_pr = np.abs(np.diff(pred, axis=1)).mean(0) * DB
    fb = np.concatenate([[0], np.cumsum(np.asarray(ph_durs, dtype=np.int64))]).clip(max=T)
    acc = defaultdict(lambda: [0.0, 0])

    def add(k, v):
        if v.size:
            acc[k][0] += float(v.sum())
            acc[k][1] += int(v.size)

    names = [id2ph.get(int(p), str(int(p))) for p in ph_ids]
    for k, ph in enumerate(names):
        s, e = int(fb[k]), int(fb[k + 1])
        if e <= s or ph == "pau":
            continue
        g = "vowel" if ph in lang.vowels else "cons"
        n = e - s
        seg = err[s:e]
        add("all", seg)
        add(f"grp:{g}", seg)
        i = np.arange(n)
        p = np.full(n, 0.5) if n == 1 else i / (n - 1)
        pb = np.minimum((p * POS_BINS).astype(int), POS_BINS - 1)
        for b in range(POS_BINS):
            add(f"pos:{g}:{b}", seg[pb == b])
        ms_on = (i + 0.5) / fr * 1000.0                                      # 音素の頭からの距離
        ms_off = (n - 1 - i + 0.5) / fr * 1000.0                             # 音素の終わりまでの距離
        for j in range(len(DIST_EDGES_MS) - 1):
            lo, hi = DIST_EDGES_MS[j], DIST_EDGES_MS[j + 1]
            add(f"onset:{g}:{j}", seg[(ms_on >= lo) & (ms_on < hi)])
            add(f"offset:{g}:{j}", seg[(ms_off >= lo) & (ms_off < hi)])
        dur_ms = n / fr * 1000.0
        lj = int(np.searchsorted(LEN_EDGES_MS, dur_ms, side="right") - 1)
        add(f"len:{lj}", seg)
        if ph in lang.steady and dur_ms >= STEADY_MIN_MS:                    # 長い母音の中ほどの揺れ
            gd = int(round(STEADY_GUARD_MS / 1000.0 * fr))
            a, b = s + gd, e - gd - 1
            if b > a:
                add("steady:pred", d_pr[a:b])
                add("steady:gt", d_gt[a:b])
    for k in range(1, len(names)):                                           # 境界のつなぎの鋭さ
        if names[k - 1] == "pau" or names[k] == "pau":
            continue
        b = int(fb[k])                                                       # フレーム b-1 → b が境界
        if b < 3 or b > T - 3:
            continue
        add("sharp:pred", d_pr[b - 2:b + 1])                                 # 境界をまたぐ 3 つの差分
        add("sharp:gt", d_gt[b - 2:b + 1])

    # 帯域ごと: 誤差・レベルのずれ・微細テクスチャ。無音（pau）と、移動平均の窓が無音や端にかかる
    # フレームは数えない
    sil = np.zeros(T, dtype=bool)
    for k, ph in enumerate(names):
        if ph == "pau":
            sil[int(fb[k]):int(fb[k + 1])] = True
    half = TEX_WIN // 2
    from scipy.ndimage import uniform_filter1d
    res_p_all = (pred - uniform_filter1d(pred, TEX_WIN, axis=1, mode="nearest")) * DB
    res_g_all = (gt - uniform_filter1d(gt, TEX_WIN, axis=1, mode="nearest")) * DB
    diff_all = (pred - gt) * DB
    nb = gt.shape[0] // BANDS
    band_rows = [slice(b * nb, gt.shape[0] if b == BANDS - 1 else (b + 1) * nb) for b in range(BANDS)]
    ok = _window_ok(~sil, half)
    if ok.any():
        for b, rows in enumerate(band_rows):
            add(f"band_err:{b}", np.abs(diff_all[rows][:, ok]))
            add(f"band_bias:{b}", diff_all[rows][:, ok])
            add(f"band_tex:pred:{b}", res_p_all[rows][:, ok] ** 2)
            add(f"band_tex:gt:{b}", res_g_all[rows][:, ok] ** 2)

    # 無声の区間（正解の F0 で無声、無音 pau は除く）: 子音の作り。低域のレベルが正解より高ければ、擬似 mel の
    # 倍音の消し残りで子音がにごっている可能性がある。揺らぎは窓が無声の区間に収まるフレームだけ
    if uv is not None:
        unv = (~sil) & (np.asarray(uv, dtype=np.float32)[:T] < 0.5)
        if unv.any():
            add("uv_all", err[unv])
            ok_u = _window_ok(unv, half)
            for b, rows in enumerate(band_rows):
                add(f"uv_band_bias:{b}", diff_all[rows][:, unv])
                if ok_u.any():
                    add(f"uv_band_tex:pred:{b}", res_p_all[rows][:, ok_u] ** 2)
                    add(f"uv_band_tex:gt:{b}", res_g_all[rows][:, ok_u] ** 2)

    # 平均化（メリハリと動きの幅）: 生成と正解それぞれの二乗和（比は build_metrics の rms_ratio）。
    # 周波数方向は母音・子音のフレーム別、時間方向は窓が無音にかからないフレームだけ（30〜150 ms は窓の中心の
    # フレームで母音・子音に分ける）、GV は無音以外のフレームの bin ごとの分散
    cls = np.full(T, -1)                                                     # 0 = 母音・撥音、1 = 子音など、-1 = pau
    for k, ph in enumerate(names):
        if ph != "pau":
            cls[int(fb[k]):int(fb[k + 1])] = 0 if ph in lang.vowels else 1

    def ma(m, w, axis):
        return uniform_filter1d(m, w, axis=axis, mode="nearest")

    parts = {}
    for who, m in (("pred", pred), ("gt", gt)):
        f5 = ma(m, FREQ_FINE, 0)
        parts[who] = dict(fcon_fine=(m - f5) * DB, fcon_coarse=(f5 - ma(m, FREQ_COARSE, 0)) * DB,
                          mod_ph=(ma(m, MOD_PH[0], 1) - ma(m, MOD_PH[1], 1)) * DB,
                          mod_sy=(ma(m, MOD_SY[0], 1) - ma(m, MOD_SY[1], 1)) * DB,
                          gv=(m[:, ~sil].var(axis=1) * DB ** 2) if int((~sil).sum()) >= 2 else None)
    ok_ph = _window_ok(~sil, MOD_PH[1] // 2)
    ok_sy = _window_ok(~sil, MOD_SY[1] // 2)
    for gi, g in enumerate(GROUPS):
        fr_g = cls == gi
        fr_ph = ok_ph & fr_g
        for b, rows in enumerate(band_rows):
            for who in ("pred", "gt"):
                p = parts[who]
                if fr_g.any():
                    add(f"fcon_fine:{g}:{who}:{b}", p["fcon_fine"][rows][:, fr_g] ** 2)
                    add(f"fcon_coarse:{g}:{who}:{b}", p["fcon_coarse"][rows][:, fr_g] ** 2)
                if fr_ph.any():
                    add(f"mod_ph:{g}:{who}:{b}", p["mod_ph"][rows][:, fr_ph] ** 2)
    for b, rows in enumerate(band_rows):
        for who in ("pred", "gt"):
            p = parts[who]
            if ok_sy.any():
                add(f"mod_sy:{who}:{b}", p["mod_sy"][rows][:, ok_sy] ** 2)
            if p["gv"] is not None:
                add(f"gv:{who}:{b}", p["gv"][rows])

    # 子音の種類別・子音ごと（音素で分ける）: 誤差、周波数方向のメリハリ、スペクトルの重心のずれ。
    # 重心 = mel の振幅（exp(ln-mel)）で重みを付けた mel フィルタの中心周波数の平均。子音ごとの粗いメリハリは
    # 973 Hz 以上（帯域 1〜3）をまとめる
    hz = mel_hz(gt.shape[0]) if mel_centers is None else np.asarray(mel_centers, dtype=np.float64)

    def centroid(m):
        w = np.exp(m - m.max(axis=0, keepdims=True))                         # フレームごとに最大で割る（桁あふれ防止）
        return (w * hz[:, None]).sum(0) / w.sum(0)

    cent_d = centroid(pred) - centroid(gt)                                   # [T] Hz
    hi_rows = slice(nb, gt.shape[0])
    for k, ph in enumerate(names):
        s, e = int(fb[k]), int(fb[k + 1])
        if e <= s or ph == "pau" or ph in lang.vowels:
            continue
        c = lang.consonant_class(ph)
        add(f"ph_cls:{c}:{ph}", np.ones(1))                                  # 子音ごとの表の種類（phoneme_rows）
        add(f"cerr:{c}", err[s:e])
        add(f"cent:{c}", cent_d[s:e])
        add(f"ph_err:{ph}", err[s:e])
        add(f"ph_cent:{ph}", cent_d[s:e])
        for who in ("pred", "gt"):
            p = parts[who]
            add(f"ph_fcon:{ph}:{who}", p["fcon_coarse"][hi_rows, s:e] ** 2)
            for b, rows in enumerate(band_rows):
                add(f"fcon_fine:{c}:{who}:{b}", p["fcon_fine"][rows, s:e] ** 2)
                add(f"fcon_coarse:{c}:{who}:{b}", p["fcon_coarse"][rows, s:e] ** 2)

    if uv is not None and f0_logf0 is not None:                              # 倍音の構造（正解の F0 で有声、無音を除く）
        _harmonic_stats(pred, gt, f0_logf0, (~sil) & (np.asarray(uv, dtype=np.float32)[:T] >= 0.5), hz, add)

    # 模様（規則的なパターン）: 帯域ごとに PAT_FRAMES フレームのタイル（無音を含まないもの、PAT_HOP ずつずらす）の
    # 2 次元パワースペクトル（タイルの平均を引いて 2 次元のハン窓）を生成・正解それぞれ足し合わせる（配列のまま。
    # pattern_db・図が使う。[合計の配列, タイル数]）
    windows = {}
    for t0 in range(0, T - PAT_FRAMES + 1, PAT_HOP):
        if sil[t0:t0 + PAT_FRAMES].any():
            continue
        for b, rows in enumerate(band_rows):
            for who, m in (("pred", pred), ("gt", gt)):
                tile = np.asarray(m[rows, t0:t0 + PAT_FRAMES], dtype=np.float64)
                n_r = tile.shape[0]
                if n_r not in windows:
                    windows[n_r] = np.outer(np.hanning(n_r), np.hanning(PAT_FRAMES))
                a = acc[f"pat:{who}:{b}"]
                a[0] = a[0] + np.abs(np.fft.rfft2((tile - tile.mean()) * windows[n_r])) ** 2
                a[1] += 1
    return dict(acc)


def pattern_db(stats_list, b: int):
    """帯域 b の、生成/正解 の 2 次元パワースペクトルの比（dB）[bin 方向の周波数, 時間方向の周波数]。bin 方向は
    fftshift 済み（中央の行 = 0）。規則的な模様があると、正解に無い間隔の成分が周りより突出する。タイルが無ければ None。"""
    sp = [st[f"pat:pred:{b}"][0] for st in stats_list if f"pat:pred:{b}" in st]
    sg = [st[f"pat:gt:{b}"][0] for st in stats_list if f"pat:gt:{b}" in st]
    if not sp:
        return None
    r = 10.0 * np.log10(np.maximum(np.sum(sp, axis=0), 1e-30) / np.maximum(np.sum(sg, axis=0), 1e-30))
    return np.fft.fftshift(r, axes=0)


def pattern_peak(db) -> float:
    """模様の突出（dB）: 比（dB）から周りの PAT_LOCAL × PAT_LOCAL の中央値を引いた分の最大。規則的な模様は 2 次元の
    スペクトルの数点だけに出るので最大で見る（実データはタイルが数千あるので、揺れによる最大は小さい）。なだらかな
    減り方（平均化でのっぺりする）は周りと一緒に下がるので数えない。中心付近の低い成分（タイルの中の傾き）は除く。"""
    if db is None:
        return float("nan")
    from scipy.ndimage import median_filter
    excess = db - median_filter(db, size=PAT_LOCAL, mode="nearest")
    c = db.shape[0] // 2
    mask = np.ones(db.shape, dtype=bool)
    mask[c - PAT_EXCLUDE:c + PAT_EXCLUDE + 1, :PAT_EXCLUDE + 1] = False
    return float(excess[mask].max())


# ── 集計と比較 ────────────────────────────────────────────────────────────────
# ck = {step: [句ごとの phrase_stats]}（1 つの学習ぶん）。句の並びは全モデル・全 step で同じ。

def _phrase_sums(stats_list, keys, j=0) -> np.ndarray:
    """句ごとに keys の値を足した配列 [n]（j=0: 合計、j=1: 個数）。"""
    return np.array([sum(st[k][j] for k in keys if k in st) for st in stats_list], dtype=np.float64)


def _div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _wsum(W, x):
    """W [B, n] で重み付けした x [n] の和 → [B]。W @ x は macOS の Accelerate で根拠のない
    浮動小数点の警告を出すので使わない。"""
    return (W * x).sum(axis=1)


def metric_fn(stats_list, metric):
    """指標を「句の重み W [B, n]（各句を何回数えるか）→ 値 [B]」の関数にする。W を全部 1 にすれば
    全句そのまま、復元抽出で選ばれた回数にすればブートストラップの 1 回分。
    kind: mean（平均）/ signed_mean（符号付きの平均、0 に近いほど良い）/ ratio（生成/正解）/
    rms_ratio（二乗和の比の平方根 = RMS の比）/ contrast（平均(keys[0]) − 平均(keys[1])）"""
    kind, keys = metric["kind"], metric["keys"]
    if kind in ("mean", "signed_mean"):                      # フレーム（・bin）数で重み付けした平均
        s, c = _phrase_sums(stats_list, keys), _phrase_sums(stats_list, keys, 1)
        return lambda W: _div(_wsum(W, s), _wsum(W, c))
    if kind in ("ratio", "rms_ratio"):                       # 生成/正解（合計同士）
        sp, sg = _phrase_sums(stats_list, [keys[0]]), _phrase_sums(stats_list, [keys[1]])
        if kind == "ratio":
            return lambda W: _div(_wsum(W, sp), _wsum(W, sg))
        return lambda W: np.sqrt(_div(_wsum(W, sp), _wsum(W, sg)))
    if kind == "contrast":
        fa = metric_fn(stats_list, dict(kind="mean", keys=keys[0]))
        fb = metric_fn(stats_list, dict(kind="mean", keys=keys[1]))
        return lambda W: fa(W) - fb(W)
    raise ValueError(f"unknown metric kind: {kind}")


def metric_value(stats_list, metric) -> float:
    return float(metric_fn(stats_list, metric)(np.ones((1, len(stats_list))))[0])


def score(v, metric):
    """比べるときの値（小さいほど良い）。比は |比 − 1|、符号付きのずれは |ずれ|、それ以外はそのまま。"""
    kind = metric["kind"]
    if kind in ("ratio", "rms_ratio"):
        return np.abs(v - 1.0)
    if kind == "signed_mean":
        return np.abs(v)
    return v


def median_over_ckpts(ck: dict, metric) -> float:
    """チェックポイントごとの値の中央値（たまたま悪い 1 つに引きずられない）。"""
    v = np.array([metric_value(ck[s], metric) for s in sorted(ck)])
    return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan")


def ckpt_pairs(ck_model: dict, ck_base: dict) -> list:
    """比べるチェックポイントの組 [(モデルの step, 基準の step)]。同じ step があればそれ同士。重ならない
    （例: 生徒の 16k〜20k と基準の 31k〜35k）ときは、並び順で後ろ（学習の進んだ方）からそろえて組にする。"""
    steps = sorted(set(ck_model) & set(ck_base))
    if steps:
        return [(s, s) for s in steps]
    sm, sb = sorted(ck_model), sorted(ck_base)
    n = min(len(sm), len(sb))
    return list(zip(sm[len(sm) - n:], sb[len(sb) - n:])) if n else []


def compare(ck_model: dict, ck_base: dict, metric, n_boot: int = 1000, seed: int = 0) -> dict:
    """同じ step 同士（重ならなければ並び順の組。ckpt_pairs）の差（モデル − 基準、score の差）をチェックポイント
    ごとに出し、その中央値に句のブートストラップ 95% 区間を付ける。負 = モデルの方が良い。"""
    pairs = ckpt_pairs(ck_model, ck_base)
    if not pairs:                                                        # 比べられるチェックポイントが無い
        return dict(steps=[], base_steps=[], per_step=[], point=float("nan"), lo=float("nan"), hi=float("nan"))
    n = len(ck_base[pairs[0][1]])
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones((1, n)), rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)])
    D = np.stack([score(metric_fn(ck_model[a], metric)(W), metric) - score(metric_fn(ck_base[b], metric)(W), metric)
                  for a, b in pairs], axis=1)                            # [1 + n_boot, 組]。行 0 = 全句
    out = dict(steps=[a for a, _ in pairs], base_steps=[b for _, b in pairs], per_step=[float(d) for d in D[0]],
               point=float("nan"), lo=float("nan"), hi=float("nan"))
    ok = np.isfinite(D).all(axis=1)
    if ok[0] and ok[1:].any():
        med = np.median(D[ok], axis=1)                                   # 行ごとに step 間の中央値
        out["point"] = float(np.median(D[0]))
        out["lo"], out["hi"] = (float(x) for x in np.percentile(med[1:], [2.5, 97.5]))
    return out


def judge(c: dict) -> str:
    """改善 / 悪化 は「区間が 0 をまたがない」かつ「全チェックポイントで差の向きが同じ」とき。"""
    d = np.asarray(c["per_step"])
    if not np.isfinite(c["point"]):
        return "データなし"
    if c["hi"] < 0 and np.all(d < 0):
        return "改善"
    if c["lo"] > 0 and np.all(d > 0):
        return "悪化"
    if c["hi"] < 0 or c["lo"] > 0:
        return "はっきりしない（チェックポイントで向きが変わる）"
    return "差なし（誤差の範囲）"


def build_metrics(band_names=None):
    band_names = list(band_names or band_labels())
    m = [dict(sec="all", key="all", ja="無音以外の全フレームの誤差", kind="mean", keys=["all"])]
    for g in GROUPS:                                         # 位置情報が効いたかの決め手
        m.append(dict(sec="position", key=f"edge_mid:{g}", ja=f"{GROUP_JA[g]}: 端 − 中ほど（位置情報の決め手）",
                      kind="contrast", keys=([f"onset:{g}:0", f"offset:{g}:0"], [f"pos:{g}:4", f"pos:{g}:5"])))
    for g in GROUPS:
        m.append(dict(sec="position", key=f"onset0:{g}", ja=f"{GROUP_JA[g]}の頭 0–20 ms", kind="mean",
                      keys=[f"onset:{g}:0"]))
        m.append(dict(sec="position", key=f"offset0:{g}", ja=f"{GROUP_JA[g]}の終わり 0–20 ms", kind="mean",
                      keys=[f"offset:{g}:0"]))
        m.append(dict(sec="position", key=f"mid:{g}", ja=f"{GROUP_JA[g]}の中ほど（相対位置 0.4–0.6）", kind="mean",
                      keys=[f"pos:{g}:4", f"pos:{g}:5"]))
    m.append(dict(sec="position", key="len_long", ja="300 ms 以上の長い音素", kind="mean", keys=["len:3"]))
    m.append(dict(sec="position", key="sharp", ja="境界のつなぎの鋭さ（生成/正解、1 に近いほど良い）", kind="ratio",
                  keys=["sharp:pred", "sharp:gt"]))
    m.append(dict(sec="position", key="steady", ja="長い母音の中の揺れ（生成/正解、1 に近いほど良い）", kind="ratio",
                  keys=["steady:pred", "steady:gt"]))
    for b, nm in enumerate(band_names):
        m.append(dict(sec="band", key=f"band_tex:{b}", ja=f"{nm} の微細テクスチャ（生成/正解、1 に近いほど良い）",
                      kind="rms_ratio", keys=[f"band_tex:pred:{b}", f"band_tex:gt:{b}"]))
    for b, nm in enumerate(band_names):
        m.append(dict(sec="band", key=f"band_err:{b}", ja=f"{nm} の誤差", kind="mean", keys=[f"band_err:{b}"]))
    for b, nm in enumerate(band_names):
        m.append(dict(sec="band", key=f"band_bias:{b}", ja=f"{nm} のレベルのずれ（生成 − 正解、0 に近いほど良い）",
                      kind="signed_mean", keys=[f"band_bias:{b}"]))
    m.append(dict(sec="unvoiced", key="uv_all", ja="無声の区間の誤差", kind="mean", keys=["uv_all"]))
    for b, nm in enumerate(band_names):
        m.append(dict(sec="unvoiced", key=f"uv_band_bias:{b}",
                      ja=f"無声の区間 {nm} のレベルのずれ（生成 − 正解、0 に近いほど良い）",
                      kind="signed_mean", keys=[f"uv_band_bias:{b}"]))
    for b, nm in enumerate(band_names):
        m.append(dict(sec="unvoiced", key=f"uv_band_tex:{b}",
                      ja=f"無声の区間 {nm} の微細テクスチャ（生成/正解、1 に近いほど良い）",
                      kind="rms_ratio", keys=[f"uv_band_tex:pred:{b}", f"uv_band_tex:gt:{b}"]))
    averaging = [("fcon_fine", "周波数方向の細かいメリハリ（母音では倍音の山と谷）", True),
                 ("fcon_coarse", "スペクトルの形の山と谷（粗いメリハリ。母音ではフォルマント）", True),
                 ("mod_ph", "時間方向 約 30〜150 ms の動き（音素の速さ）", True),
                 ("mod_sy", "時間方向 約 150〜600 ms の動き（音節・音符の速さ）", False),
                 ("gv", "動く幅全体（GV）", False)]
    for key, ja, by_group in averaging:
        for g in (GROUPS if by_group else (None,)):
            for b, nm in enumerate(band_names):
                k = f"{key}:{g}" if g else key
                who = f"{GROUP_JA[g]} " if g else ""
                m.append(dict(sec="averaging", key=f"{k}:{b}", ja=f"{who}{nm} の{ja}",
                              kind="rms_ratio", keys=[f"{k}:pred:{b}", f"{k}:gt:{b}"]))
    for c in CLASSES:                                        # 「子音など」の内訳
        m.append(dict(sec="consonants", key=f"cerr:{c}", ja=f"{CLASS_JA[c]}の誤差", kind="mean", keys=[f"cerr:{c}"]))
        m.append(dict(sec="consonants", key=f"cent:{c}", ja=f"{CLASS_JA[c]}のスペクトルの重心のずれ（生成 − 正解、"
                      "0 に近いほど良い）", kind="signed_mean", keys=[f"cent:{c}"], unit="Hz"))
        for key, ja in (("fcon_coarse", "スペクトルの形の山と谷（粗いメリハリ）"), ("fcon_fine", "細かいメリハリ")):
            for b, nm in enumerate(band_names):
                m.append(dict(sec="consonants", key=f"{key}:{c}:{b}", ja=f"{CLASS_JA[c]} {nm} の{ja}",
                              kind="rms_ratio", keys=[f"{key}:{c}:pred:{b}", f"{key}:{c}:gt:{b}"]))
    for g in HARM_RANGES:                                    # 倍音の構造
        r = HARM_RANGE_JA[g]
        m.append(dict(sec="harmonics", key=f"harm_depth:{g}", ja=f"{r}: 倍音の山と谷の深さ（生成/正解、1 未満 = 浅い）",
                      kind="ratio", keys=[f"harm_depth:{g}:pred", f"harm_depth:{g}:gt"]))
        m.append(dict(sec="harmonics", key=f"harm_peak_bias:{g}", ja=f"{r}: 倍音の山のレベルのずれ（生成 − 正解）",
                      kind="signed_mean", keys=[f"harm_peak_bias:{g}"]))
        m.append(dict(sec="harmonics", key=f"harm_valley_bias:{g}",
                      ja=f"{r}: 倍音の谷のレベルのずれ（生成 − 正解、+ = 谷が埋まっている）",
                      kind="signed_mean", keys=[f"harm_valley_bias:{g}"]))
        for part, ja in (("peak", "山"), ("valley", "谷")):
            m.append(dict(sec="harmonics", key=f"harm_grain_{part}:{g}",
                          ja=f"{r}: 倍音の{ja}のざらつき（フレームごとのギザギザ、生成/正解、1 を超えるとざらつく）",
                          kind="ratio", keys=[f"harm_grain_{part}:{g}:pred", f"harm_grain_{part}:{g}:gt"]))
    return m


def phoneme_rows(ck_all: dict, names) -> list:
    """子音ごとの表の行（基準の最初のチェックポイントで、全句のフレーム数が PH_MIN_FRAMES 以上の子音）。
    行 = {ph, cls, frames, err, fcon, cent}。err・fcon・cent は {モデル名: チェックポイント間の中央値}。
    fcon = 973 Hz 以上（帯域 1〜3）のスペクトルの形の山と谷の 生成/正解 比、cent = 重心のずれ（Hz）。"""
    base = names[0]
    frames = defaultdict(int)
    cls = {}                                                 # 子音 → 種類（phrase_stats が言語パックで決めたもの）
    for st in ck_all[base][sorted(ck_all[base])[0]]:
        for k, (_s, n) in st.items():
            if k.startswith("ph_err:"):
                frames[k[len("ph_err:"):]] += n
            elif k.startswith("ph_cls:"):
                c, p = k[len("ph_cls:"):].split(":", 1)
                cls[p] = c
    order = {c: i for i, c in enumerate(CLASSES)}
    phs = sorted((p for p in frames if frames[p] >= PH_MIN_FRAMES),
                 key=lambda p: (order[cls.get(p, "other")], -frames[p]))
    rows = []
    for p in phs:
        r = dict(ph=p, cls=cls.get(p, "other"), frames=int(frames[p]))
        for col, mt in (("err", dict(kind="mean", keys=[f"ph_err:{p}"])),
                        ("fcon", dict(kind="rms_ratio", keys=[f"ph_fcon:{p}:pred", f"ph_fcon:{p}:gt"])),
                        ("cent", dict(kind="signed_mean", keys=[f"ph_cent:{p}"]))):
            r[col] = {nm: median_over_ckpts(ck_all[nm], mt) for nm in names}
        rows.append(r)
    return rows


# ── 図 ────────────────────────────────────────────────────────────────────────

def plot_curves(ck_all: dict, names, out_dir, band_names=None):
    """ck_all = {モデル名: {step: [句ごとの集計]}}。曲線・棒はチェックポイント間の中央値。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    band_names = list(band_names or band_labels())

    def med(nm, keys, kind="mean"):
        return median_over_ckpts(ck_all[nm], dict(kind=kind, keys=keys))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = (np.arange(POS_BINS) + 0.5) / POS_BINS
    for ax, g in zip(axes, GROUPS):
        for nm in names:
            ax.plot(x, [med(nm, [f"pos:{g}:{b}"]) for b in range(POS_BINS)], "o-", label=nm)
        ax.set_title(f"{GROUP_EN[g]}: error vs relative position in the phoneme")
        ax.set_xlabel("relative position in the phoneme (0 = start, 1 = end)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mel error [dB] (median over checkpoints)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_position.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    nb = len(DIST_EDGES_MS) - 1
    w = 0.8 / max(1, len(names))
    for ax, side, title in zip(axes, ("onset", "offset"), ("distance from the phoneme start",
                                                            "distance to the phoneme end")):
        for gi, g in enumerate(GROUPS):
            for ni, nm in enumerate(names):
                y = [med(nm, [f"{side}:{g}:{j}"]) for j in range(nb)]
                xs = np.arange(nb) + gi * (nb + 1) + (ni - (len(names) - 1) / 2) * w
                ax.bar(xs, y, width=w, label=nm if gi == 0 else None, color=f"C{ni}")
        ticks = [j + gi * (nb + 1) for gi in range(len(GROUPS)) for j in range(nb)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{GROUP_SHORT[g]}\n{_bin_label(DIST_EDGES_MS, j)}" for g in GROUPS for j in range(nb)],
                           fontsize=7)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("mel error [dB] (median over checkpoints)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_boundary_distance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    nl = len(LEN_EDGES_MS) - 1
    for ni, nm in enumerate(names):
        y = [med(nm, [f"len:{j}"]) for j in range(nl)]
        ax.bar(np.arange(nl) + (ni - (len(names) - 1) / 2) * w, y, width=w, label=nm, color=f"C{ni}")
    ax.set_xticks(np.arange(nl))
    ax.set_xticklabels([_bin_label(LEN_EDGES_MS, j) for j in range(nl)])
    ax.set_xlabel("phoneme length")
    ax.set_ylabel("mel error [dB] (median over checkpoints)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_phoneme_length.png", dpi=120)
    plt.close(fig)

    def band_figure(panels, fname, ncols=None):
        ncols = ncols or len(panels)
        nrows = -(-len(panels) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax in axes[len(panels):]:
            ax.axis("off")
        xb = np.arange(BANDS)
        for ax, (title, kind, key, unit) in zip(axes, panels):
            for ni, nm in enumerate(names):
                keys = ((lambda b: [f"{key}:pred:{b}", f"{key}:gt:{b}"]) if kind == "rms_ratio"
                        else (lambda b: [f"{key}:{b}"]))
                y = [med(nm, keys(b), kind) for b in range(BANDS)]
                ax.bar(xb + (ni - (len(names) - 1) / 2) * w, y, width=w, label=nm, color=f"C{ni}")
            if kind == "rms_ratio":
                ax.axhline(1.0, color="k", lw=0.8, ls="--")
                ax.set_ylim(0.0, max(1.35, ax.get_ylim()[1]))        # 凡例を 1.0 の線の上に置く余白
            if kind == "signed_mean":
                ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xticks(xb)
            ax.set_xticklabels(band_names, fontsize=8)
            ax.set_title(title)
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    band_figure((("fine texture, generated / GT (1 = same as GT)", "rms_ratio", "band_tex", "ratio"),
                 ("error", "mean", "band_err", "dB"),
                 ("level bias, generated - GT", "signed_mean", "band_bias", "dB")), "band_metrics.png")
    band_figure((("unvoiced frames: fine texture, generated / GT", "rms_ratio", "uv_band_tex", "ratio"),
                 ("unvoiced frames: level bias, generated - GT", "signed_mean", "uv_band_bias", "dB")),
                "unvoiced_band_metrics.png")
    band_figure(tuple((f"{GROUP_EN[g]}: {what}", "rms_ratio", f"{key}:{g}", "generated / GT")
                      for key, what in (("fcon_fine", "spectral fine contrast (<5 bins)"),
                                        ("fcon_coarse", "spectral shape contrast (5-21 bins)"),
                                        ("mod_ph", "change over ~30-150 ms (phonemes)"))
                      for g in GROUPS)
                + (("change over ~150-600 ms (syllables, notes)", "rms_ratio", "mod_sy", "generated / GT"),
                   ("global variance (GV)", "rms_ratio", "gv", "generated / GT")),
                "averaging_metrics.png", ncols=4)
    band_figure(tuple((f"{CLASS_EN[c]}: spectral shape contrast (5-21 bins)", "rms_ratio", f"fcon_coarse:{c}",
                       "generated / GT") for c in CLASSES)
                + tuple((f"{CLASS_EN[c]}: spectral fine contrast (<5 bins)", "rms_ratio", f"fcon_fine:{c}",
                         "generated / GT") for c in CLASSES),
                "consonant_classes.png", ncols=4)

    rows = phoneme_rows(ck_all, names)                      # 子音ごと（x = 子音、色 = モデル）
    if rows:
        x = np.arange(len(rows))
        fig, axes = plt.subplots(3, 1, figsize=(min(30.0, max(10.0, 0.45 * len(rows) * len(names))), 11),
                                 sharex=True)
        for ax, (col, title, unit) in zip(axes, (("err", "error per consonant", "dB"),
                                                ("fcon", "spectral shape contrast >= 973 Hz, generated / GT", "ratio"),
                                                ("cent", "spectral centroid shift, generated - GT", "Hz"))):
            for ni, nm in enumerate(names):
                ax.bar(x + (ni - (len(names) - 1) / 2) * w, [r[col][nm] for r in rows], width=w, label=nm,
                       color=f"C{ni}")
            if col == "fcon":
                ax.axhline(1.0, color="k", lw=0.8, ls="--")
            if col == "cent":
                ax.axhline(0.0, color="k", lw=0.8)
            ax.set_title(title)
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3, axis="y")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{r['ph']}\n{r['cls']}" for r in rows], fontsize=8)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "consonant_phonemes.png", dpi=110)
        plt.close(fig)

    # 倍音の構造: 深さの比・レベルのずれ・ざらつきの比（x = 指標、色 = モデル）
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    panels = (("harmonic peak-valley depth, generated / GT", "ratio",
               [("depth <1k", "harm_depth:lo"), ("depth 1-3k", "harm_depth:mid")]),
              ("harmonic level bias, generated - GT [dB]", "signed_mean",
               [("peak <1k", "harm_peak_bias:lo"), ("peak 1-3k", "harm_peak_bias:mid"),
                ("valley <1k", "harm_valley_bias:lo"), ("valley 1-3k", "harm_valley_bias:mid")]),
              ("frame-to-frame roughness, generated / GT", "ratio",
               [("peak <1k", "harm_grain_peak:lo"), ("peak 1-3k", "harm_grain_peak:mid"),
                ("valley <1k", "harm_grain_valley:lo"), ("valley 1-3k", "harm_grain_valley:mid")]))
    for ax, (title, kind, items) in zip(axes, panels):
        x = np.arange(len(items))
        for ni, nm in enumerate(names):
            keys = ((lambda key: [f"{key}:pred", f"{key}:gt"]) if kind == "ratio" else (lambda key: [key]))
            ax.bar(x + (ni - (len(names) - 1) / 2) * w, [med(nm, keys(key), kind) for _, key in items], width=w,
                   label=nm, color=f"C{ni}")
        ax.axhline(1.0 if kind == "ratio" else 0.0, color="k", lw=0.8, ls="--" if kind == "ratio" else "-")
        if kind == "ratio":
            ax.set_ylim(0.0, max(1.35, ax.get_ylim()[1]))            # 凡例を 1.0 の線の上に置く余白
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in items], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "harmonics.png", dpi=120)
    plt.close(fig)

    # 模様: 2 次元パワースペクトルの 生成/正解（dB）。行 = モデル、列 = 帯域、全チェックポイントの合計
    fig, axes = plt.subplots(len(names), BANDS, figsize=(3.4 * BANDS, 2.8 * len(names)), squeeze=False)
    im = None
    for i, nm in enumerate(names):
        allst = [st for s in sorted(ck_all[nm]) for st in ck_all[nm][s]]
        for b in range(BANDS):
            ax = axes[i, b]
            db = pattern_db(allst, b)
            if db is None:
                ax.axis("off")
                continue
            n = db.shape[0]
            im = ax.imshow(db, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-6.0, vmax=6.0,
                           extent=(-0.5, db.shape[1] - 0.5, -(n // 2) - 0.5, n - n // 2 - 0.5))
            kt = [k for k in (2, 4, 8, 16, 32) if k < db.shape[1]]
            ax.set_xticks(kt)
            ax.set_xticklabels([f"{PAT_FRAMES // k}" for k in kt], fontsize=7)
            kf = [k for k in (-8, -4, 0, 4, 8) if abs(k) <= n // 2]
            ax.set_yticks(kf)
            ax.set_yticklabels(["-" if k == 0 else f"{n // abs(k)}" for k in kf], fontsize=7)
            ax.set_title(f"{nm} / {band_names[b]}", fontsize=8)
            if i == len(names) - 1:
                ax.set_xlabel("period along time [frames]", fontsize=8)
            if b == 0:
                ax.set_ylabel("period along mel bins", fontsize=8)
    if im is not None:
        fig.subplots_adjust(right=0.9, hspace=0.45, wspace=0.25)
        fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.012, 0.7]), label="generated / GT [dB]")
        fig.savefig(out_dir / "pattern_spectrum.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for nm in names:
        st = sorted(ck_all[nm])
        ax.plot(st, [metric_value(ck_all[nm][s], ALL) for s in st], "o-", label=nm)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mel error [dB] (all non-pau frames)")
    ax.set_title("overall error per checkpoint (how much training noise there is)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_step.png", dpi=120)
    plt.close(fig)


# ── レポート ──────────────────────────────────────────────────────────────────

def _fmt(v: float, kind: str, unit: str = "dB") -> str:
    if kind in ("ratio", "rms_ratio"):
        return f"{v:.3f}"
    if kind == "signed_mean":
        return f"{v:+.0f} {unit}" if unit == "Hz" else f"{v:+.2f} {unit}"
    return f"{v:.3f} {unit}"


def _num(v: float, unit: str) -> str:
    """比較の表の数値（Hz は整数、それ以外は小数 3 桁）。"""
    return f"{v:+.0f}" if unit == "Hz" else f"{v:+.3f}"


def build_report(ck_all: dict, names, n_items: int, n_boot: int = 1000, band_names=None):
    """ck_all = {モデル名: {step: [句ごとの集計]}}、names[0] が基準 → (report.md の文字列, summary)。"""
    base = names[0]
    band_names = list(band_names or band_labels())
    metrics = build_metrics(band_names)
    all_steps = sorted({s for nm in names for s in ck_all[nm]})
    n_steps = min(len(ck_all[nm]) for nm in names)
    lines = ["# A/B 評価", "",
             f"- 評価用の句: {n_items}（学習に使っていない曲。正解の音素長と F0 を与えて合成）",
             f"- チェックポイント: {all_steps}　基準: **{base}**",
             "- 誤差 = mel の各 bin の |生成 − 正解| の平均（dB、無音 pau を除く）。小さいほど良い",
             "- 各指標はチェックポイントごとに計算し、チェックポイント間の中央値で代表させる"
             "（たまたま悪いチェックポイントが 1 つあっても引きずられない）",
             "- 比較は同じ step 同士の差（モデル − 基準）をチェックポイントごとに出し、その中央値で見る。判定は",
             "  - **改善 / 悪化**: 差の中央値の 95% 区間（句のブートストラップ）が 0 をまたがず、"
             "しかも全部のチェックポイントで差の向きが同じ",
             "  - **はっきりしない**: 区間は 0 をまたがないが、チェックポイントによって向きが変わる（学習の揺れの範囲）",
             "  - **差なし（誤差の範囲）**: 区間が 0 をまたぐ",
             "- **端 − 中ほど**（位置情報の決め手）: 音素の端（頭・終わり 0–20 ms）の誤差から中ほど（相対位置 0.4–0.6）の"
             "誤差を引いた値。位置情報が効けば、端が中ほどより多く良くなって、この値が基準より小さく（差が負に）なる。"
             "モデル全体が一律に良い・悪いズレには左右されにくい",
             f"- **帯域**: mel の bin を 4 等分（{' / '.join(band_names)}）。**微細テクスチャ** = mel から時間方向 "
             f"{TEX_WIN} フレームの移動平均を引いた残り（細かい揺らぎ）の RMS の 生成/正解 比。1 未満はのっぺり、"
             "1 を超えるとざらつき過多。擬似 mel のノイズ量などの A/B ではこれが主な指標。細部が正しく出ても、"
             "ランダムな細部は正解と一致しないので、誤差はむしろ増えることがある",
             "- **無声の区間**: 正解の F0 で無声と判定されたフレーム（無音 pau を除く）だけで見た帯域の指標。"
             "低域のレベルが正解より高い（ずれが正）なら、擬似 mel の倍音の消し残りで子音がにごっている可能性がある",
             "- **平均化**: 生成/正解 の RMS 比（1 未満 = 平均化されて縮んでいる）。周波数方向の細かいメリハリ（bin 方向 "
             f"{FREQ_FINE} 本の移動平均からのはみ出し。母音では倍音の山と谷）と粗いメリハリ（{FREQ_FINE} 本と "
             f"{FREQ_COARSE} 本の移動平均の差 = スペクトルの形の山と谷。母音ではフォルマント、無声子音では狭めの前の空間の"
             "共鳴でできる雑音の山）は母音・子音のフレーム別。時間方向の動きは 約 30〜150 ms（音素の速さ、母音・子音別）と"
             " 約 150〜600 ms（音節・音符の速さ）。GV は無音以外のフレームでの bin ごとの分散（動く幅全体）",
             "- **母音と子音の分け方**: 言語パック（`dict/<lang>.yaml`）の vowels が母音、vowels にも silence にも入らない"
             "音素が「子音など」",
             "- **子音の種類別**: 「子音など」を言語パックの devoiced（無声化した母音）・voiceless（無声子音）・voiced（有声"
             "子音）で分け、どれにも入らないもの（日本語の cl など）は「その他」。スペクトルの重心 = mel の振幅で重みを"
             "付けた mel フィルタの中心周波数の平均。重心のずれ（生成 − 正解）は雑音の山の位置（s は高く sh は低いなど、"
             "子音を聞き分ける手がかり）のずれの目安",
             "- **倍音の構造**: 有声のフレーム（正解の F0 で有声、無音を除く）で、正解の F0 から倍音（k·F0）と倍音の間の谷"
             "（(k+0.5)·F0）の位置を決めて mel を読む（bin の間隔が F0/3 以下で 3 kHz 以下の所だけ。1 kHz 未満と 1〜3 kHz に"
             "分ける）。深さ = 山 − 谷。ざらつき = 3 フレームでの 2 階差分の大きさ（ビブラートなどのゆっくりした動きは打ち消し、"
             "フレームごとのギザギザだけを見る）。谷だけが持ち上がっていれば波形のノイズのような埋まり方、山までざらついて"
             "いれば対数の mel で足したノイズの残り",
             "- 基準と step が重ならない run（例: 生徒の 16k〜20k と基準の 31k〜35k）は、チェックポイントを並び順（学習の"
             "進んだ方から）で組にして比べ、各チェックポイントの差を「モデルの step↔基準の step」で示す",
             "- 比の指標は 1 に近いほど良く、差は |比 − 1| の差（負 = 1 に近づいた）。レベルのずれは |ずれ| の差"]
    if n_steps < MIN_STEPS_ADVISED:
        lines.append(f"- 注意: チェックポイントが {n_steps} つしかなく、学習の揺れを判定に十分入れられていない。"
                     f"{MIN_STEPS_ADVISED} つ以上を推奨")
    lines += ["", "## チェックポイントごとの全体誤差（学習の揺れの目安、図 `error_vs_step.png`）", "",
              "| step | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    per_ckpt_all = {nm: {} for nm in names}
    for s in all_steps:
        row = []
        for nm in names:
            if s in ck_all[nm]:
                v = metric_value(ck_all[nm][s], ALL)
                per_ckpt_all[nm][str(s)] = v
                row.append(f"{v:.3f} dB")
            else:
                row.append("—")
        lines.append(f"| {s} | " + " | ".join(row) + " |")

    summary = {"items": n_items, "steps": all_steps, "base": base, "band_names": band_names,
               "per_ckpt_all": per_ckpt_all, "metrics": {}}
    lines += ["", "## 指標（チェックポイント間の中央値）"]
    for sec, title in SECTIONS:
        lines += ["", f"### {title}", "", "| 指標 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for mt in (x for x in metrics if x["sec"] == sec):
            vals = [median_over_ckpts(ck_all[nm], mt) for nm in names]
            lines.append(f"| {mt['ja']} | " + " | ".join(_fmt(v, mt["kind"], mt.get("unit", "dB")) for v in vals)
                         + " |")
            summary["metrics"][mt["key"]] = {
                "ja": mt["ja"], "kind": mt["kind"], "section": sec, "values": dict(zip(names, map(float, vals))),
                "per_ckpt": {nm: {str(s): metric_value(ck_all[nm][s], mt) for s in sorted(ck_all[nm])}
                             for nm in names}}

    rows = phoneme_rows(ck_all, names)                      # 子音ごと（中央値の表だけ。区間は付けない）
    summary["phonemes"] = rows
    if rows:
        lines += ["", f"## 子音ごと（チェックポイント間の中央値。全句で {PH_MIN_FRAMES} フレーム以上の子音。図 "
                  "`consonant_phonemes.png`）"]
        for col, title, kind, unit in (("err", "誤差", "mean", "dB"),
                                       ("fcon", "973 Hz 以上のスペクトルの形の山と谷（生成/正解、1 に近いほど良い）",
                                        "rms_ratio", ""),
                                       ("cent", "スペクトルの重心のずれ（生成 − 正解、0 に近いほど良い）",
                                        "signed_mean", "Hz")):
            lines += ["", f"### {title}", "", "| 子音 | 種類 | フレーム数 | " + " | ".join(names) + " |",
                      "|---|---|---|" + "---|" * len(names)]
            for r in rows:
                lines.append(f"| {r['ph']} | {CLASS_JA[r['cls']]} | {r['frames']} | "
                             + " | ".join(_fmt(r[col][nm], kind, unit) for nm in names) + " |")

    def _pat_median(ck, b):
        v = [x for x in (pattern_peak(pattern_db(ck[s], b)) for s in sorted(ck)) if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    pat = {nm: {bn: _pat_median(ck_all[nm], b) for b, bn in enumerate(band_names)} for nm in names}
    summary["pattern_peak"] = pat
    lines += ["", "## 模様（規則的なパターン。チェックポイント間の中央値。図 `pattern_spectrum.png`）", "",
              f"帯域ごとに mel を {PAT_FRAMES} フレーム（約 0.37 秒）× 32 bin のタイルに切り、2 次元のパワースペクトル（どの"
              "間隔のくり返しがどれだけあるか）を 生成/正解 の比（dB）で比べる。表の値は、その比が周りの成分より突出している"
              "分の最大（dB）。正解に無い間隔の規則的な模様があると大きくなる。平均化でなだらかに減るのは数えない。"
              "0 に近いほど良い（正解どうしなら 0。模様が無くてもデータの揺れの分は少し出る）", "",
              "| 帯域 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for bn in band_names:
        lines.append(f"| {bn} | " + " | ".join(f"{pat[nm][bn]:.2f} dB" for nm in names) + " |")

    for nm in names[1:]:
        lines += ["", f"## {nm} と {base} の比較"]
        summary[nm] = {}
        for sec, title in SECTIONS:
            lines += ["", f"### {title}", "",
                      "| 指標 | 差（中央値） | 95% 区間 | 各チェックポイントの差 | 良くなった数 | 判定 |",
                      "|---|---|---|---|---|---|"]
            for mt in (x for x in metrics if x["sec"] == sec):
                c = compare(ck_all[nm], ck_all[base], mt, n_boot=n_boot)
                j = judge(c)
                u = "" if mt["kind"] in ("ratio", "rms_ratio") else mt.get("unit", "dB")
                lab = mt["ja"] + {"ratio": "（|比 − 1| の差）", "rms_ratio": "（|比 − 1| の差）",
                                  "signed_mean": "（|ずれ| の差）"}.get(mt["kind"], "")
                per = ", ".join(f"{_k(a) if a == b else _k(a) + '↔' + _k(b)}: {_num(d, u)}"
                                for a, b, d in zip(c["steps"], c["base_steps"], c["per_step"]))
                better = f"{sum(d < 0 for d in c['per_step'])}/{len(c['steps'])}"
                ci = f"[{_num(c['lo'], u)}, {_num(c['hi'], u)}]" if np.isfinite(c["lo"]) else "—"
                lines.append(f"| {lab} | {_num(c['point'], u)}{' ' + u if u else ''} | {ci} | {per} | {better} | {j} |")
                summary[nm][mt["key"]] = dict(diff=c["point"], ci=[c["lo"], c["hi"]],
                                              per_step={str(s): d for s, d in zip(c["steps"], c["per_step"])},
                                              judge=j)
    lines += ["", "## 図（どれもチェックポイント間の中央値）", "",
              "- `band_metrics.png`: 帯域ごとの微細テクスチャの比・誤差・レベルのずれ",
              "- `unvoiced_band_metrics.png`: 無声の区間だけの、帯域ごとの微細テクスチャの比とレベルのずれ",
              "- `averaging_metrics.png`: 平均化の指標（周波数方向のメリハリ・時間方向の動き・GV）を帯域ごとに",
              "- `consonant_classes.png`: 子音の種類別の、スペクトルの形の山と谷・細かいメリハリ（帯域ごと）",
              "- `consonant_phonemes.png`: 子音ごとの誤差・973 Hz 以上のスペクトルの形の山と谷・重心のずれ",
              "- `harmonics.png`: 倍音の山と谷の深さ・レベルのずれ・ざらつき（1 kHz 未満と 1〜3 kHz）",
              "- `pattern_spectrum.png`: 模様の 2 次元パワースペクトルの 生成/正解（dB、全チェックポイントの合計）。横 = 時間方向の"
              "くり返しの間隔（フレーム）、縦 = bin 方向の間隔（正負は模様の傾きの向き）。赤い点や筋が正解に無い規則的な模様",
              "- `error_vs_position.png`: 音素の中の位置ごとの誤差。位置情報が効いていれば、両端（0 と 1 の近く）が"
              "中ほどより大きく下がって曲線の形が変わる。全体が上下に平行にずれるだけなら位置情報の効果ではない",
              "- `error_vs_boundary_distance.png`: 音素の頭から / 終わりまでの距離ごとの誤差",
              "- `error_vs_phoneme_length.png`: 音素の長さ別の誤差",
              "- `error_vs_step.png`: チェックポイントごとの全体誤差（学習の揺れ）"]
    return "\n".join(lines) + "\n", summary


# ── main ──────────────────────────────────────────────────────────────────────

def parse_run(spec: str):
    """--runs の 1 つ → (フォルダ, step 数 or None, チェックポイントの step の並び or None)。
    「フォルダ@16」ならその run だけ 16 step で生成する（--num_steps より優先。1 step 用のモデルと複数 step の
    教師を同じ表で比べるとき）。「フォルダ@1:16000,17000」なら使うチェックポイントも決める（--steps より優先）。"""
    steps = None
    head, sep, tail = spec.rpartition(":")
    if sep and head and tail and all(x.strip().isdigit() for x in tail.split(",")):
        spec, steps = head, [int(x) for x in tail.split(",")]
    path, sep, k = spec.rpartition("@")
    if sep and path and k.isdigit() and int(k) >= 1:
        return path, int(k), steps
    return spec, None, steps


def run_labels(name: str, ks) -> dict:
    """step 数ごとの表示名。ks が 1 つ（または None = ckpt の値）なら名前そのまま、複数なら「名前@k8」。"""
    if len(ks) == 1:
        return {ks[0]: name}
    return {k: f"{name}@k{k}" for k in ks}


def load_eval_items(cfg: dict, data_dirs, limit=None):
    dcfg = cfg.get("data", {}) or {}
    spk_map = dict(dcfg.get("spk_map") or {})
    style_map = dict(dcfg.get("style_map") or {})
    for d in data_dirs:                                       # train.py と同じレシピの補完
        dbname = os.path.basename(os.path.normpath(d))
        rp = ROOT / "configs" / "recipes" / f"{dbname}.yaml"
        if rp.exists():
            rr = yaml.safe_load(open(rp, encoding="utf-8"))
            spk_map.setdefault(dbname, int(rr.get("spk_id", 0)))
            style_map.setdefault(dbname, int(rr.get("style_id", 0)))
    ds = LeapSingerDataset(data_dirs, "eval", eval_songs=dcfg.get("eval_songs", 2),
                           eval_dbs=dcfg.get("eval_dbs"), min_sec=dcfg.get("min_sec", 0.3),
                           silence=dcfg.get("silence", True),
                           silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
                           spk_map=spk_map, style_map=style_map,
                           loudness_agc=dcfg.get("loudness_agc") or None,
                           max_sec=dcfg.get("max_sec") or None)
    n = len(ds) if not limit else min(limit, len(ds))
    return [ds[i] for i in range(n)], ds.frame_rate


def main():
    ap = argparse.ArgumentParser(description="学習した run どうしを評価用の句で比べる（A/B 評価）")
    ap.add_argument("--config", required=True, help="学習に使った config（評価 split の決め方を揃える）")
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="学習の出力フォルダ。最初が比較の基準。「フォルダ@16」でその run だけ step 数を、"
                         "「フォルダ@1:16000,17000」でチェックポイントも決める")
    ap.add_argument("--names", nargs="*", default=None, help="表示名（既定はフォルダ名）")
    ap.add_argument("--steps", nargs="+", type=int, default=None,
                    help=f"比べるチェックポイントの step（{MIN_STEPS_ADVISED} つ以上を推奨。--runs で決めた run は除く）")
    ap.add_argument("--out", default="log/eval_ab")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="励起ノイズの乱数（全モデル共通）")
    ap.add_argument("--limit", type=int, default=None, help="評価する句数の上限（動作確認用）")
    ap.add_argument("--threads", type=int, default=2, help="CPU で回すときの torch スレッド数")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--num_steps", nargs="*", type=int, default=None,
                    help="生成の step 数（既定はチェックポイントに記録された値）。複数渡すと、run ごとに "
                         "「名前@k8」のように step 数ごとのモデルとして並べて比べる（reflow の教師用）")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mel = cfg.get("mel", {}) or {}
    mel_range = (int(mel.get("n_mels", 128)), float(mel.get("fmin", 40.0)), float(mel.get("fmax", 16000.0)))
    band_names = band_labels(*mel_range)
    centers = mel_hz(*mel_range)
    items, fr = load_eval_items(cfg, args.data_dirs, args.limit)
    runs = [parse_run(r) for r in args.runs]          # [(フォルダ, その run だけの step 数, チェックポイント)]
    names = args.names or [Path(r).name + (f"@k{k}" if k else "") for r, k, _ in runs]
    if len(names) != len(runs):
        raise SystemExit("--names の数が --runs と違います")
    if args.steps is None and any(s is None for _, _, s in runs):
        raise SystemExit("--steps を指定するか、全部の run に「フォルダ@K:step,step」でチェックポイントを書いてください")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"評価用の句: {len(items)}  frame_rate={fr:.3f}  device={args.device}")

    ks = list(dict.fromkeys(args.num_steps)) if args.num_steps else [None]   # None = ckpt の infer_steps
    ck_all = {}                                               # {モデル名: {step: [句ごとの集計]}}
    order = []
    for (run, k_run, steps_run), nm in zip(runs, names):
        ks_run = [k_run] if k_run else ks
        want = sorted(set(steps_run or args.steps))
        ckpts = [(s, Path(run) / f"ckpt_{s:06d}.pt") for s in want]
        ckpts = [(s, p) for s, p in ckpts if p.exists()]
        missing = sorted(set(want) - {s for s, _ in ckpts})
        if missing:
            print(f"[{nm}] 見つからない step: {missing}")
        if not ckpts:
            print(f"[{nm}] 使えるチェックポイントが無いので飛ばします")
            continue
        labels = run_labels(nm, ks_run)
        for lab in labels.values():
            ck_all[lab] = {}
            order.append(lab)
        for step, path in ckpts:
            model, mcfg = load_acoustic(str(path), args.device)
            id2ph = {i: p for i, p in enumerate(mcfg.get("phonemes") or [])}
            lang = find_lang(mcfg.get("phonemes") or [])          # 母音と子音の分け方（dict/<lang>.yaml）
            if lang is None and step == ckpts[0][0]:
                print(f"[{nm}] 音素の一覧に合う言語パック（dict/<lang>.yaml）がありません。母音と子音で分ける指標と"
                      "子音の種類別は意味を持ちません（ほかの指標は出ます）", flush=True)
            for k in ks_run:
                steps_n = int(k) if k is not None else int(mcfg.get("infer_steps", 1))
                stats = []
                for it in items:
                    pred = infer_mel(model, it, num_steps=steps_n, device=args.device, seed=args.seed)
                    stats.append(phrase_stats(pred, it["target_mel"], it["ph_ids"], it["ph_durs"], id2ph, fr,
                                              uv=it.get("uv"), mel_centers=centers, f0_logf0=it["f0_logf0"],
                                              lang=lang))
                ck_all[labels[k]][step] = stats
                print(f"[{labels[k]}] step {step}（{steps_n} step で生成）: 無音以外の全フレームの誤差 "
                      f"{metric_value(stats, ALL):.3f} dB  (noise_ratio={mcfg.get('noise_ratio')}, "
                      f"harm_decay={mcfg.get('harm_decay')}, x0_sigma={mcfg.get('x0_sigma', 0.0)}, "
                      f"pos_cond={mcfg.get('pos_cond')})", flush=True)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    names = order
    if len(names) < 1:
        raise SystemExit("評価できたモデルがありません")
    report, summary = build_report(ck_all, names, len(items), n_boot=args.n_boot, band_names=band_names)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    plot_curves(ck_all, names, out_dir, band_names=band_names)
    print("\n" + report)
    print(f"wrote {out_dir}/report.md, summary.json, *.png")


if __name__ == "__main__":
    main()
