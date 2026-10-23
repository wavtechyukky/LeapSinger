"""tools/eval_ab.py の集計: チェックポイントごとの差・中央値・判定、端 − 中ほど、帯域のテクスチャ（モデル不要）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import eval_ab as ev  # noqa: E402
from leapsinger.phonemes import load_lang  # noqa: E402

STEPS = (18000, 19000, 20000)
JA = load_lang("ja")                                                 # 母音と子音の分け方（dict/ja.yaml）


def _phrase(edge, mid, n=10):
    """母音だけの句の集計（端 n フレーム×2・中ほど n フレーム×2、誤差は edge / mid dB）。"""
    return {"all": [2 * edge * n + 2 * mid * n, 4 * n],
            "onset:vowel:0": [edge * n, n], "offset:vowel:0": [edge * n, n],
            "pos:vowel:4": [mid * n, n], "pos:vowel:5": [mid * n, n],
            "sharp:pred": [0.9 * n, n], "sharp:gt": [1.0 * n, n]}


def _run(shift_edge=None, shift_mid=None, seed=0, n_phr=40):
    """{step: [句ごとの集計]}。shift_*: {step: dB} を全句に足す（基準の句の誤差は seed で決まる）。"""
    rng = np.random.default_rng(seed)
    base_edge = rng.uniform(5.0, 6.0, n_phr)
    base_mid = rng.uniform(4.0, 5.0, n_phr)
    shift_edge = shift_edge or {}
    shift_mid = shift_mid or {}
    return {s: [_phrase(e + shift_edge.get(s, 0.0), m + shift_mid.get(s, 0.0)) for e, m in zip(base_edge, base_mid)]
            for s in STEPS}


ALL = ev.build_metrics()[0]
EDGE_MID = next(m for m in ev.build_metrics() if m["key"] == "edge_mid:vowel")
SHARP = next(m for m in ev.build_metrics() if m["key"] == "sharp")


def test_metric_value_is_frame_weighted_mean():
    ck = _run()
    st = ck[STEPS[0]]
    s = sum(p["all"][0] for p in st)
    c = sum(p["all"][1] for p in st)
    assert abs(ev.metric_value(st, ALL) - s / c) < 1e-12
    assert abs(ev.metric_value(st, SHARP) - 0.9) < 1e-12


def test_identical_runs_have_zero_difference():
    c = ev.compare(_run(), _run(), ALL, n_boot=200)
    assert c["point"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0
    assert ev.judge(c) == "差なし（誤差の範囲）"


def test_one_bad_base_checkpoint_is_not_an_improvement():
    # 2026-10-17 の評価の形: 基準の 18000 だけ全体が 0.8 dB 悪く、残りは ±0.02 dB の揺れ
    bad = {18000: 0.8}
    base = _run(shift_edge=bad, shift_mid=bad)
    jitter = {19000: 0.02, 20000: -0.02}
    model = _run(shift_edge=jitter, shift_mid=jitter)
    c = ev.compare(model, base, ALL, n_boot=200)
    assert abs(c["point"] - (-0.02)) < 1e-9                    # 中央値は 18000 に引きずられない
    assert c["per_step"][0] < -0.7 and c["per_step"][1] > 0
    assert ev.judge(c) == "はっきりしない（チェックポイントで向きが変わる）"
    # 平均（旧版のやり方）なら -0.27 dB の「改善」に見えていた
    old = np.mean(c["per_step"])
    assert old < -0.25


def test_consistent_improvement_is_judged_improved():
    better = {s: -0.1 for s in STEPS}
    c = ev.compare(_run(shift_edge=better, shift_mid=better), _run(), ALL, n_boot=200)
    assert ev.judge(c) == "改善" and all(d < 0 for d in c["per_step"])
    worse = {s: +0.1 for s in STEPS}
    c = ev.compare(_run(shift_edge=worse, shift_mid=worse), _run(), ALL, n_boot=200)
    assert ev.judge(c) == "悪化"


def test_edge_minus_middle_ignores_uniform_shifts():
    uniform = {s: -0.3 for s in STEPS}
    c = ev.compare(_run(shift_edge=uniform, shift_mid=uniform), _run(), EDGE_MID, n_boot=200)
    assert abs(c["point"]) < 1e-9 and ev.judge(c) == "差なし（誤差の範囲）"
    edges_only = {s: -0.2 for s in STEPS}                      # 端（頭・終わり）だけ良くなった = 位置情報が効いた形
    c = ev.compare(_run(shift_edge=edges_only), _run(), EDGE_MID, n_boot=200)
    assert abs(c["point"] - (-0.2)) < 1e-9
    assert ev.judge(c) == "改善"


def test_report_builds_and_marks_few_checkpoints():
    ck = {"base": _run(), "model": _run(shift_edge={s: -0.1 for s in STEPS}, shift_mid={s: -0.1 for s in STEPS})}
    report, summary = ev.build_report(ck, ["base", "model"], n_items=40, n_boot=100)
    assert "## model と base の比較" in report and "改善" in report
    assert "5 つ以上を推奨" in report                          # 3 つしかないので注意が出る
    assert set(summary["metrics"]["all"]["per_ckpt"]["model"]) == {str(s) for s in STEPS}
    assert summary["model"]["all"]["judge"] == "改善"
    assert "### 帯域" in report and summary["model"]["band_tex:3"]["judge"] == "データなし"   # 帯域の集計が無い


def _band_phrases():
    """正解（ランダムな細部あり）と、のっぺりした生成・一律に大きい生成。句 = pau 20 / a 160 / pau 20 フレーム。"""
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(0)
    gt = rng.normal(-5.0, 1.0, (128, 200))
    smooth = uniform_filter1d(gt, 9, axis=1, mode="nearest")
    args = ([0, 1, 0], [20, 160, 20], {0: "pau", 1: "a"}, 172.27)
    return (ev.phrase_stats(gt.copy(), gt, *args), ev.phrase_stats(smooth, gt, *args),
            ev.phrase_stats(gt + 0.1, gt, *args))


def test_band_texture_error_and_bias():
    same, smooth, loud = _band_phrases()
    for b in range(ev.BANDS):
        tex = dict(kind="rms_ratio", keys=[f"band_tex:pred:{b}", f"band_tex:gt:{b}"])
        assert abs(ev.metric_value([same], tex) - 1.0) < 1e-9
        assert ev.metric_value([smooth], tex) < 0.6                  # のっぺり = 1 を大きく下回る
        assert abs(ev.metric_value([loud], tex) - 1.0) < 1e-9          # 一律のレベル差はテクスチャに出ない
        bias = dict(kind="signed_mean", keys=[f"band_bias:{b}"])
        assert abs(ev.metric_value([loud], bias) - 0.1 * ev.DB) < 1e-6
        assert ev.metric_value([same], dict(kind="mean", keys=[f"band_err:{b}"])) == 0.0
    # 無音と、移動平均の窓が無音にかかるフレームは数えない: (160 − 2·2) フレーム × 32 bin
    assert same["band_err:0"][1] == 156 * 32


def test_unvoiced_band_metrics():
    """無声の区間だけの指標: 無声フレームの低域にだけ余計なエネルギー（倍音の消し残り）があると、そこだけ正に出る。"""
    rng = np.random.default_rng(1)
    gt = rng.normal(-5.0, 1.0, (128, 200))
    uv = np.ones(200)
    uv[80:120] = 0.0                                                   # 40 フレームの無声区間
    pred = gt.copy()
    pred[:32, 80:120] += 0.2                                           # 無声区間の低域だけ +0.2 ln = +1.7 dB
    st = ev.phrase_stats(pred, gt, [0, 1, 0], [20, 160, 20], {0: "pau", 1: "a"}, 172.27, uv=uv)
    bias = lambda b: ev.metric_value([st], dict(kind="signed_mean", keys=[f"uv_band_bias:{b}"]))
    assert abs(bias(0) - 0.2 * ev.DB) < 1e-6 and abs(bias(3)) < 1e-12
    assert st["uv_all"][1] == 40 and st["uv_band_bias:0"][1] == 40 * 32
    assert st["uv_band_tex:gt:0"][1] == (40 - 4) * 32                  # 窓が無声区間に収まるフレームだけ
    assert "uv_all" not in ev.phrase_stats(pred, gt, [0, 1, 0], [20, 160, 20], {0: "pau", 1: "a"}, 172.27)
    keys = {m["key"] for m in ev.build_metrics() if m["sec"] == "unvoiced"}
    assert {"uv_all", "uv_band_bias:0", "uv_band_tex:3"} <= keys


def test_band_texture_comparison_prefers_ratio_near_one():
    same, smooth, loud = _band_phrases()
    ck_base = {s: [smooth] for s in STEPS}                           # 基準: のっぺり
    ck_model = {s: [same] for s in STEPS}                            # モデル: 正解と同じ細部
    tex = next(m for m in ev.build_metrics() if m["key"] == "band_tex:3")
    c = ev.compare(ck_model, ck_base, tex, n_boot=50)
    assert c["point"] < -0.4 and ev.judge(c) == "改善"
    bias = next(m for m in ev.build_metrics() if m["key"] == "band_bias:3")
    c = ev.compare({s: [loud] for s in STEPS}, ck_model, bias, n_boot=50)
    assert ev.judge(c) == "悪化"                                       # |ずれ| が増えた


def _avg_stats(pred_fn, seed=2):
    """平均化の指標用の句。正解 = 周波数方向の起伏 f(bin) + 時間方向の起伏 g(t)（それぞれ別の指標にだけ効く）。
    句 = pau 60 / a 150 / k 60 / a 150 / pau 60 フレーム。"""
    rng = np.random.default_rng(seed)
    gt = rng.normal(0.0, 1.0, 128)[:, None] + rng.normal(0.0, 1.0, 480)[None, :] - 5.0
    return ev.phrase_stats(pred_fn(gt), gt, [0, 1, 2, 1, 0], [60, 150, 60, 150, 60],
                           {0: "pau", 1: "a", 2: "k"}, 172.27, lang=JA)


def _avg_value(st, key):
    return ev.metric_value([st], next(m for m in ev.build_metrics() if m["key"] == key))


def test_averaging_metrics_find_the_scale_that_shrank():
    from scipy.ndimage import uniform_filter1d
    same = _avg_stats(lambda g: g.copy())
    for mt in (m for m in ev.build_metrics() if m["sec"] == "averaging"):
        assert abs(ev.metric_value([same], mt) - 1.0) < 1e-9, mt["key"]
    fs = _avg_stats(lambda g: uniform_filter1d(g, 9, axis=0, mode="nearest"))   # 周波数方向にぼかした生成
    ts = _avg_stats(lambda g: uniform_filter1d(g, 9, axis=1, mode="nearest"))   # 時間方向にぼかした生成
    for b in range(ev.BANDS):
        for grp in ("vowel", "cons"):
            assert _avg_value(fs, f"fcon_fine:{grp}:{b}") < 0.6                    # 周波数方向のメリハリだけ縮む
            assert abs(_avg_value(ts, f"fcon_fine:{grp}:{b}") - 1.0) < 1e-6
            assert _avg_value(ts, f"mod_ph:{grp}:{b}") < 0.7                       # 時間方向の動きだけ縮む
            assert abs(_avg_value(fs, f"mod_ph:{grp}:{b}") - 1.0) < 1e-6
        assert _avg_value(ts, f"mod_sy:{b}") > _avg_value(ts, f"mod_ph:vowel:{b}")  # 遅い動きは縮みにくい
        assert _avg_value(ts, f"gv:{b}") < 0.5 and abs(_avg_value(fs, f"gv:{b}") - 1.0) < 1e-6
    # 子音（k、60 フレーム）だけで数える。30〜150 ms の窓は母音にかかってもよい（無音にかからなければ）
    assert same["fcon_fine:cons:gt:0"][1] == 60 * 32 and same["mod_ph:cons:gt:0"][1] == 60 * 32
    assert same["mod_sy:gt:0"][1] == (360 - 2 * 50) * 32                          # 窓（±50）が無音にかからないフレーム


def test_parse_run_fixes_the_step_count_and_checkpoints_per_run():
    assert ev.parse_run("log/decay_15@1") == ("log/decay_15", 1, None)
    assert ev.parse_run("log/teacher_s015@16") == ("log/teacher_s015", 16, None)
    assert ev.parse_run("log/base_v2") == ("log/base_v2", None, None)
    assert ev.parse_run("log/odd@name") == ("log/odd@name", None, None)
    assert ev.parse_run("log/student@1:16000,17000") == ("log/student", 1, [16000, 17000])
    assert ev.parse_run("log/student:18000") == ("log/student", None, [18000])


def _cons_stats(pred_fn, seed=3, lang=JA):
    """子音の種類別の指標用の句: pau / a / s / a / n / a / A / cl / pau（無声子音 s 30・有声子音 n 30・
    無声化した母音 A 20・その他 cl 10 フレーム）。正解の s には bin 70 あたりに雑音の山を置く。
    pred_fn(gt, s_frames) が生成を作る。"""
    rng = np.random.default_rng(seed)
    durs = [20, 60, 30, 60, 30, 60, 20, 10, 20]
    gt = rng.normal(-5.0, 1.0, (128, sum(durs)))
    gt[:, 80:110] += 4.0 * np.exp(-((np.arange(128) - 70) / 5.0) ** 2)[:, None]
    st = ev.phrase_stats(pred_fn(gt, slice(80, 110)), gt, [0, 1, 2, 1, 3, 1, 4, 5, 0], durs,
                         {0: "pau", 1: "a", 2: "s", 3: "n", 4: "A", 5: "cl"}, 172.27, lang=lang)
    return st


def test_consonant_classes_split_voiced_and_voiceless():
    from scipy.ndimage import uniform_filter1d
    same = _cons_stats(lambda g, s: g.copy())
    assert [same[f"cerr:{c}"][1] for c in ev.CLASSES] == [20, 30, 30, 10]       # 無声化母音・無声・有声・その他
    assert same["ph_err:s"][1] == 30 and same["ph_fcon:s:gt"][1] == 30 * 96     # 973 Hz 以上 = 帯域 1〜3
    assert [JA.consonant_class(p) for p in ("A", "sh", "ts", "n", "r", "cl", "br")] == \
           ["devoiced", "ucons", "ucons", "vcons", "vcons", "other", "other"]
    # 言語パックが無ければ母音が無い扱い: pau 以外は全部「子音など」・子音の種類は「その他」（ほかの指標は同じ）
    none = _cons_stats(lambda g, s: g.copy(), lang=None)
    assert "grp:vowel" not in none and none["grp:cons"][1] == sum([60, 30, 60, 30, 60, 20, 10])
    assert none["cerr:other"][1] == 270 and none["all"] == same["all"]

    def smooth_s(g, s):                                                          # s だけ周波数方向にぼかす
        p = g.copy()
        p[:, s] = uniform_filter1d(g[:, s], 9, axis=0, mode="nearest")
        return p

    def raise_s(g, s):                                                           # s だけスペクトルを 8 bin 上へ
        p = g.copy()
        p[8:, s] = g[:-8, s]
        p[:8, s] = g[0, s]
        return p

    sm, up = _cons_stats(smooth_s), _cons_stats(raise_s)
    val = lambda st, key: ev.metric_value([st], next(m for m in ev.build_metrics() if m["key"] == key))
    for b in range(ev.BANDS):
        assert val(sm, f"fcon_coarse:ucons:{b}") < 0.8 and abs(val(sm, f"fcon_coarse:vcons:{b}") - 1.0) < 1e-12
    assert val(up, "cent:ucons") > 50.0 and val(up, "cent:vcons") == 0.0 and val(same, "cent:ucons") == 0.0


def test_compare_pairs_checkpoints_by_rank_when_steps_differ():
    model = {s - 2000: v for s, v in _run(shift_edge={s: -0.3 for s in STEPS},
                                           shift_mid={s: -0.3 for s in STEPS}).items()}   # 16k〜18k
    base = {s + 13000: v for s, v in _run().items()}                                      # 31k〜33k
    assert ev.ckpt_pairs(model, base) == [(16000, 31000), (17000, 32000), (18000, 33000)]
    c = ev.compare(model, base, ALL, n_boot=50)
    assert c["steps"] == [16000, 17000, 18000] and c["base_steps"] == [31000, 32000, 33000]
    assert ev.judge(c) == "改善"
    assert ev.ckpt_pairs({1: [], 2: [], 3: []}, {5: [], 6: []}) == [(2, 5), (3, 6)]   # 学習の進んだ方からそろえる


def test_report_lists_consonants_and_draws_the_consonant_figures(tmp_path):
    from pathlib import Path
    ck = {"base": {s: [_cons_stats(lambda g, s_: g.copy(), seed=i) for i in range(12)] for s in STEPS},
          "model": {s: [_cons_stats(lambda g, s_: g + 0.05, seed=i) for i in range(12)] for s in STEPS}}
    report, summary = ev.build_report(ck, ["base", "model"], n_items=12, n_boot=50)
    assert "## 子音ごと" in report and "### 子音の種類別" in report
    rows = {r["ph"]: r for r in summary["phonemes"]}
    assert set(rows) == {"s", "n"} and rows["s"]["frames"] == 360               # 30 × 12 句（A・cl は 300 未満）
    assert rows["s"]["cls"] == "ucons" and abs(rows["s"]["err"]["model"] - 0.05 * ev.DB) < 1e-9
    assert "Hz" in report
    assert "## 模様" in report and summary["pattern_peak"]["base"][summary["band_names"][0]] == 0.0
    ev.plot_curves(ck, ["base", "model"], Path(tmp_path))
    assert (tmp_path / "consonant_classes.png").stat().st_size > 0
    assert (tmp_path / "consonant_phonemes.png").stat().st_size > 0
    assert (tmp_path / "pattern_spectrum.png").stat().st_size > 0


def _harm_stats(pred_fn, seed=5, f0=250.0, T=200):
    """倍音の指標用の句（pau 10 / a 180 / pau 10）。正解 = 倍音の位置で山・間で谷になる櫛形（mel の中心周波数 f で
    cos(2π f / F0)、ln で ±3）+ 時間方向の小さな揺らぎ。pred_fn(gt, rng) が生成を作る。"""
    rng = np.random.default_rng(seed)
    gt = -5.0 + 3.0 * np.cos(2 * np.pi * ev.mel_hz(128) / f0)[:, None] + rng.normal(0, 0.05, (128, T))
    return ev.phrase_stats(pred_fn(gt, rng), gt, [0, 1, 0], [10, 180, 10], {0: "pau", 1: "a"}, 172.27,
                           uv=np.ones(T), f0_logf0=np.full(T, np.log2(f0)))


def test_harmonic_metrics_separate_filled_valleys_from_log_domain_noise():
    val =lambda st, key: ev.metric_value([st], next(m for m in ev.build_metrics() if m["key"] == key))
    same = _harm_stats(lambda g, r: g.copy())
    for g in ev.HARM_RANGES:
        assert abs(val(same, f"harm_depth:{g}") - 1.0) < 1e-12 and val(same, f"harm_valley_bias:{g}") == 0.0
        assert abs(val(same, f"harm_grain_peak:{g}") - 1.0) < 1e-12
    filled = _harm_stats(lambda g, r: np.log(np.exp(g) + np.exp(-7.0)))     # 振幅でノイズの床を足す → 谷が埋まる
    noisy = _harm_stats(lambda g, r: g + r.normal(0, 0.5, g.shape))         # 対数の mel でノイズを足す
    for g in ev.HARM_RANGES:
        assert val(filled, f"harm_depth:{g}") < 0.9
        assert val(filled, f"harm_valley_bias:{g}") > 3.0 * abs(val(filled, f"harm_peak_bias:{g}"))
        assert abs(val(filled, f"harm_grain_peak:{g}") - 1.0) < 0.2           # 山はざらつかない
        assert val(noisy, f"harm_grain_peak:{g}") > 3.0 and val(noisy, f"harm_grain_valley:{g}") > 3.0
    assert "harm_depth:lo:gt" not in _cons_stats(lambda g, s: g.copy())     # F0 を渡さなければ出さない


def test_report_and_figure_include_the_harmonic_metrics(tmp_path):
    from pathlib import Path
    ck = {"base": {s: [_harm_stats(lambda g, r: g.copy(), seed=i) for i in range(3)] for s in STEPS},
          "filled": {s: [_harm_stats(lambda g, r: np.log(np.exp(g) + np.exp(-7.0)), seed=i) for i in range(3)]
                     for s in STEPS}}
    report, summary = ev.build_report(ck, ["base", "filled"], n_items=3, n_boot=50)
    assert "### 倍音の構造" in report and summary["filled"]["harm_depth:lo"]["judge"] == "悪化"
    ev.plot_curves(ck, ["base", "filled"], Path(tmp_path))
    assert (tmp_path / "harmonics.png").stat().st_size > 0


def test_pattern_metric_finds_a_regular_pattern_but_not_smoothing():
    """4 フレームおきの縦縞（規則的な模様）は突出として出る。時間方向にぼかしただけ（なだらかな減り方）は出ない。"""
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(4)
    gt = rng.normal(-5.0, 1.0, (128, 640))
    args = ([0, 1, 0], [20, 600, 20], {0: "pau", 1: "a"}, 172.27)
    stripes = gt + 0.8 * np.sin(2 * np.pi * np.arange(640) / 4.0)[None, :]
    smooth = uniform_filter1d(gt, 3, axis=1, mode="nearest")

    def peak(pred):
        st = ev.phrase_stats(pred, gt, *args)
        return max(ev.pattern_peak(ev.pattern_db([st], b)) for b in range(ev.BANDS))

    assert peak(gt.copy()) == 0.0
    assert peak(stripes) > 6.0
    assert peak(smooth) < 2.0
    st = ev.phrase_stats(stripes, gt, *args)
    assert st["pat:gt:0"][1] == 17                                   # 無音を含まないタイル（32 フレームずつ）
    db = ev.pattern_db([st], 0)
    kf, kt = np.unravel_index(np.argmax(db), db.shape)
    assert kt == ev.PAT_FRAMES // 4 and kf == db.shape[0] // 2      # 時間方向 4 フレーム周期・bin 方向は一様


def test_report_and_figures_include_the_averaging_metrics(tmp_path):
    from pathlib import Path
    from scipy.ndimage import uniform_filter1d
    ck = {"base": {s: [_avg_stats(lambda g: g.copy(), seed=s)] for s in STEPS},
          "smooth": {s: [_avg_stats(lambda g: uniform_filter1d(g, 9, axis=1, mode="nearest"), seed=s)] for s in STEPS}}
    report, summary = ev.build_report(ck, ["base", "smooth"], n_items=1, n_boot=50)
    assert "### 平均化" in report and summary["metrics"]["gv:0"]["values"]["base"] == 1.0
    assert summary["smooth"]["gv:0"]["diff"] > 0.4                                # |比 − 1| が増えた
    ev.plot_curves(ck, ["base", "smooth"], Path(tmp_path))
    assert (tmp_path / "averaging_metrics.png").stat().st_size > 0
