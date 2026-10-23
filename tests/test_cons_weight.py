"""子音の損失の重み（train.cons_weight、leapsinger/phonemes.py）: 音素の分け方、フレームの重み、dataset と collate、
重みの付いた損失（付けないときは今までと同じ）、train.py を通した起動。"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LEAPSINGER_EXC_COMPILE", "0")

from test_reflow import _cfg, _mini_db, _mini_ds, _model  # noqa: E402


def _ja_consonants():
    from leapsinger.phonemes import load_lang
    return load_lang("ja").consonant_ids()


def test_japanese_consonants_are_everything_but_vowels_and_silence():
    """子音 = 母音（a i u e o N EP）と静音（pau br）以外のすべて（cl・無声化母音・Edge なども子音。dict/ja.yaml）。"""
    from leapsinger.phonemes import find_lang, load_lang
    from preprocess.vocab import Vocab
    ph = Vocab.load().phonemes
    ja = load_lang("ja")
    assert find_lang(ph) == ja
    cons = {ph[i] for i in ja.consonant_ids(ph)}
    assert cons == set(ph) - {"a", "i", "u", "e", "o", "N", "EP", "pau", "br"}
    assert {"k", "U", "r", "cl", "Edge", "GlottalStop", "h_end", "cl_end"} <= cons


def test_frame_weights_cover_the_consonant_and_the_head_of_the_next_phoneme():
    from leapsinger.phonemes import consonant_frame_weights
    K, A, S = 12, 1, 16                                                             # cons_ids で子音を決める
    ids, frames = [0, K, A, S, 0], [5, 4, 10, 3, 6]                                 # 計 28 フレーム
    w = consonant_frame_weights(ids, frames, {K, S}, 28, 2.0, next_frames=5)
    ref = np.ones(28, np.float32)
    ref[5:9] = 2.0                                                                  # k
    ref[9:14] = 2.0                                                                 # 次の a の頭 5 フレーム
    ref[19:22] = 2.0                                                                # s（次が pau なので延ばさない）
    assert w.dtype == np.float32 and np.array_equal(w, ref)
    w = consonant_frame_weights(ids, frames, {K, S}, 28, 2.0)                       # next_frames 0 = 子音だけ
    assert w[8] == 2.0 and w[9] == 1.0
    assert np.array_equal(consonant_frame_weights(ids, frames, {K, S}, 28, 1.0, next_frames=5), np.ones(28, np.float32))
    # 次の音素が短いときはその音素の中だけ
    w = consonant_frame_weights([K, A, 3], [4, 2, 6], {K}, 12, 3.0, next_frames=5)
    assert np.array_equal(w, np.array([3] * 6 + [1] * 6, np.float32))
    # 音素のフレームの合計が n_frames より長くても（dataset は各音素を最低 1 フレームにする）はみ出さない
    w = consonant_frame_weights([0, K, A], [3, 4, 30], {K}, 10, 2.0, next_frames=5)
    assert np.array_equal(w, np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2], np.float32))
    w = consonant_frame_weights([A, K], [3, 2], {K}, 5, 2.0, next_frames=5)         # 最後の音素が子音
    assert np.array_equal(w, np.array([1, 1, 1, 2, 2], np.float32))


def test_dataset_returns_frame_weights_only_when_set(tmp_path):
    from dataset import LeapSingerDataset, acoustic_collate_fn
    ds = LeapSingerDataset([_mini_db(tmp_path)], "train", eval_songs=0, spk_map={"ritsu": 2})
    assert "frame_w" not in ds[0] and "frame_w" not in acoustic_collate_fn([ds[0], ds[1]])
    ds.set_consonant_weights(_ja_consonants(), 3.0, next_frames=5)
    it0, it1 = ds[0], ds[1]                                                         # 50 と 70 フレーム。pau・k・a・o・pau
    d = it0["ph_durs"]
    ref = np.ones(50, np.float32)
    ref[d[0]:d[0] + d[1] + 5] = 3.0                                                 # k と a の頭 5 フレーム
    assert np.array_equal(it0["frame_w"], ref)
    b = acoustic_collate_fn([it0, it1])
    assert b["frame_w"].shape == (2, 70) and b["frame_w"].dtype == torch.float32
    assert torch.equal(b["frame_w"][0, :50], torch.as_tensor(ref)) and (b["frame_w"][0, 50:] == 0).all()
    assert torch.equal(b["frame_w"][1], torch.as_tensor(it1["frame_w"]))
    n_w = sum(int((ds[i]["frame_w"] != 1).sum()) for i in range(len(ds)))
    assert ds.weighted_frame_share() == pytest.approx(n_w / sum(ds.frame_counts))


def test_weighted_loss_is_the_weighted_mean_and_all_ones_is_unweighted(tmp_path):
    """reflow の t=0 で比べる（乱数を使わないので、何回呼んでも同じ出力になる）。"""
    from dataset import acoustic_collate_fn
    from leapsinger.reflow import make_reflow_targets
    from train import _forward_flow_gan
    model, _ = _model(0.1)
    ds = _mini_ds(tmp_path, model)
    tg, _ = make_reflow_targets(model, ds, draws=1, teacher_steps=1, device="cpu")
    ds.attach_reflow_targets(tg)
    ds.set_consonant_weights(_ja_consonants(), 3.0, next_frames=5)
    b = acoustic_collate_fn([ds[0], ds[1]])
    plain = {k: v for k, v in b.items() if k != "frame_w"}
    ones = {**plain, "frame_w": (~b["frame_mask"]).float()}                         # 全部 1（詰め物は 0）
    with torch.no_grad():
        out = _forward_flow_gan(model, b, "l2", reflow_t="zero")
        ref = _forward_flow_gan(model, plain, "l2", reflow_t="zero")
        one = _forward_flow_gan(model, ones, "l2", reflow_t="zero")
        l1 = _forward_flow_gan(model, b, "l1", reflow_t="zero")
    assert torch.equal(out["x1_pred_n"], ref["x1_pred_n"])                          # 重みは出力を変えない
    assert torch.equal(one["flow"], ref["flow"]) and torch.equal(one["mel_l1"], ref["mel_l1"])
    W = (~b["frame_mask"]).float()[:, None, :] * b["frame_w"][:, None, :]
    err = out["x1_pred_n"] - b["reflow_x1"].float()                                 # t=0: 1 step の出力と教師の差
    assert torch.allclose(out["flow"], (err.pow(2) * W).sum() / (W.sum() * 128), rtol=1e-4)
    assert torch.allclose(l1["flow"], (err.abs() * W).sum() / (W.sum() * 128), rtol=1e-4)
    assert torch.allclose(out["mel_l1"], (err.abs() * W).sum() / (W.sum() * 128), rtol=1e-4)
    assert torch.equal(out["gt_l1"], ref["gt_l1"])                                  # 正解との差（記録用）は重みなし
    assert not torch.allclose(out["flow"], ref["flow"])


def test_cons_weight_settings_and_the_recipe():
    from train import cons_weight_settings, recipe_changes, train_recipe
    assert cons_weight_settings({}) is None
    assert cons_weight_settings({"cons_weight": 1.0}) is None and cons_weight_settings({"cons_weight": None}) is None
    assert cons_weight_settings({"cons_weight": 2}) == {"weight": 2.0, "next_ms": 30.0}
    assert cons_weight_settings({"cons_weight": 2, "cons_weight_next_ms": 0}) == {"weight": 2.0, "next_ms": 0.0}
    for bad in ({"cons_weight": 0}, {"cons_weight": -1}, {"cons_weight": float("nan")},
                {"cons_weight": 2, "cons_weight_next_ms": -5}):
        with pytest.raises(SystemExit):
            cons_weight_settings(bad)
    cfg = _cfg(0.0, 0.05)
    saved = train_recipe(cfg)
    assert saved["cons_weight"] is None                                             # 設定ファイルの既定は付けない
    old = {k: v for k, v in saved.items() if k != "cons_weight"}                    # 今までの run の ckpt（記録なし）
    assert recipe_changes(train_recipe(cfg), old) == {}
    cfg["train"]["cons_weight"] = 2.0
    assert recipe_changes(train_recipe(cfg), old) == {"cons_weight": (None, {"weight": 2.0, "next_ms": 30.0})}


def _run_train(tmp_path, data_dir, run_name, *extra):
    import train
    argv = ["train", "--config", os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), "--data_dirs", data_dir,
            "--run_name", run_name, "--out_root", str(tmp_path / "log"), "--device", "cpu", "--max_updates", "2",
            "--set", "gan.enabled=false", "--set", "train.amp=fp32", "--set", "train.num_workers=0",
            "--set", "train.vocoder=null", "--set", "train.eval_interval=1000", "--set", "train.save_interval=1000",
            "--set", "train.eval_items=1", "--set", "train.eval_gen_metrics=false", "--set", "data.eval_songs=1", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        train.main()
    finally:
        sys.argv = old
    return tmp_path / "log" / run_name


def test_train_main_runs_with_and_without_the_consonant_weights(tmp_path, capsys):
    data_dir = _mini_db(tmp_path)
    out = _run_train(tmp_path, data_dir, "cw", "--set", "train.cons_weight=2.0")
    log = capsys.readouterr().out
    assert "[cons_weight] 子音 " in log and "頭 5 フレーム（30 ms）" in log, log
    assert torch.load(out / "ckpt_000002.pt", map_location="cpu")["train_recipe"]["cons_weight"] == {
        "weight": 2.0, "next_ms": 30.0}
    off = _run_train(tmp_path, data_dir, "off")
    assert "[cons_weight] off" in capsys.readouterr().out
    assert torch.load(off / "ckpt_000002.pt", map_location="cpu")["train_recipe"]["cons_weight"] is None
