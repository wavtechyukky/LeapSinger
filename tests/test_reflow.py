"""reflow の教師（確率的な出発点 excitation.x0_sigma）まわり: ノイズの足し方、ckpt との往復、書き出しの拒否、
評価ツールの step 数別の名前。後半は生徒（leapsinger/reflow.py と train.reflow_*）: 組の作り方、dataset からの
取り出し、t=0 の損失、train.py を通した起動。"""
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("LEAPSINGER_EXC_COMPILE", "0")


def _cfg(x0_sigma, noise_ratio=0.0):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    cfg["model"]["n_speakers"] = 3
    cfg["excitation"]["x0_sigma"] = x0_sigma
    cfg["excitation"]["noise_ratio"] = noise_ratio
    return cfg


def _model(x0_sigma, noise_ratio=0.0):
    from preprocess.vocab import Vocab
    from train import build_model
    torch.manual_seed(0)
    model, arch = build_model(_cfg(x0_sigma, noise_ratio), Vocab.load())
    return model.eval(), arch


def _f0(T=60):
    return torch.full((1, T), float(np.log2(220.0))), torch.ones(1, T)


def test_x0_sigma_adds_gaussian_noise_in_normalised_units():
    plain, _ = _model(0.0)
    noisy, _ = _model(0.1)
    f0, uv = _f0()
    with torch.no_grad():
        base = plain._excitation_x0(f0, uv)
        eps = torch.randn_like(base)
        assert torch.equal(plain._excitation_x0(f0, uv, eps=eps), base)          # sigma 0: eps は効かない
        x = noisy._excitation_x0(f0, uv, eps=eps)
        assert torch.allclose(x, base + 0.1 * eps, atol=1e-6)                     # noise_ratio 0 なので決定論的
        a, b = noisy._excitation_x0(f0, uv), noisy._excitation_x0(f0, uv)
        assert not torch.allclose(a, b)                                           # eps 省略時は毎回新規
        assert abs(float((a - base).std()) - 0.1) < 0.01


def test_x0_sigma_round_trips_through_the_checkpoint_config():
    from infer import _build_from_config
    from preprocess.vocab import Vocab
    from train import ckpt_config, excitation_changes
    cfg = _cfg(0.15)
    model, arch = _model(0.15)
    cc = ckpt_config(cfg, arch, Vocab.load())
    assert cc["x0_sigma"] == 0.15
    rebuilt = _build_from_config(cc, torch.device("cpu"))
    assert rebuilt.x0_sigma == 0.15
    rebuilt.load_state_dict(model.state_dict())                                   # 構造は変わらない
    other = copy.deepcopy(cfg)
    other["excitation"]["x0_sigma"] = 0.05
    assert excitation_changes(other, cc) == {"x0_sigma": (0.15, 0.05)}


def test_multistep_inference_runs_from_the_noisy_start():
    model, _ = _model(0.1)
    f0, uv = _f0(40)
    ids = torch.tensor([[0, 12, 1, 0]])
    durs = torch.tensor([[5, 10, 20, 5]])
    with torch.no_grad():
        torch.manual_seed(3)
        m1 = model.infer(ids, durs, f0, uv, n_frames=40, num_steps=1, spk_id=torch.tensor([0]))
        torch.manual_seed(3)
        m8 = model.infer(ids, durs, f0, uv, n_frames=40, num_steps=8, spk_id=torch.tensor([0]))
    assert m1.shape == m8.shape == (1, 128, 40) and torch.isfinite(m8).all()
    assert not torch.allclose(m1, m8)


def test_export_refuses_a_stochastic_start():
    from export.cli import _validate
    model, _ = _model(0.1)
    try:
        _validate({"use_uv": False}, model, "diffsinger", "embed")
    except SystemExit as e:
        assert "x0_sigma" in str(e)
        return
    raise AssertionError("export should refuse x0_sigma > 0")


def test_eval_labels_per_step_count():
    import eval_ab as ev
    assert ev.run_labels("teacher", [None]) == {None: "teacher"}
    assert ev.run_labels("teacher", [8]) == {8: "teacher"}
    assert ev.run_labels("teacher", [1, 8]) == {1: "teacher@k1", 8: "teacher@k8"}


# ── 生徒 ──────────────────────────────────────────────────────────────────────

FR = 172.265625
SONGS = {"songA": (50, 70), "songB": (60,), "songC": (55,)}


def _mini_db(root, name="ritsu", songs=SONGS):
    """合成の小さな DB（shard.npz + metadata.json）。フレーズは pau・3 音素・pau、F0 は 220 Hz 付近。
    2 番目の音素は無声（uv 0。無声の区間の指標が出るように）。"""
    from preprocess.vocab import PAU_ID
    d = os.path.join(str(root), name)
    os.makedirs(d)
    rng = np.random.default_rng(0)
    arrays, phrases = {}, {}
    for song, lens in songs.items():
        for j, T in enumerate(lens):
            n = f"{song}_{j:04d}"
            phrases[n] = T
            frames = np.array([5, T // 3, T // 4, T - 10 - T // 3 - T // 4, 5])
            arrays[f"{n}|phoneme_ids"] = np.array([PAU_ID, 12, 1, 5, PAU_ID], np.int16)
            arrays[f"{n}|dur_sec"] = (frames / FR).astype(np.float32)
            arrays[f"{n}|f0_interp"] = (220.0 * (1 + 0.02 * np.sin(np.arange(T) / 5))).astype(np.float32)
            uv = np.ones(T, np.float32)
            uv[5:5 + T // 3] = 0.0
            arrays[f"{n}|uv"] = uv
            arrays[f"{n}|mel"] = rng.uniform(-11.0, 0.0, (128, T)).astype(np.float32)
    np.savez(os.path.join(d, "shard.npz"), **arrays)
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"frame_rate": FR, "hop_size": 256, "sample_rate": 44100, "phrases": phrases}, f)
    return d


def _mini_ds(root, model):
    from dataset import LeapSingerDataset
    ds = LeapSingerDataset([_mini_db(root)], "train", eval_songs=0, spk_map={"ritsu": 2})
    ds.warm_harm_cache(n_harm=model.n_harm, harm_decay=model.harm_decay, exc_hop=model.exc_hop,
                       use_uv=model.use_uv, device="cpu")
    return ds


def test_reflow_eps_is_reproducible_per_phrase_and_draw():
    from leapsinger.reflow import reflow_entropy, reflow_eps
    a = reflow_eps("ritsu/songA_0000", 1, 128, 40)
    assert a.shape == (128, 40) and a.dtype == torch.float32
    assert torch.equal(a, reflow_eps("ritsu/songA_0000", 1, 128, 40))
    assert not torch.allclose(a, reflow_eps("ritsu/songA_0000", 2, 128, 40))
    assert not torch.allclose(a, reflow_eps("ritsu/songA_0001", 1, 128, 40))
    assert not torch.allclose(a, reflow_eps("ritsu/songA_0000", 1, 128, 40, seed=1))
    assert abs(float(reflow_eps("x", 0, 128, 2000).std()) - 1.0) < 0.01
    assert len({tuple(reflow_entropy(f"x{k}", d, s)) for k in range(4) for d in range(8) for s in range(2)}) == 64
    torch.manual_seed(5)
    np.random.seed(5)
    r, rn = torch.rand(3), np.random.rand(3)
    reflow_eps("x", 0, 128, 10)                                                    # 全体の乱数は進めない
    torch.manual_seed(5)
    np.random.seed(5)
    assert torch.equal(torch.rand(3), r) and np.array_equal(np.random.rand(3), rn)


def test_reflow_settings_and_problems():
    import pytest
    from leapsinger.reflow import reflow_problems, reflow_settings
    assert reflow_settings({}) is None and reflow_settings({"reflow_teacher": None}) is None
    s = reflow_settings({"reflow_teacher": "log/t/ckpt_035000.pt"})
    assert s == {"teacher": "log/t/ckpt_035000.pt", "draws": 4, "teacher_steps": 8, "algorithm": "euler",
                 "t": "zero", "seed": 0}
    for bad in ({"reflow_t": "half"}, {"reflow_algorithm": "heun"}, {"reflow_draws": 0}):
        with pytest.raises(SystemExit):
            reflow_settings({"reflow_teacher": "x.pt", **bad})
    assert reflow_problems(_cfg(0.15)) == []
    assert reflow_problems(_cfg(0.15, 0.05)) == [] and reflow_problems(_cfg(0.0, 0.05)) == []   # 出発点ごと持つ
    assert len(reflow_problems(_cfg(0.0))) == 1                                   # ランダムさが無い
    c = _cfg(0.15)
    c["train"]["pitch_aug"] = True
    assert len(reflow_problems(c)) == 1


def test_config_mismatch_ignores_only_the_free_keys():
    from leapsinger.reflow import config_mismatch
    from preprocess.vocab import Vocab
    from train import ckpt_config
    _, arch = _model(0.15)
    teacher = ckpt_config(_cfg(0.15), arch, Vocab.load())
    cfg = _cfg(0.15)
    cfg["train"]["num_steps"] = 1
    cfg["train"]["recon_weight"] = 0.0
    assert config_mismatch(ckpt_config(cfg, arch, Vocab.load()), teacher) == {}
    cfg["excitation"]["harm_decay"] = 2.0
    assert config_mismatch(ckpt_config(cfg, arch, Vocab.load()), teacher) == {"harm_decay": (1.5, 2.0)}
    old = {k: v for k, v in teacher.items() if k != "x0_sigma"}                    # 記録の無い古い ckpt
    assert config_mismatch(ckpt_config(_cfg(0.15), arch, Vocab.load()), old) == {"x0_sigma": (None, 0.15)}


def test_reflow_targets_are_the_teachers_inference_from_the_seeded_start(tmp_path):
    from leapsinger.reflow import make_reflow_targets, reflow_eps
    model, _ = _model(0.1)
    ds = _mini_ds(tmp_path, model)
    tg, x0 = make_reflow_targets(model, ds, draws=3, teacher_steps=2, device="cpu")
    assert x0 is None                                                              # noise_ratio 0: eps を作り直す
    assert sorted(tg) == list(range(len(ds)))
    for i in range(len(ds)):
        T = ds.frame_counts[i]
        assert tg[i].shape == (3, 128, T) and tg[i].dtype == np.float16
    assert not np.allclose(tg[0][0], tg[0][1])                                    # 通りごとに違う出発点
    again, _ = make_reflow_targets(model, ds, draws=3, teacher_steps=2, device="cpu")
    assert all(np.array_equal(tg[i], again[i]) for i in tg)                        # 決定論的
    # infer_mel と同じ batch 1・倍音波形を毎回作る経路で、同じ eps から解いたものと一致する
    it = ds[1]
    T = len(it["f0_logf0"])
    eps = reflow_eps(ds.phrase_key(1), 2, 128, T)
    with torch.no_grad():
        mel = model.infer(torch.as_tensor(it["ph_ids"])[None], torch.as_tensor(it["ph_durs"])[None],
                          torch.as_tensor(it["f0_logf0"])[None], torch.as_tensor(it["uv"])[None],
                          n_frames=T, num_steps=2, spk_id=torch.tensor([2]), eps=eps[None])
    ref = model.flow._norm(mel)[0].numpy()
    assert np.abs(tg[1][2].astype(np.float32) - ref).max() < 2e-3


def test_dataset_returns_a_pair_and_collate_pads_it(tmp_path):
    from dataset import acoustic_collate_fn
    from leapsinger.reflow import make_reflow_targets, reflow_eps
    import pytest
    model, _ = _model(0.1)
    ds = _mini_ds(tmp_path, model)
    with pytest.raises(ValueError):
        ds.attach_reflow_targets({0: None})                                        # 足りない組は受け付けない
    tg, _ = make_reflow_targets(model, ds, draws=2, teacher_steps=1, seed=7, device="cpu")
    ds.attach_reflow_targets(tg, seed=7)
    seen = set()
    for _ in range(20):
        it = ds[0]
        d = it["reflow_d"]
        seen.add(d)
        assert np.array_equal(it["reflow_x1"], tg[0][d]) and "reflow_x0" not in it
        assert np.array_equal(it["reflow_eps"], reflow_eps(ds.phrase_key(0), d, 128, ds.frame_counts[0], 7).numpy())
    assert seen == {0, 1}
    items = [ds[0], ds[1]]                                                         # 50 と 70 フレーム
    b = acoustic_collate_fn(items)
    assert "reflow_x0" not in b
    assert b["reflow_x1"].shape == b["reflow_eps"].shape == (2, 128, 70)
    assert b["reflow_x1"].dtype == torch.float16 and b["reflow_eps"].dtype == torch.float32
    assert torch.equal(b["reflow_x1"][1], torch.as_tensor(items[1]["reflow_x1"]))
    assert torch.equal(b["reflow_x1"][0, :, 50:], torch.zeros(128, 20, dtype=torch.float16))
    assert torch.equal(b["reflow_eps"][0, :, :50], torch.as_tensor(items[0]["reflow_eps"]))
    assert b["reflow_d"].tolist() == [items[0]["reflow_d"], items[1]["reflow_d"]]


def test_student_loss_at_t0_is_the_one_step_gap_to_the_teacher(tmp_path):
    from dataset import acoustic_collate_fn
    from leapsinger.reflow import make_reflow_targets
    from train import _forward_flow_gan
    model, _ = _model(0.1)
    ds = _mini_ds(tmp_path, model)
    tg, _ = make_reflow_targets(model, ds, draws=2, teacher_steps=1, device="cpu")
    ds.attach_reflow_targets(tg)
    b = acoustic_collate_fn([ds[0], ds[1]])
    gt = model.flow._norm(b["target_mel"])
    valid = (~b["frame_mask"])[:, None, :].float()
    with torch.no_grad():
        out = _forward_flow_gan(model, b, "l2", reflow_t="zero")
        x0 = model._excitation_x0(b["f0_logf0"], b["uv"], harm_wave=b["harm_wave"], eps=b["reflow_eps"])
    assert torch.equal(out["t"], torch.zeros(2))
    assert torch.allclose(out["x_t"], x0, atol=1e-6)                               # t=0 なので出発点そのもの
    gap = ((out["x1_pred_n"] - b["reflow_x1"].float()).pow(2) * valid).sum() / (valid.sum() * 128)
    assert torch.allclose(out["flow"], gap, rtol=1e-4)                             # 1 step の出力と教師の差
    assert torch.equal(out["x1_n"], gt)                                            # 判別器の正例は正解の mel
    l1 = ((out["x1_pred_n"] - gt).abs() * valid).sum() / (valid.sum() * 128)
    assert torch.allclose(out["gt_l1"], l1, rtol=1e-4)
    with torch.no_grad():
        uni = _forward_flow_gan(model, b, "l2", reflow_t="uniform")
        plain = _forward_flow_gan(model, b, "l2")
    assert (uni["t"] > 0).all()
    assert "gt_l1" not in plain and torch.equal(plain["x1_n"], gt)


def test_waveform_noise_teacher_keeps_the_start_point_with_each_pair(tmp_path):
    """擬似 mel に波形の白色ノイズがある教師（noise_ratio 0.05、σ 0）: 出発点そのものを組と一緒に持つ。
    通りごとに違うノイズ・作り直しても同じ・教師にその出発点を渡して解いた出力と一致・生徒の t=0 はその出発点から。"""
    from dataset import acoustic_collate_fn
    from leapsinger.reflow import make_reflow_targets
    from train import _forward_flow_gan
    model, _ = _model(0.0, 0.05)
    ds = _mini_ds(tmp_path, model)
    tg, x0 = make_reflow_targets(model, ds, draws=3, teacher_steps=2, device="cpu")
    assert x0 is not None and sorted(x0) == sorted(tg)
    assert x0[1].shape == tg[1].shape and x0[1].dtype == np.float16
    assert not np.allclose(x0[1][0], x0[1][1])                                    # 通りごとに違う波形のノイズ
    tg2, x02 = make_reflow_targets(model, ds, draws=3, teacher_steps=2, device="cpu")
    assert all(np.array_equal(x0[i], x02[i]) and np.array_equal(tg[i], tg2[i]) for i in tg)   # 作り直しても同じ
    it = ds[1]
    T = len(it["f0_logf0"])
    with torch.no_grad():
        mel = model.infer(torch.as_tensor(it["ph_ids"])[None], torch.as_tensor(it["ph_durs"])[None],
                          torch.as_tensor(it["f0_logf0"])[None], torch.as_tensor(it["uv"])[None],
                          n_frames=T, num_steps=2, spk_id=torch.tensor([2]),
                          x0=torch.as_tensor(x0[1][2].astype(np.float32))[None])
    assert np.abs(tg[1][2].astype(np.float32) - model.flow._norm(mel)[0].numpy()).max() < 2e-3
    import pytest
    with pytest.raises(ValueError):
        ds.attach_reflow_targets(tg, x0={0: x0[0]})                                # 足りない出発点は受け付けない
    ds.attach_reflow_targets(tg, x0=x0)
    item = ds[1]
    assert np.array_equal(item["reflow_x0"], x0[1][item["reflow_d"]]) and "reflow_eps" not in item
    b = acoustic_collate_fn([ds[0], ds[1]])
    assert b["reflow_x0"].dtype == torch.float16 and "reflow_eps" not in b
    with torch.no_grad():
        out = _forward_flow_gan(model, b, "l2", reflow_t="zero")
    assert torch.equal(out["x_t"], b["reflow_x0"].float())                         # t=0 は保存した出発点そのもの


def test_train_recipe_records_the_reflow_settings():
    from train import recipe_changes, train_recipe
    cfg = _cfg(0.15)
    assert train_recipe(cfg)["reflow"] is None
    cfg["train"]["reflow_teacher"] = "log/teacher_s015/ckpt_035000.pt"
    saved = train_recipe(cfg)
    assert saved["reflow"]["teacher"] == "log/teacher_s015/ckpt_035000.pt"
    cfg["train"]["reflow_draws"] = 8
    ch = recipe_changes(train_recipe(cfg), saved)
    assert list(ch) == ["reflow"] and ch["reflow"][1]["draws"] == 8
    old = {k: v for k, v in saved.items() if k != "reflow"}                       # 教師の run の ckpt（記録なし）
    assert recipe_changes(train_recipe(_cfg(0.15)), old) == {}


def _run_train(tmp_path, data_dir, teacher, *extra):
    import train
    argv = ["train", "--config", os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), "--data_dirs", data_dir,
            "--run_name", "student", "--out_root", str(tmp_path / "log"), "--device", "cpu", "--max_updates", "2",
            "--set", "excitation.x0_sigma=0.1", "--set", "excitation.noise_ratio=0.0",
            "--set", "gan.enabled=false", "--set", f"train.reflow_teacher={teacher}",
            "--set", "train.reflow_draws=2", "--set", "train.reflow_teacher_steps=2",
            "--set", "train.lr=0.0", "--set", "train.amp=fp32", "--set", "train.num_workers=0",
            "--set", "train.vocoder=null", "--set", "train.eval_interval=1000", "--set", "train.save_interval=1000",
            "--set", "train.eval_items=1", "--set", "data.eval_songs=1", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        train.main()
    finally:
        sys.argv = old
    return tmp_path / "log" / "student"


def test_train_main_starts_the_student_from_the_teacher(tmp_path):
    """train.py を CPU で 2 step だけ通す（lr 0 なので重みは教師のまま残る）。"""
    import pytest
    from preprocess.vocab import Vocab
    from train import ckpt_config
    data_dir = _mini_db(tmp_path)
    cfg = _cfg(0.1)
    model, arch = _model(0.1)
    torch.manual_seed(1)
    for p in model.parameters():                                                   # 初期値と区別できる重み
        p.data.add_(0.01 * torch.randn_like(p))
    teacher = tmp_path / "teacher.pt"
    torch.save({"model": model.state_dict(), "config": ckpt_config(cfg, arch, Vocab.load()), "step": 123,
                "code": {"sha1": "abc", "commit": None, "dirty": None}}, teacher)
    out = _run_train(tmp_path, data_dir, teacher)
    ck = torch.load(out / "ckpt_000002.pt", map_location="cpu")
    assert ck["reflow"]["teacher_step"] == 123 and ck["reflow"]["draws"] == 2
    assert ck["train_recipe"]["reflow"]["teacher"] == str(teacher)
    assert all(torch.equal(v, ck["model"][k]) for k, v in model.state_dict().items())
    # 最後の評価で TensorBoard の eval_gen/ に生成の指標が出て、保存した ckpt を eval_ab で測った値と同じ
    import eval_ab as ev
    from infer import infer_mel, load_acoustic
    from tensorboard.backend.event_processing import event_accumulator as ea
    acc = ea.EventAccumulator(str(next(out.glob("events.out.tfevents.*"))), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    gen = [t for t in acc.Tags()["scalars"] if t.startswith("eval_gen/")]
    assert len(gen) == 15 and "eval_gen/err_db" in gen, gen
    assert sum(t.startswith("eval_gen/uv_bias_db_b") for t in gen) == 4, gen
    tb = acc.Scalars("eval_gen/err_db")[-1]
    ecfg = _cfg(0.1)
    ecfg["data"]["eval_songs"] = 1
    items, fr = ev.load_eval_items(ecfg, [data_dir])
    m, mcfg = load_acoustic(str(out / "ckpt_000002.pt"), "cpu")
    id2ph = dict(enumerate(mcfg["phonemes"]))
    stats = [ev.phrase_stats(infer_mel(m, it, num_steps=1, device="cpu"), it["target_mel"], it["ph_ids"],
                             it["ph_durs"], id2ph, fr, uv=it["uv"]) for it in items]
    assert tb.step == 2 and abs(ev.metric_value(stats, ev.ALL) - tb.value) < 1e-3
    with pytest.raises(SystemExit) as e:                                          # 教師と違う設定は止める
        _run_train(tmp_path, data_dir, teacher, "--set", "excitation.harm_decay=2.0", "--run_name", "s2")
    assert "harm_decay: 1.5 -> 2.0" in str(e.value)
    with pytest.raises(SystemExit) as e:                                          # 教師と違う noise_ratio も止める
        _run_train(tmp_path, data_dir, teacher, "--set", "excitation.noise_ratio=0.05", "--run_name", "s3")
    assert "noise_ratio" in str(e.value)
    # 波形の白色ノイズがある教師（noise_ratio 0.05、σ 0）: 出発点ごと組にして学習が回る
    model2, arch2 = _model(0.0, 0.05)
    teacher2 = tmp_path / "teacher_wave.pt"
    torch.save({"model": model2.state_dict(), "config": ckpt_config(_cfg(0.0, 0.05), arch2, Vocab.load()), "step": 7,
                "code": {"sha1": "def", "commit": None, "dirty": None}}, teacher2)
    _run_train(tmp_path, data_dir, teacher2, "--set", "excitation.x0_sigma=0.0",
               "--set", "excitation.noise_ratio=0.05", "--run_name", "s4")
    ck2 = torch.load(tmp_path / "log" / "s4" / "ckpt_000002.pt", map_location="cpu")
    assert ck2["reflow"]["teacher_step"] == 7
    assert all(torch.equal(v, ck2["model"][k]) for k, v in model2.state_dict().items())   # lr 0 で教師のまま
