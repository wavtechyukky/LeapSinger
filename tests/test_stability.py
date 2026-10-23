"""学習の安定化まわり: 極小バッチを作らないサンプラー（train.min_batch_items）、長尺の除外
（data.max_sec）、パディングを除いた varL、勾配ノルムの記録（StepMonitor）。"""
import collections
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import FrameBasedBatchSampler, LeapSingerDataset  # noqa: E402
from leapsinger.modules.discriminators.jcu import laplacian_var_ratio  # noqa: E402

FR = 44100 / 256


def _old_build(frame_counts, indices, max_frames, max_items):
    """変更前の _build（短い順に詰める）をそのまま写したもの。"""
    order = sorted(indices, key=lambda i: frame_counts[i])
    batches, cur, cur_max = [], [], 0
    for idx in order:
        fc = frame_counts[idx]
        nmax = max(cur_max, fc)
        if cur and (nmax * (len(cur) + 1) > max_frames or len(cur) + 1 > max_items):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, fc)
    if cur:
        batches.append(cur)
    return batches


def _lengths(seed=0, n=2500):
    """実データに似た長さ分布（数秒が中心、少数の 30〜55 秒）。"""
    rng = np.random.default_rng(seed)
    fc = list(np.clip(rng.gamma(3.0, 2.5, n) * FR, 0.4 * FR, 25 * FR).astype(int))
    fc[:4] = [int(55.4 * FR), int(41.0 * FR), int(39.2 * FR), int(33.7 * FR)]
    return fc


def test_min_items_zero_keeps_the_old_packing():
    fc = _lengths()
    w = [1.0] * len(fc)
    s = FrameBasedBatchSampler(fc, 240000, 128, shuffle=True, weights=w, min_items=0)
    for _ in range(3):
        g = torch.Generator().manual_seed(s.seed + s.epoch)
        idxs = torch.multinomial(torch.as_tensor(w), len(fc), replacement=True, generator=g).tolist()
        assert s._build(idxs) == _old_build(fc, idxs, 240000, 128)
        list(iter(s))                                   # advance one epoch


def test_old_packing_leaves_the_longest_alone_and_the_fix_does_not():
    # 239 short phrases + one 55 s phrase: shortest-first packing closes the second batch when the
    # long one would push n*maxT over the budget, so the 55 s phrase ends up in a batch of its own
    fc = [1000] * 239 + [9550]
    old = _old_build(fc, range(len(fc)), 240000, 128)
    assert [len(b) for b in old][-1] == 1 and fc[old[-1][0]] == 9550
    new = FrameBasedBatchSampler(fc, 240000, 128, shuffle=False, min_items=16)._build(range(len(fc)))
    assert min(len(b) for b in new) >= 16
    assert all(len(b) * max(fc[i] for i in b) <= 240000 and len(b) <= 128 for b in new)
    assert sorted(i for b in new for i in b) == list(range(len(fc)))


def test_min_items_removes_tiny_batches_and_keeps_limits():
    fc = _lengths()
    w = [1.0] * len(fc)
    s = FrameBasedBatchSampler(fc, 240000, 128, shuffle=True, weights=w, min_items=16)
    for _ in range(20):
        g = torch.Generator().manual_seed(s.seed + s.epoch)
        idxs = torch.multinomial(torch.as_tensor(w), len(fc), replacement=True, generator=g).tolist()
        batches = s._build(idxs)
        assert collections.Counter(i for b in batches for i in b) == collections.Counter(idxs)
        for b in batches:
            assert len(b) >= 16, len(b)
            assert len(b) <= 128 and len(b) * max(fc[i] for i in b) <= 240000
        list(iter(s))


def test_max_sec_drops_long_phrases():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "db")
        os.makedirs(d)
        phrases = {f"song{i}_0000": int(s * FR) for i, s in enumerate([3, 12, 29.9, 30.5, 55.4])}
        json.dump({"phrases": phrases, "frame_rate": FR, "hop_size": 256}, open(os.path.join(d, "metadata.json"), "w"))
        all_ = LeapSingerDataset([d], "train", eval_songs=0, min_sec=0.3)
        cut = LeapSingerDataset([d], "train", eval_songs=0, min_sec=0.3, max_sec=30)
        assert len(all_) == 5 and len(cut) == 3
        assert max(cut.frame_counts) <= 30 * FR


def test_varL_mask_ignores_padding():
    torch.manual_seed(0)
    gt = torch.randn(2, 128, 300)
    pred = gt * 0.8
    valid = torch.ones(2, 300, dtype=torch.bool)
    valid[1, 200:] = False
    a = laplacian_var_ratio(pred, gt, valid=valid)
    pred2 = pred.clone()
    pred2[1, :, 200:] = torch.randn(128, 100) * 50     # garbage in the padding
    b = laplacian_var_ratio(pred2, gt, valid=valid)
    assert abs(a - b) < 1e-6
    assert abs(laplacian_var_ratio(pred2, gt) - a) > 1.0  # unmasked version is polluted (the old metric)
    assert abs(a - 0.64) < 0.02                          # (0.8)^2


def test_step_monitor_logs_spikes():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train import StepMonitor
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "spikes.log")
        m = StepMonitor(path, window=50, spike_factor=5.0, warmup=50)
        for s in range(1, 101):
            line = m.update(s, 0.5 + 0.01 * (s % 3), None, 128, 0.03, lambda: "n=128")
            assert line is None
        line = m.update(101, 9.0, 0.2, 2, 0.4, lambda: "n=2 FACE_0005")
        assert line and "FACE_0005" in line and "step 101" in line
        assert open(path, encoding="utf-8").read().count("\n") == 1
        sm = m.summary()
        assert sm["train/grad_norm_max"] == 9.0 and sm["train/batch_items_min"] == 2.0
        assert sm["train/d_grad_norm_max"] == 0.2
        assert m.summary() == {}                          # reset after summary
        m.add_time(data_wait=0.01, h2d=0.02, step=0.3)
        m.add_time(data_wait=0.03, h2d=0.02, step=0.5)
        m.add_gpu(exc=0.01, net=0.1)
        m.add_gpu(exc=0.03, net=0.3)
        m.add_gpu()                                       # empty (no marks) is ignored
        sm = m.summary()
        assert abs(sm["time/data_wait_ms"] - 20.0) < 1e-9 and abs(sm["time/step_ms"] - 400.0) < 1e-9
        assert abs(sm["time/steps_per_sec"] - 2 / 0.88) < 1e-9     # GPU spans are not counted twice
        assert abs(sm["time/gpu_exc_ms"] - 20.0) < 1e-9 and abs(sm["time/gpu_net_ms"] - 200.0) < 1e-9


def _tiny_model_and_batch():
    os.environ.setdefault("LEAPSINGER_EXC_COMPILE", "0")
    import yaml
    from preprocess.vocab import Vocab
    from train import build_model
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    cfg["model"]["n_speakers"] = 3
    cfg["model"]["pos_cond"]["phoneme"] = True
    torch.manual_seed(0)
    model, _ = build_model(cfg, Vocab.load())
    model.train()
    B, Tp = 2, 6
    ids = torch.tensor([[0, 12, 1, 25, 3, 0], [0, 11, 2, 0, 0, 0]])
    durs = torch.tensor([[20, 6, 40, 5, 30, 20], [15, 7, 50, 20, 0, 0]])
    T = int(durs.sum(1).max())
    fm = torch.zeros(B, T, dtype=torch.bool); fm[1, int(durs[1].sum()):] = True
    pm = torch.zeros(B, Tp, dtype=torch.bool); pm[1, 4:] = True
    b = dict(ph_ids=ids, ph_durs=durs, f0_logf0=torch.full((B, T), float(np.log2(250.0))),
             uv=torch.ones(B, T), target_mel=torch.randn(B, 128, T) * 2 - 6, ph_mask=pm, frame_mask=fm,
             spk_id=torch.tensor([0, 2]), style_id=torch.tensor([0, 0]))
    return model, b


def test_resume_keeps_the_checkpoint_position_settings():
    from train import align_pos_cond_to_ckpt
    new_default = {"phoneme": True, "unit": False, "bins": 8, "encoding": "sinusoidal", "nucleus": ["a"]}
    # base_agc（位置なし）を新しい既定の config で再開 → ckpt に合わせて off
    cfg = {"model": {"pos_cond": dict(new_default), "dur_cond": False}}
    ch = align_pos_cond_to_ckpt(cfg, {"pos_cond": {"phoneme": False, "unit": False, "bins": 8}, "dur_cond": False})
    assert ch == ((True, False, "sinusoidal", None, False), (False, False, None, None, False))
    assert cfg["model"]["pos_cond"]["phoneme"] is False and cfg["model"]["pos_cond"]["unit"] is False
    # pos_agc（音素 + 単位、encoding の記録なし = 旧符号化）
    cfg = {"model": {"pos_cond": dict(new_default), "dur_cond": False}}
    align_pos_cond_to_ckpt(cfg, {"pos_cond": {"phoneme": True, "unit": True, "bins": 8}, "dur_cond": False})
    pc = cfg["model"]["pos_cond"]
    assert pc["unit"] is True and pc["encoding"] == "raised_cosine" and pc["bins"] == 8
    # dur_agc（長さだけ）
    cfg = {"model": {"pos_cond": dict(new_default), "dur_cond": False}}
    align_pos_cond_to_ckpt(cfg, {"pos_cond": {"phoneme": False, "unit": False, "bins": 8}, "dur_cond": True})
    assert cfg["model"]["dur_cond"] is True and cfg["model"]["pos_cond"]["phoneme"] is False
    # 位置の条件ができる前の ckpt（config に pos_cond が無い）→ 全部 off。同じなら何もしない
    cfg = {"model": {"pos_cond": dict(new_default)}}
    align_pos_cond_to_ckpt(cfg, {"hidden": 256})
    assert cfg["model"]["pos_cond"]["phoneme"] is False and cfg["model"]["dur_cond"] is False
    assert align_pos_cond_to_ckpt(cfg, {"pos_cond": {"phoneme": False, "unit": False, "bins": 8}}) is None
    # 新しい既定で始めた run（sinusoidal）を同じ既定で再開 → そのまま。bins の違いは構造に関係しない
    cfg = {"model": {"pos_cond": dict(new_default, bins=4), "dur_cond": False}}
    assert align_pos_cond_to_ckpt(cfg, {"pos_cond": {"phoneme": True, "unit": False, "bins": 8,
                                                     "encoding": "sinusoidal"}, "dur_cond": False}) is None


def test_half_batch_has_no_tiny_batches():
    """速度対策の腕（最大 120000 フレーム・64 本、30 秒超は除外）でも 16 本未満のバッチを作らない。"""
    fc = [f for f in _lengths() if f <= 30 * FR]
    w = [1.0] * len(fc)
    s = FrameBasedBatchSampler(fc, 120000, 64, shuffle=True, weights=w, min_items=16)
    for _ in range(10):
        g = torch.Generator().manual_seed(s.seed + s.epoch)
        idxs = torch.multinomial(torch.as_tensor(w), len(fc), replacement=True, generator=g).tolist()
        for b in s._build(idxs):
            assert 16 <= len(b) <= 64 and len(b) * max(fc[i] for i in b) <= 120000
        list(iter(s))


def test_lr_factor_follows_diffsinger_pace():
    from train import lr_factor
    assert lr_factor(0, 0.8, 5000) == 1.0 and lr_factor(123456, 1.0, 5000) == 1.0
    assert abs(lr_factor(5000, 0.8, 5000) - 0.8) < 1e-12
    assert abs(lr_factor(20000, 0.8, 5000) - 0.8 ** 4) < 1e-12           # 0.41
    assert abs(lr_factor(2500, 0.8, 5000) - 0.8 ** 0.5) < 1e-12          # 段にならず、なめらかに下がる


def test_train_recipe_is_recorded_and_compared_on_resume():
    from train import recipe_changes, train_recipe
    cfg = {"train": {"lr": 8e-4, "max_batch_frames": 240000, "max_batch_size": 128, "min_batch_items": 16,
                     "lr_decay": 1.0, "lr_decay_steps": 5000, "amp": "fp32"},
           "gan": {"d_lr": 2e-4}, "data": {"max_sec": 30}}
    saved = train_recipe(cfg)
    assert saved["lr_decay_steps"] is None                                # 一定なら刻みは関係ない
    assert recipe_changes(train_recipe(cfg), saved) == {}
    assert recipe_changes(train_recipe(cfg), None) is None                # 記録の無い古い ckpt
    fast = {**cfg, "train": {**cfg["train"], "max_batch_frames": 120000, "max_batch_size": 64,
                             "lr_decay": 0.8, "amp": "bf16"}}
    ch = recipe_changes(train_recipe(fast), saved)
    assert ch["max_batch_frames"] == (240000, 120000) and ch["amp"] == ("fp32", "bf16")
    assert ch["lr_decay"] == (1.0, 0.8) and ch["lr_decay_steps"] == (None, 5000)
    assert ch["amp_fp32_io"] == (None, True)                              # bf16 の中身も記録する
    plain = {**fast, "train": {**fast["train"], "amp_fp32_io": False}}
    assert recipe_changes(train_recipe(plain), train_recipe(fast)) == {"amp_fp32_io": (True, False)}


def test_excitation_changes_on_resume():
    """共通の 20k の ckpt を写して noise_ratio だけ変えて続ける A/B: 変えた設定が分かる。"""
    from train import excitation_changes
    ck = {"n_harm": 256, "noise_ratio": 0.05, "exc_scale": 0.15, "harm_decay": 1.0}
    cfg = {"excitation": {"n_harm": 256, "noise_ratio": 0.2, "exc_scale": 0.15, "harm_decay": 1.0}}
    assert excitation_changes(cfg, ck) == {"noise_ratio": (0.05, 0.2)}
    cfg["excitation"]["noise_ratio"] = 0.05
    assert excitation_changes(cfg, ck) == {}
    assert excitation_changes({"excitation": {}}, {"noise_ratio": 0.05, "n_harm": 256}) == {"n_harm": (256, 50)}


def test_old_runs_resume_strictly_with_the_new_default_config():
    """既定を「音素内の位置・sinusoidal」にした後でも、2026-10-16 の A/B の各 run（位置なし / 旧符号化の
    音素+単位 / 長さだけ）の ckpt が strict に読める。新しい既定は DiffSinger 型の埋め込みを持つ。"""
    import copy
    import yaml
    from preprocess.vocab import Vocab
    from train import align_pos_cond_to_ckpt, build_model, ckpt_config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = yaml.safe_load(open(os.path.join(root, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    base["model"]["n_speakers"] = 3
    pc = base["model"]["pos_cond"]
    assert pc["phoneme"] is True and pc["unit"] is False and pc["encoding"] == "sinusoidal"
    vocab = Vocab.load()
    fresh, _ = build_model(copy.deepcopy(base), vocab)
    assert fresh.pos_cond.stretch_embed is not None and fresh.pos_cond.proj is None
    old_runs = {"base_agc": {"phoneme": False, "unit": False},
                "pos_agc": {"phoneme": True, "unit": True},
                "dur_agc": {"phoneme": False, "unit": False, "dur": True}}
    for name, o in old_runs.items():
        old_cfg = copy.deepcopy(base)
        old_cfg["model"]["pos_cond"] = {"phoneme": o["phoneme"], "unit": o["unit"], "bins": 8,
                                        "nucleus": pc["nucleus"]}           # encoding の無い旧 config
        old_cfg["model"]["dur_cond"] = o.get("dur", False)
        old, arch = build_model(old_cfg, vocab)
        ck_cfg = ckpt_config(old_cfg, arch, vocab)
        ck_cfg["pos_cond"].pop("encoding")                                   # 旧 ckpt には encoding が無い
        cfg = copy.deepcopy(base)
        assert align_pos_cond_to_ckpt(cfg, ck_cfg) is not None, name
        new, _ = build_model(cfg, vocab)
        new.load_state_dict(old.state_dict())                                # strict: 構造が一致


def test_gpu_spans_mark_the_forward_sections():
    """GpuSpans（CPU では perf_counter）: 励起（擬似 mel）などの区間が forward の中で切られる。"""
    from train import GpuSpans, _forward_flow_gan
    model, b = _tiny_model_and_batch()
    sp = GpuSpans("cpu")
    sp.mark("start")
    l = _forward_flow_gan(model, b, "l2", spans=sp)
    l["flow"].backward()
    sp.mark("bwd")
    d = sp.pop()
    assert list(d) == ["enc", "exc", "net", "bwd"] and all(v >= 0 for v in d.values())
    assert sp.pop() == {}                                 # marks are cleared
    torch.manual_seed(1)
    a = _forward_flow_gan(model, b, "l2")["flow"].item()  # spans=None: same numbers, no marks
    torch.manual_seed(1)
    c = _forward_flow_gan(model, b, "l2", spans=GpuSpans("cpu"))["flow"].item()
    assert a == c


def test_velocity_net_keeps_fp32_islands_under_bf16():
    """bf16 の autocast の中でも、速度場ネットの出力は fp32 で、fp32 で回した値に近い。autocast の外は元のまま。"""
    from leapsinger.modules.backbones.dilated_conv import DilatedConvDenoiser
    torch.manual_seed(0)
    net = DilatedConvDenoiser(mel_bins=16, ch=32, dim_cond=24, dilations=[1, 2, 4, 8]).eval()
    torch.nn.init.normal_(net.out[-1].weight, std=0.05)
    x, t, cond = torch.rand(2, 16, 80) * 2 - 1, torch.rand(2), torch.randn(2, 24, 80)
    with torch.no_grad():
        ref = net(x, t, cond)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = net(x, t, cond)
            plain = net.in_conv(x)                                       # 比較: 普通の層は bf16 になる
    assert ref.dtype == torch.float32 and y.dtype == torch.float32 and plain.dtype == torch.bfloat16
    assert float((y - ref).norm() / ref.norm()) < 0.02
    net.fp32_io = False                                                  # train.amp_fp32_io: false = 全部 bf16
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert net(x, t, cond).dtype == torch.bfloat16


def test_code_fingerprint_changes_with_the_code():
    """同じコマンドでもコードが変われば指紋が変わる（再開時に ckpt と比べる）。"""
    import shutil
    from train import code_fingerprint
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "ls")
        os.makedirs(os.path.join(root, "leapsinger", "modules"))
        for rel, text in (("train.py", "a = 1\n"), ("dataset.py", ""), ("infer.py", ""),
                          (os.path.join("leapsinger", "modules", "x.py"), "b = 2\n"),
                          ("README.md", "doc\n")):
            open(os.path.join(root, rel), "w").write(text)
        a = code_fingerprint(root)
        assert a["commit"] is None and len(a["sha1"]) == 10                 # git の外: commit なし
        open(os.path.join(root, "README.md"), "w").write("changed\n")       # 学習に効かないファイル
        assert code_fingerprint(root)["sha1"] == a["sha1"]
        open(os.path.join(root, "leapsinger", "modules", "x.py"), "w").write("b = 3\n")
        assert code_fingerprint(root)["sha1"] != a["sha1"]
        shutil.rmtree(os.path.join(root, "leapsinger"))
        assert code_fingerprint(root)["sha1"] != a["sha1"]


def test_bf16_autocast_forward_backward_matches_fp32():
    """train.amp=bf16 の経路: 損失は fp32 とほぼ同じ、逆伝播が通り、勾配が有限（CPU の autocast で確認）。"""
    from train import _forward_flow_gan
    model, b = _tiny_model_and_batch()
    out = {}
    for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
        torch.manual_seed(1)                              # same t / excitation noise for both
        l = _forward_flow_gan(model, b, "l2", amp_dtype=dt)
        model.zero_grad()
        (l["flow"] + 0.5 * l["mel_l1"]).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(g).all()
        assert l["flow"].dtype == torch.float32 and l["x1_pred_n"].dtype == torch.float32
        out[name] = (l["flow"].item(), l["mel_l1"].item())
    assert abs(out["bf16"][0] - out["fp32"][0]) < 0.05 * abs(out["fp32"][0]) + 1e-3
    assert abs(out["bf16"][1] - out["fp32"][1]) < 0.05 * abs(out["fp32"][1]) + 1e-3
