"""後方互換: v0.1.0 の形の ckpt（config に pos_cond・x0_sigma・loudness_agc が無い）が、新しい config と
--init_from の組み合わせでも、infer・export でも読める。"""
import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.vocab import Vocab  # noqa: E402
from train import align_pos_cond_to_ckpt, build_model, ckpt_config, load_init_checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V010_ONLY = ("arch", "n_phonemes", "phonemes", "hidden", "mel_bins", "mel_vmin", "mel_vmax", "backbone_ch",
             "n_cycles", "dilation_schedule", "n_speakers", "n_styles", "spk_dim", "flow_loss", "use_uv",
             "n_harm", "noise_ratio", "exc_scale", "harm_decay", "exc_hop", "hop", "sample_rate",
             "recon_weight", "infer_steps")                                  # v0.1.0 の ckpt_config のキー


def _small(cfg: dict) -> dict:
    """試験用に小さくした model（構造の規則は同じ）。"""
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(hidden=32, backbone_ch=32, n_cycles=1, n_speakers=3)
    cfg["excitation"]["n_harm"] = 8
    return cfg


def _v010_checkpoint(path: str):
    """v0.1.0 と同じ形の ckpt: pos_cond の無い config で作ったモデル、config のキーは当時のものだけ。"""
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    old_cfg = _small(base)
    old_cfg["model"].pop("pos_cond", None)
    old_cfg["model"].pop("dur_cond", None)
    old_cfg["excitation"].pop("x0_sigma", None)
    old_cfg["excitation"]["harm_decay"] = 1.0                                 # v0.1.0 の既定
    old_cfg.pop("data", None)
    vocab = Vocab.load()
    model, arch = build_model(old_cfg, vocab)
    ck_cfg = {k: v for k, v in ckpt_config(old_cfg, arch, vocab).items() if k in V010_ONLY}
    assert "pos_cond" not in ck_cfg and "x0_sigma" not in ck_cfg and "loudness_agc" not in ck_cfg
    torch.save({"model": model.state_dict(), "step": 50000, "config": ck_cfg, "n_speakers": 3}, path)
    return model, old_cfg, vocab


def test_init_from_an_old_checkpoint_aligns_the_structure_and_loads_strictly(tmp_path, capsys):
    path = str(tmp_path / "3speaker_gan2d.pth")
    old, _, vocab = _v010_checkpoint(path)
    new_cfg = _small(yaml.safe_load(open(os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), encoding="utf-8")))
    assert new_cfg["model"]["pos_cond"]["phoneme"] is True                   # 新しい既定は位置の条件 on
    ck = load_init_checkpoint(new_cfg, path)
    out = capsys.readouterr().out
    assert "[init_from] 位置の条件を" in out and "harm_decay 1.0 -> 1.5" in out
    assert new_cfg["model"]["pos_cond"]["phoneme"] is False and new_cfg["model"]["dur_cond"] is False
    model, _ = build_model(new_cfg, vocab)
    model.load_state_dict(ck["model"])                                        # strict
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), old.state_dict().values()))
    assert float(model.harm_decay) == 1.5                                    # 励起は今の設定（重みを持たない）


def test_align_pos_cond_is_a_no_op_for_a_matching_checkpoint():
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    cfg = _small(base)
    vocab = Vocab.load()
    _, arch = build_model(copy.deepcopy(cfg), vocab)
    assert align_pos_cond_to_ckpt(copy.deepcopy(cfg), ckpt_config(cfg, arch, vocab)) is None


def test_infer_and_export_accept_an_old_checkpoint(tmp_path):
    from export.cli import _validate
    from infer import infer_mel, load_acoustic
    path = str(tmp_path / "3speaker_gan2d.pth")
    _v010_checkpoint(path)
    model, cfg = load_acoustic(path)
    assert "pos_cond" not in cfg and not any(k.startswith("pos_cond") for k in model.state_dict())   # 位置の条件の無い構造
    T = 60
    item = dict(ph_ids=np.array([0, 5, 6, 0]), ph_durs=np.array([10, 20, 20, 10]),
                f0_logf0=np.log2(np.full(T, 220.0, np.float32)), uv=np.ones(T, np.float32), spk_id=1)
    mel = infer_mel(model, item, num_steps=1)
    assert tuple(mel.shape) == (cfg["mel_bins"], T) and np.isfinite(mel).all()
    _validate(cfg, model, "diffsinger", "embed")                              # 書き出しの前提も通る
