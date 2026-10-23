"""export: --speaker embed のときに書く {model}.speakers.json（話者ごとの spk_embed と音素の一覧）。"""
import copy
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import spk_embed as spk  # noqa: E402
from preprocess.vocab import Vocab  # noqa: E402
from train import build_model, ckpt_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiny_multispk():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "3speaker_gan2d.yaml"), encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["model"].update(hidden=32, backbone_ch=32, n_cycles=1, n_speakers=3)
    cfg["excitation"]["n_harm"] = 8
    vocab = Vocab.load()
    model, arch = build_model(cfg, vocab)
    return model, ckpt_config(cfg, arch, vocab)


def test_speakers_json_has_every_speaker_vector_and_the_phoneme_list(tmp_path):
    model, cfg = _tiny_multispk()
    assert spk.has_speakers(model)
    path = str(tmp_path / "m.speakers.json")
    table = spk.write_speakers_json(model, cfg, path)
    j = json.load(open(path, encoding="utf-8"))
    assert j == table
    assert j["dim"] == 32 and sorted(j["speakers"]) == ["0", "1", "2"]
    assert j["phonemes"] == cfg["phonemes"] and j["phonemes"][0] == "pau"
    for i in range(3):
        assert np.allclose(np.asarray(j["speakers"][str(i)], np.float32), spk.speaker_vector(model, i))
    # 1 人の話者を焼き込んだときに足されるものと同じ（_spk_add）
    with torch.no_grad():
        idx = torch.tensor([1])
        v = model.spk_proj(model.spk_bank(idx))[0].numpy()
    assert np.allclose(np.asarray(j["speakers"]["1"], np.float32), v)
