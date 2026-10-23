"""tools/loudness_tuner/app.py — データセットの見つけ方、shard_wav.npz からの試聴、エンドポイントの形。"""
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")                       # TestClient が使う
from fastapi.testclient import TestClient  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FR, HOP, SR = 172.265625, 256, 44100


def _load_app():
    spec = importlib.util.spec_from_file_location("loudness_tuner_app", os.path.join(ROOT, "tools", "loudness_tuner", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                   # pydantic が `from __future__ import annotations` の型名を引けるように
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tuner():
    return _load_app()


def _write_dataset(d, name="ritsu", with_wav=True, lead_zero_frames=0):
    """小さなデータセット: 2 フレーズ（p1 は 100 フレーム、p2 は 300 フレーム、どちらも pau・a・pau）。"""
    d.mkdir(parents=True, exist_ok=True)
    arrays, wavs, phrases = {}, {}, {}
    for pname, T in (("song_0000", 100), ("song_0001", 300)):
        rng = np.random.default_rng(T)
        mel = (-4.0 + 0.5 * rng.standard_normal((128, T))).astype(np.float32)
        dur = np.array([0.1, T / FR - 0.2, 0.1], np.float32)
        f0 = np.full(T, 220.0, np.float32)
        f0[:lead_zero_frames] = 0.0                     # 古いシャードの「頭の無音は f0 = 0」
        arrays[f"{pname}|mel"] = mel
        arrays[f"{pname}|phoneme_ids"] = np.array([0, 1, 0], np.int16)
        arrays[f"{pname}|dur_sec"] = dur
        arrays[f"{pname}|f0_interp"] = f0
        arrays[f"{pname}|uv"] = np.ones(T, np.float32)
        n = (T - lead_zero_frames) * HOP               # 頭の無音の分だけ短い wav
        wavs[f"{pname}|wav"] = (0.3 * np.sin(np.arange(n) * 0.05) * 32767).astype(np.int16)
        phrases[pname] = T
    np.savez(d / "shard.npz", **arrays)
    if with_wav:
        np.savez(d / "shard_wav.npz", **wavs)
    (d / "metadata.json").write_text(json.dumps(dict(
        phrases=phrases, phoneme_vocab={"pau": 0, "a": 1}, n_phonemes=2, sample_rate=SR, hop_size=HOP,
        frame_rate=FR, n_mels=128)), encoding="utf-8")
    return d


def test_scan_finds_the_dataset_itself_and_its_parent(tuner, tmp_path):
    ds = _write_dataset(tmp_path / "data" / "ritsu")
    c = TestClient(tuner.app)
    j = c.post("/scan", json={"path": str(ds)}).json()
    assert [d["name"] for d in j["dbs"]] == ["ritsu"]
    assert j["dbs"][0]["n_phrases"] == 2 and j["dbs"][0]["audio"] == {"n": 2, "source": "shard_wav"}
    j = c.post("/scan", json={"path": str(tmp_path / "data")}).json()   # 親フォルダ
    assert [d["dir"] for d in j["dbs"]] == [str(ds)]
    r = c.post("/scan", json={"path": str(tmp_path)})                   # 2 段上: データセットが無い
    assert r.status_code == 400 and "shard.npz" in r.json()["detail"]
    r = c.post("/scan", json={"path": str(tmp_path / "nothing")})
    assert r.status_code == 400


def test_config_names_the_training_implementation(tuner):
    j = TestClient(tuner.app).get("/config").json()
    assert j["agc_source"] == "leapsinger/loudness.py"
    assert j["yaml_keys"] == ["sigma_sec", "target_db", "k", "max_gain_db", "max_cut_db"]
    assert set(j["defaults"]) == set(j["yaml_keys"])


def test_compute_audio_and_mel(tuner, tmp_path):
    ds = _write_dataset(tmp_path / "ds")
    c = TestClient(tuner.app)
    j = c.post("/compute", json={"dir": str(ds), "page_size": 5}).json()
    assert j["total"] == 2 and j["audio_source"] == "shard_wav"
    it = j["items"][0]
    assert it["has_audio"] and it["T"] == 100 and len(it["E"]) == 100 and len(it["bounds"]) == 3
    assert it["bounds"][0]["pau"] and it["bounds"][1]["ph"] == "a"
    q = {"dir": str(ds), "name": "song_0000"}
    r = c.get("/audio", params=dict(q, which="norm"), headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content[:4] == b"RIFF" and len(r.content) == 100
    r = c.get("/audio", params=q)
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    info = c.get("/audio_info", params=q).json()
    assert info["n_samples"] == 100 * HOP and info["src"].startswith("shard_wav.npz:")
    r = c.get("/mel.png", params=dict(q, which="after"))
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = c.get("/audio", params={"dir": str(ds), "name": "no_such"})
    assert r.status_code == 404


def test_short_wav_is_front_padded_by_the_leading_silence(tuner, tmp_path):
    sf = pytest.importorskip("soundfile")
    ds = _write_dataset(tmp_path / "ds", lead_zero_frames=5)
    c = TestClient(tuner.app)
    r = c.get("/audio", params={"dir": str(ds), "name": "song_0000", "which": "orig"})
    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == SR and len(wav) == 100 * HOP
    assert not wav[:5 * HOP].any() and wav[5 * HOP:6 * HOP].any()


def test_dataset_without_audio_still_computes(tuner, tmp_path):
    ds = _write_dataset(tmp_path / "ds", with_wav=False)
    c = TestClient(tuner.app)
    j = c.post("/scan", json={"path": str(ds)}).json()
    assert j["dbs"][0]["audio"] == {"n": 0, "source": None}
    j = c.post("/compute", json={"dir": str(ds), "audio_only": False}).json()
    assert j["total"] == 2 and not any(it["has_audio"] for it in j["items"])
    j = c.post("/compute", json={"dir": str(ds), "audio_only": True}).json()
    assert j["total"] == 0


def test_stats_job_runs_and_caches(tuner, tmp_path, monkeypatch):
    import time
    ds = _write_dataset(tmp_path / "ds")
    monkeypatch.setattr(tuner, "CACHE_DIR", tmp_path / "cache")
    c = TestClient(tuner.app)
    j = c.post("/stats_start", json={"dir": str(ds)}).json()
    assert not j["cached"]
    for _ in range(100):
        p = c.get("/stats_progress", params={"job_id": j["job_id"]}).json()
        if p["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert p["state"] == "done", p
    assert p["result"]["n_phrases"] == 1                 # 歌唱部 1 秒未満の song_0000 は除外
    assert p["result"]["params"]["sigma_sec"] == tuner.AGC.AGC_DEFAULTS["sigma_sec"]
    assert list((tmp_path / "cache").glob("ritsu*.json")) == [] and list((tmp_path / "cache").glob("ds_*.json"))
    j2 = c.post("/stats_start", json={"dir": str(ds)}).json()
    assert j2["cached"] and j2["result"]["n_phrases"] == 1
