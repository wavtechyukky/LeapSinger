"""英語・韓国語・中国語: ラベルの変換（tools/convert_labels.py）、前処理の from_lab と --check とレシピの phonemes、
レシピと言語ごとの設定のつじつま、英語の辞書の shard で train.py を通した起動。"""
import glob
import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("LEAPSINGER_EXC_COMPILE", "0")

LANG_CONFIGS = ("en_teacher", "ko_teacher", "ko_teacher_5spk", "ko_mechol", "ko_mechol_sodasoo", "zh_teacher")


def _recipes():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "recipes", "*.yaml"))):
        out[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(open(p, encoding="utf-8"))
    return out


# ── ラベルの変換 ─────────────────────────────────────────────────────────────

def test_opencpop_conversion_writes_contiguous_labs_and_splits_the_apical_i(tmp_path):
    import convert_labels as cl
    lines = ["a1|字|z i sh i i AP|x|x|0.1 0.2 0.1 0.3 0.4 0.25|0 0 0 0 1 0",       # 字 = z i0、是 = sh ir（スラーで続く）
             "a2|爱你|ai n i y i SP|x|x|0.5 0.1 0.2 0.1 0.3 0.2|0 0 0 0 0 0",       # 声母の無い ai、ni・yi は i のまま
             "b1|吃|ch i SP|x|x|0.2 0.6 0.3|0 0 0"]
    src = tmp_path / "transcriptions.txt"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "train.txt").write_text(lines[0] + "\n" + lines[1] + "\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text(lines[2] + "\n", encoding="utf-8")
    n = cl.convert_opencpop(str(src), str(tmp_path / "out"),
                            {"train": str(tmp_path / "train.txt"), "test": str(tmp_path / "test.txt")}, split_i=True)
    assert n == {"train": 2, "test": 1}
    rows = [ln.split() for ln in open(tmp_path / "out" / "train" / "a1.lab", encoding="utf-8")]
    assert [r[2] for r in rows] == ["z", "i0", "sh", "ir", "ir", "AP"]
    assert [int(r[0]) for r in rows[1:]] == [int(r[1]) for r in rows[:-1]]                 # 隙間なく続く
    assert int(rows[-1][1]) == round(1.35e7)
    assert [r.split()[2] for r in open(tmp_path / "out" / "train" / "a2.lab")] == ["ai", "n", "i", "y", "i", "SP"]
    assert [r.split()[2] for r in open(tmp_path / "out" / "test" / "b1.lab")] == ["ch", "ir", "SP"]
    plain = cl.convert_opencpop(str(src), str(tmp_path / "plain"))                           # 分けない・一覧なし
    assert plain == {"": 3}
    assert [r.split()[2] for r in open(tmp_path / "plain" / "a1.lab")] == ["z", "i", "sh", "i", "i", "AP"]
    with pytest.raises(ValueError):                                                         # どの一覧にも無い
        cl.convert_opencpop(str(src), str(tmp_path / "x"), {"train": str(tmp_path / "train.txt")})


def test_diffsinger_csv_conversion(tmp_path):
    import convert_labels as cl
    src = tmp_path / "transcriptions.csv"
    src.write_text("name,ph_seq,ph_dur\nseg0,SP ay SP,0.25 0.5 0.125\n", encoding="utf-8")
    assert cl.convert_csv(str(src), str(tmp_path / "out")) == 1
    rows = [ln.split() for ln in open(tmp_path / "out" / "seg0.lab", encoding="utf-8")]
    assert rows == [["0", "2500000", "SP"], ["2500000", "7500000", "ay"], ["7500000", "8750000", "SP"]]


def test_lab_conversion_tidies_fills_gaps_and_splits_at_pauses(tmp_path):
    import soundfile as sf
    import convert_labels as cl
    # 揺れ（0.05 ms の隙間・重なり）はそろえ、音声の終わりを超えた分は切る
    segs = cl.tidy_segments([(0.0, 1.0, "SP"), (1.00005, 2.0, "a"), (1.99995, 3.0, "N"), (3.0, 9.0, "SP")], 4.0)
    assert segs == [(0.0, 1.0, "SP"), (1.0, 2.0, "a"), (2.0, 3.0, "N"), (3.0, 4.0, "SP")]
    # 有声の隙間 → 前後に分ける、短い無声の隙間 → cl、長い無声の隙間 → そのまま
    sr = 16000
    t = np.arange(int(1.6 * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio[int(0.70 * sr):int(1.30 * sr)] = 0.0                                              # 0.7–1.3 s は無音
    rows = [(0.0, 0.3, "a"), (0.4, 0.7, "N"), (0.77, 0.82, "kk"), (0.82, 1.0, "a"), (1.2, 1.6, "SP")]
    out, n = cl.fill_gaps(rows, audio, sr)
    assert n == {"split": 1, "cl": 1, "kept": 0}                                            # 1.0–1.2 は SP の隣
    assert [p for _s, _e, p in out] == ["a", "N", "cl", "kk", "a", "SP"]
    assert out[0][1] == pytest.approx(0.35) and out[1][0] == pytest.approx(0.35)
    assert out[2][:2] == (0.7, 0.77)
    rows2 = [(0.0, 0.3, "a"), (0.4, 0.5, "kk"), (0.5, 0.6, "a")]
    audio2 = audio.copy()
    audio2[int(0.3 * sr):int(0.4 * sr)] = 0.0
    assert cl.fill_gaps(rows2, audio2, sr)[1] == {"split": 0, "cl": 0, "kept": 1}           # 100 ms の無声は残す
    # 休みの真ん中で片に切る（最後の短い片は前につなぐ）。片ごとの wav・lab は 0 秒から
    lab = tmp_path / "song.lab"
    lab.write_text("".join(f"{int(a * 1e7)} {int(b * 1e7)} {p}\n" for a, b, p in
                           [(0, 1, "SP"), (1, 3, "a"), (3, 4, "SP"), (4, 6, "o"), (6, 6.5, "AP"), (6.5, 7, "i"), (7, 8, "SP")]))
    sf.write(str(tmp_path / "song.wav"), np.zeros(8 * sr, np.float32), sr)
    names = cl.convert_lab(str(lab), str(tmp_path / "song.wav"), str(tmp_path / "out"), split_sec=3.0)
    assert names == ["song_s000", "song_s001"]                       # 3.5 s で切る。7.5 s で切ると残りが 0.5 s なのでつなぐ
    first = [ln.split() for ln in open(tmp_path / "out" / "song_s000.lab")]
    second = [ln.split() for ln in open(tmp_path / "out" / "song_s001.lab")]
    assert first[-1] == ["30000000", "35000000", "SP"] and second[0] == ["0", "5000000", "SP"]
    assert second[-1] == ["35000000", "45000000", "SP"]                                     # 7–8 s の SP
    assert sf.info(str(tmp_path / "out" / "song_s001.wav")).frames == int(4.5 * sr)


def test_labdir_conversion_splits_only_long_files_and_skips_names(tmp_path):
    import soundfile as sf
    import convert_labels as cl
    sr = 16000
    src = tmp_path / "db"
    (src / "wav").mkdir(parents=True)
    (src / "lab").mkdir()

    def song(name, sec, rows):
        sf.write(str(src / "wav" / f"{name}.wav"), np.zeros(int(sec * sr), np.float32), sr)
        (src / "lab" / f"{name}.lab").write_text("".join(f"{int(a * 1e7)} {int(b * 1e7)} {p}\n" for a, b, p in rows))

    # 長い 1 本（休みで切られる）、短い 1 本（末尾の SP が音声を超える → 切る）、除く 1 本、wav の無い 1 本
    # long: 3.5 s と 6.5 s の休みで切れるが、6.5 s の後は 1.3 s しかないので前の片につなぐ → 2 片
    song("long", 7.8, [(0, 1, "SP"), (1, 3, "a"), (3, 4, "SP"), (4, 6, "o"), (6, 7, "SP"), (7, 7.8, "i")])
    song("short", 2, [(0, 1, "a"), (1, 30, "SP")])
    song("bad", 2, [(0, 2, "a")])
    (src / "lab" / "nowav.lab").write_text("0 10000000 a\n")
    t = cl.convert_labdir(str(src), str(tmp_path / "out"), prefix="x_", split_sec=3.0, split_long_sec=5.0,
                          skip=["bad"])
    assert t == {"files": 2, "pieces": 3, "skipped": ["bad"], "no_wav": ["nowav"]}
    assert sorted(p.stem for p in (tmp_path / "out").glob("*.lab")) == ["x_long_s000", "x_long_s001", "x_short"]
    short = [ln.split() for ln in open(tmp_path / "out" / "x_short.lab")]
    assert short[-1] == ["10000000", "20000000", "SP"]                                    # 音声の長さ（2 s）で切った


def test_gtsinger_ipa_maps_to_coda_by_position():
    import convert_labels as cl
    from leapsinger.phonemes import DICT_DIR, load_phonemes
    ph = set(load_phonemes(os.path.join(DICT_DIR, "ko_coda.phonemes")))

    def coda(ipa):
        segs = [(i * 0.1, (i + 1) * 0.1, p) for i, p in enumerate(ipa.split())]
        out, st = cl.ipa_to_coda(segs)
        return " ".join(p for _s, _e, p in out), st

    # 없는 = ʌ m n ɯ n（鼻音化）: 母音の後で次が子音・休みの n・m は終声
    assert coda("ʌ m n ɯ n <SP>")[0] == "eo M n eu N SP"
    # 같아 = k͈ a tʰ a: 次が母音の tʰ は初声のまま。감옥 の終わりの k̚ は K
    assert coda("k͈ a tʰ a")[0] == "kk a t a"
    assert coda("k a m o k̚ <AP>")[0] == "g a m o K AP"
    # 하늘에 = h a n ɯ ɾ e、흘러 = h ɯ ɭ ɾ ʌ、물 = m u l: ɾ は初声 r、ɭ・l は終声 L
    assert coda("h a n ɯ ɾ e")[0] == "h a n eu r e"
    assert coda("h ɯ ɭ ɾ ʌ")[0] == "h eu L r eo"
    assert coda("m u l <SP>")[0] == "m u L SP"
    # 平音の終声（解放の印 ̚ が無い書き方）と、平音の有声の書き分け（ɡ・d）
    out, st = coda("ɯ k t ɐ ɡ i")
    assert out == "eu K d a g i" and st["coda"] == {"g->K": 1}
    # 의 = ɰ i → eu i、위 = ɥ i → w i、외 = ø → w e（区間を等分）、ㅐ = ɛ → e、/i/・長音・結合の弧は落とす
    assert coda("ɰ i ɥ i")[0] == "eu i w i"
    out, _ = cl.ipa_to_coda([(0.0, 0.2, "ø")])
    assert out == [(0.0, 0.1, "w"), (0.1, 0.2, "e")]
    assert coda("ɛ /i/ aː t͡ɕʰ a")[0] == "e i a ch a"
    assert coda("")[0] == ""
    out, st = coda("a ʒ a")                                                                 # 知らない記号は SP にして数える
    assert out == "a SP a" and st["unknown"] == {"ʒ": 1}
    # 写した先はすべて ko_coda の音素（SP・AP は前処理で pau）
    every = " ".join(cl._IPA_BASE) + " ø kʷ"
    assert set(coda(every)[0].split()) <= ph | {"SP", "AP"}


def test_gtsinger_conversion_joins_songs_excludes_segments_and_splits(tmp_path):
    import soundfile as sf
    import convert_labels as cl
    sr = 16000
    src = tmp_path / "KO-Test-1"
    for tech in ("Breathy", "Vibrato"):                                   # 同じ曲が 2 つの技法のフォルダに
        for k in range(3):
            d = src / tech / "노래" / "Control_Group"
            d.mkdir(parents=True, exist_ok=True)
            sf.write(str(d / f"{k:04d}.wav"), np.zeros(int(10.0 * sr), np.float32), sr)
            items = [{"word": "나", "ph": ["n", "a"], "ph_start": [0.0, 1.0], "ph_end": [1.0, 9.0]},
                     {"word": "<SP>", "ph": ["<SP>"], "ph_start": [9.0], "ph_end": [10.0]}]
            (d / f"{k:04d}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        (src / tech / "노래" / "Breathy_Group").mkdir(exist_ok=True)               # 技法のグループは読まない
    out = tmp_path / "out"
    t = cl.convert_gtsinger(str(src), str(out), "gts_x", exclude={("Vibrato", "노래", "0001")}, split_sec=15.0)
    assert (t["songs"], t["segments"], t["excluded"]) == (1, 5, 1)
    # 5 区間（10 s）+ つなぎ目 4 つ（0.3 s）= 51.2 s。15 s を超えた後の休みで切る → 3 片（最後の短い片は前につなぐ）
    names = sorted(p.stem for p in out.glob("*.lab"))
    assert names == ["gts_x_song00_s000", "gts_x_song00_s001", "gts_x_song00_s002"]
    assert t["max_piece_sec"] < 25.0
    total = sum(sf.info(str(out / f"{n}.wav")).duration for n in names)
    assert total == pytest.approx(51.2, abs=0.01)
    first = [ln.split() for ln in open(out / "gts_x_song00_s000.lab", encoding="utf-8")]
    assert [r[2] for r in first[:3]] == ["n", "a", "SP"] and first[0][0] == "0"
    rows = (out / "songs.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == ["id", "song", "techs", "segments", "sec", "pieces"]
    assert rows[1].split("\t")[:4] == ["gts_x_song00", "노래", "Breathy,Vibrato", "5"] and rows[1].endswith("\t3")


# ── 前処理: from_lab・--check・レシピの phonemes ──────────────────────────────

def _tiny_db(root):
    """wav 3 本（a, b, c）と、a・b だけにラベル（b には辞書に無い記号 zz）。"""
    import soundfile as sf
    os.makedirs(root / "wavs")
    os.makedirs(root / "labs")
    for s in ("a", "b", "c"):
        sf.write(str(root / "wavs" / f"{s}.wav"), np.zeros(44100, np.float32), 44100)
    (root / "labs" / "a.lab").write_text("0 3000000 SP\n3000000 6000000 vf\n6000000 10000000 aa\n", encoding="utf-8")
    (root / "labs" / "b.lab").write_text("0 5000000 SP\n5000000 12000000 zz\n", encoding="utf-8")


def _recipe(tmp_path, **kw):
    r = dict(name="tiny", phonemes="dict/en_full.phonemes", lab_unit="100ns", layout="from_lab",
             db_root=str(tmp_path), wav="wavs/{song}.wav", lab="labs/{song}.lab", phon_norm={"SP": "pau", "vf": "vf"})
    r.update(kw)
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(r, allow_unicode=True), encoding="utf-8")
    return str(p), r


def test_from_lab_layout_lists_only_labelled_songs(tmp_path):
    from preprocess.run import _enumerate_songs
    _tiny_db(tmp_path)
    _, r = _recipe(tmp_path)
    assert _enumerate_songs(r, str(tmp_path)) == ["a", "b"]
    r_abs = dict(r, lab=str(tmp_path / "labs" / "{song}.lab"))                              # 絶対パスのテンプレート
    assert _enumerate_songs(r_abs, "/nonexistent") == ["a", "b"]
    assert _enumerate_songs(dict(r, layout="type_split", wav_dir="wavs"), str(tmp_path)) == ["a", "b", "c"]


def test_check_reports_unknown_phonemes_and_overruns_without_writing(tmp_path, capsys):
    from preprocess.run import run_recipe
    _tiny_db(tmp_path)
    path, _ = _recipe(tmp_path)
    out_root = tmp_path / "data"
    assert run_recipe(path, out_root=str(out_root), check=True) == 1                        # a は OK、b は zz で NG
    log = capsys.readouterr().out
    assert "[UNKNOWN zz] b" in log and "phonemes not in the vocab: ['zz']" in log
    assert "labels run past the audio end (> 50 ms) in 1 songs, at most 0.20 s (b" in log  # 1.2 s のラベル・1.0 s の音声
    assert not out_root.exists()                                                            # 何も書かない
    # vf は既定の写像（vf → pau）でなく phon_norm で音素のまま。en_full に vf があるので a は通る。日本語の辞書では通らない
    from preprocess.vocab import Vocab
    assert run_recipe(path, out_root=str(out_root), check=True, vocab=Vocab.load()) == 0


# ── レシピと言語ごとの設定のつじつま ────────────────────────────────────────

def test_every_recipe_with_a_phoneme_list_has_a_language_pack_and_maps_into_it():
    from leapsinger.phonemes import find_lang, load_phonemes
    n = 0
    for name, r in _recipes().items():
        if not r.get("phonemes"):
            continue                                                                        # 日本語（既定の一覧）
        n += 1
        ph = load_phonemes(os.path.join(ROOT, r["phonemes"]))
        assert find_lang(ph) is not None, name
        for src, dst in (r.get("phon_norm") or {}).items():
            assert dst in ph, (name, src, dst)
    assert n == 28                                                                          # NUS-48E の 12 本は公開版に無い


def test_language_configs_match_the_recipes():
    from leapsinger.phonemes import find_lang, load_phonemes
    from preprocess.vocab import Vocab
    from train import cons_weight_settings, pos_cond_config, train_recipe
    recipes = _recipes()
    for cfg_name in LANG_CONFIGS:
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", f"{cfg_name}.yaml"), encoding="utf-8"))
        spk = cfg["data"]["spk_map"]
        assert max(spk.values()) < cfg["model"]["n_speakers"], cfg_name
        assert set(cfg["data"].get("eval_dbs") or []) <= set(spk), cfg_name
        lists = {recipes[db]["phonemes"] for db in spk}                                     # レシピがあり、辞書が 1 つ
        assert len(lists) == 1, (cfg_name, lists)
        ph = load_phonemes(os.path.join(ROOT, lists.pop()))
        lang = find_lang(ph)
        pc = pos_cond_config(cfg, Vocab(ph))                                                # 核音素 = 言語パックの母音
        assert pc["nucleus_ids"] == sorted(i for i, p in enumerate(ph) if p in lang.vowels), cfg_name
        assert cfg["gan"]["enabled"] is False and cfg["excitation"]["x0_sigma"] == 0.0
        assert train_recipe(cfg)["cons_weight"] is None and cons_weight_settings(cfg["train"]) is None


# ── 英語の辞書の shard で train.py を通す ────────────────────────────────────

def _en_db(root, name):
    """en_full の音素の id を使った小さな shard（metadata に音素の一覧を持つ = 前処理と同じ）。"""
    from leapsinger.phonemes import DICT_DIR, load_phonemes
    ph = load_phonemes(os.path.join(DICT_DIR, "en_full.phonemes"))
    ids = [ph.index(p) for p in ("pau", "s", "aa", "vf", "ng", "pau")]
    fr = 172.265625
    d = os.path.join(str(root), name)
    os.makedirs(d)
    rng = np.random.default_rng(0)
    arrays, phrases = {}, {}
    for song, T in (("songA", 60), ("songA", 70), ("songB", 64), ("songC", 58)):
        n = f"{song}_{len([k for k in phrases if k.startswith(song)]):04d}"
        phrases[n] = T
        frames = np.array([5, 10, 20, 6, T - 46, 5])
        arrays[f"{n}|phoneme_ids"] = np.array(ids, np.int16)
        arrays[f"{n}|dur_sec"] = (frames / fr).astype(np.float32)
        arrays[f"{n}|f0_interp"] = np.full(T, 196.0, np.float32)
        arrays[f"{n}|uv"] = np.ones(T, np.float32)
        arrays[f"{n}|mel"] = rng.uniform(-11.0, 0.0, (128, T)).astype(np.float32)
    np.savez(os.path.join(d, "shard.npz"), **arrays)
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"frame_rate": fr, "hop_size": 256, "sample_rate": 44100, "phrases": phrases, "phonemes": ph}, f)
    return d, ph


def test_train_main_runs_on_an_english_shard(tmp_path, capsys):
    import train
    d1, ph = _en_db(tmp_path, "aidol")
    d2, _ = _en_db(tmp_path, "nus_adiz")
    argv = ["train", "--config", os.path.join(ROOT, "configs", "en_teacher.yaml"), "--data_dirs", d1, d2,
            "--run_name", "en", "--out_root", str(tmp_path / "log"), "--device", "cpu", "--max_updates", "2",
            "--set", "train.amp=fp32", "--set", "train.num_workers=0", "--set", "train.vocoder=null",
            "--set", "train.eval_interval=1000", "--set", "train.save_interval=1000", "--set", "train.eval_items=1",
            "--set", "train.num_steps=2", "--set", "data.eval_songs=1", "--set", "train.cons_weight=2.0"]
    old = sys.argv
    sys.argv = argv
    try:
        train.main()
    finally:
        sys.argv = old
    log = capsys.readouterr().out
    assert "言語パック: dict/en_full.yaml" in log and "[cons_weight] 子音 " in log
    ck = torch.load(tmp_path / "log" / "en" / "ckpt_000002.pt", map_location="cpu")
    assert ck["config"]["phonemes"] == ph and ck["config"]["n_phonemes"] == len(ph)
