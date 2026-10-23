"""言語パック（dict/<lang>.yaml、leapsinger/phonemes.py）: 読み込み、音素の一覧からの検索、子音の決め方、
書き間違いの検出。"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_every_pack_in_dict_loads_and_classifies_every_phoneme_once():
    from leapsinger.phonemes import DICT_DIR, load_lang
    packs = sorted(f[:-len(".yaml")] for f in os.listdir(DICT_DIR) if f.endswith(".yaml"))
    assert {"ja", "en"} <= set(packs)
    for name in packs:
        lp = load_lang(name)
        for p in lp.phonemes:                                                       # 母音・静音・子音のどれか 1 つ
            assert (p in lp.vowels) + (p in lp.silence) + (p in lp.consonants) == 1, (name, p)
        assert lp.phonemes[0] == "pau" and "pau" in lp.silence
        assert lp.steady <= lp.vowels


def test_find_lang_matches_the_phoneme_list_exactly():
    from leapsinger.phonemes import DICT_DIR, find_lang, load_lang, load_phonemes
    from preprocess.vocab import Vocab
    assert find_lang(Vocab.load().phonemes).name == "ja"
    en = load_phonemes(os.path.join(DICT_DIR, "en.phonemes"))
    assert find_lang(en) == load_lang("en")
    assert find_lang(en[:-1]) is None                                               # 1 つでも違えば別の一覧
    assert find_lang(["pau", "a"]) is None


def test_english_groups():
    from leapsinger.phonemes import load_lang
    en = load_lang("en")
    assert {"aa", "er", "ax"} <= en.vowels and "cl" in en.consonants
    assert en.consonant_class("s") == "ucons" and en.consonant_class("dh") == "vcons"
    assert en.consonant_class("cl") == "other" and not en.devoiced
    assert "ay" in en.vowels and "ay" not in en.steady                              # 二重母音は揺れの指標から外す


def _pack(tmp_path, body, phonemes="pau\na\nk\ng\n"):
    (tmp_path / "x.phonemes").write_text(phonemes, encoding="utf-8")
    p = tmp_path / "x.yaml"
    p.write_text("phonemes: x.phonemes\n" + body, encoding="utf-8")
    return str(p)


def test_load_lang_rejects_typos(tmp_path):
    from leapsinger.phonemes import load_lang
    ok = load_lang(_pack(tmp_path, "vowels: [a]\nsilence: [pau, sil]\nvoiceless: [k]\nvoiced: [g]\n"))
    assert ok.name == "x" and ok.consonants == {"k", "g"} and ok.consonant_ids() == {2, 3}
    for body in ("vowels: [a]\nsilence: [pau]\nvowel: [a]\n",                       # 知らないキー
                 "vowels: [a, o]\nsilence: [pau]\n",                                  # 一覧に無い母音
                 "vowels: [a]\n",                                                     # silence が無い
                 "vowels: [a, pau]\nsilence: [pau]\n",                                # 母音と静音の両方
                 "vowels: [a]\nsilence: [pau]\nvoiced: [a]\n",                        # 子音ではない
                 "vowels: [a]\nsilence: [pau]\nvoiced: [k]\nvoiceless: [k]\n"):      # 2 つの組にある
        with pytest.raises(ValueError):
            load_lang(_pack(tmp_path, body))


def test_preprocess_reads_phoneme_lists_with_the_same_function():
    import leapsinger.phonemes as lp
    import preprocess.vocab as pv
    assert pv.load_phonemes is lp.load_phonemes
