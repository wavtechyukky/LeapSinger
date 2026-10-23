"""言語パック（dict/<lang>.yaml）と、子音のフレームの損失の重み。

コードには言語ごとの音素の名前を書かない。言語パックが音素の一覧（dict/<lang>.phonemes）を参照し、
一覧だけでは表せない音素の種類（母音・静音、評価用の子音の種類）を持つ。モデルや shard の音素の一覧と
中身が一致する言語パックを dict/ から探して使う（find_lang）。

子音 = 母音（vowels）にも静音（silence）にも入らない音素すべて。子音の損失の重み（train.cons_weight）、
評価（tools/eval_ab.py）の母音と子音の分け方、ポストフィルタ（leapsinger/postfilter.py）が使う。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

DICT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dict")
_KEYS = {"phonemes", "vowels", "silence", "devoiced", "voiceless", "voiced", "steady_vowels", "nucleus"}
CLASSES = ("devoiced", "ucons", "vcons", "other")       # 子音の種類（評価用）: 無声化した母音・無声子音・有声子音・その他


@dataclass(frozen=True)
class LangPack:
    name: str                                            # 言語パックのファイル名（拡張子なし）。例: ja
    phonemes: tuple                                      # 音素の一覧（id の順）
    vowels: frozenset                                    # 母音
    silence: frozenset                                   # 静音（一覧に無い名前も書ける。例: 前処理で pau になる sil）
    devoiced: frozenset = frozenset()                    # ── 評価用（無くてもよい）: 子音の種類別の表
    voiceless: frozenset = frozenset()
    voiced: frozenset = frozenset()
    steady_vowels: frozenset = frozenset()               # 長い母音の中ほどの揺れを見る母音（空なら vowels）
    nucleus: tuple = ()                                  # 位置の条件 unit の既定の核音素（空なら vowels）

    @property
    def consonants(self) -> frozenset:
        """子音 = 母音にも静音にも入らない音素すべて。"""
        return frozenset(p for p in self.phonemes if p not in self.vowels and p not in self.silence)

    @property
    def steady(self) -> frozenset:
        return self.steady_vowels or self.vowels

    def consonant_class(self, ph: str) -> str:
        """子音の種類（評価用）: devoiced / ucons / vcons / other（どの組にも入らない子音。cl など）。"""
        if ph in self.devoiced:
            return "devoiced"
        if ph in self.voiceless:
            return "ucons"
        if ph in self.voiced:
            return "vcons"
        return "other"

    def consonant_ids(self, phonemes=None) -> frozenset:
        """音素の一覧（id の順。既定はこの言語パックの一覧）のうち、子音の id。"""
        cons = self.consonants
        return frozenset(i for i, p in enumerate(self.phonemes if phonemes is None else phonemes) if p in cons)


def load_phonemes(path: str) -> list[str]:
    """Read a phoneme-list file -> ordered list. One phoneme per line; blank lines and text
    after '#' are ignored. The order defines the id (first entry = id 0)."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if ln:
                out.append(ln)
    if not out:
        raise ValueError(f"no phonemes found in {path}")
    if out[0] != "pau":
        raise ValueError(f"{path}: id 0 must be 'pau' (silence/pad), got {out[0]!r}")
    if len(set(out)) != len(out):
        dup = sorted({p for p in out if out.count(p) > 1})
        raise ValueError(f"{path}: duplicate phonemes {dup}")
    return out


def load_lang(name_or_path: str) -> LangPack:
    """言語パックを読む。name_or_path は dict/ の言語の名前（ja）か yaml のパス。音素の一覧は yaml の
    phonemes（yaml からの相対パス）。書き間違いを防ぐため、知らないキーや、一覧に無い音素の名前（silence 以外）は
    エラーにする。"""
    path =name_or_path if name_or_path.endswith((".yaml", ".yml")) else os.path.join(DICT_DIR, f"{name_or_path}.yaml")
    d = yaml.safe_load(open(path, encoding="utf-8")) or {}
    bad = sorted(set(d) - _KEYS)
    if bad:
        raise ValueError(f"{path}: 知らないキー {bad}（使えるのは {sorted(_KEYS)}）")
    for k in ("phonemes", "vowels", "silence"):
        if not d.get(k):
            raise ValueError(f"{path}: {k} がありません")
    phonemes = tuple(load_phonemes(os.path.join(os.path.dirname(path), d["phonemes"])))
    have = set(phonemes)
    sets = {k: [str(p) for p in (d.get(k) or [])] for k in _KEYS - {"phonemes"}}
    for k, names in sets.items():
        missing = [p for p in names if p not in have]
        if missing and k != "silence":
            raise ValueError(f"{path}: {k} の {missing} が {d['phonemes']} にありません")
    lp = LangPack(name=os.path.splitext(os.path.basename(path))[0], phonemes=phonemes,
                  vowels=frozenset(sets["vowels"]), silence=frozenset(sets["silence"]),
                  devoiced=frozenset(sets["devoiced"]), voiceless=frozenset(sets["voiceless"]),
                  voiced=frozenset(sets["voiced"]), steady_vowels=frozenset(sets["steady_vowels"]),
                  nucleus=tuple(sets["nucleus"]))
    if lp.vowels & lp.silence:
        raise ValueError(f"{path}: vowels と silence の両方にある音素 {sorted(lp.vowels & lp.silence)}")
    subs = {"devoiced": lp.devoiced, "voiceless": lp.voiceless, "voiced": lp.voiced}
    for k, s in subs.items():
        if s - lp.consonants:
            raise ValueError(f"{path}: {k} の {sorted(s - lp.consonants)} は子音ではありません（vowels か silence にある）")
    if (lp.devoiced & lp.voiceless) or (lp.devoiced & lp.voiced) or (lp.voiceless & lp.voiced):
        raise ValueError(f"{path}: devoiced・voiceless・voiced の 2 つ以上にある音素があります")
    return lp


_FOUND: dict = {}


def find_lang(phonemes):
    """音素の一覧（id の順）と中身が同じ言語パックを dict/ から探す。無ければ None。"""
    key = tuple(phonemes)
    if key not in _FOUND:
        _FOUND[key] = None
        for f in sorted(os.listdir(DICT_DIR)):
            if f.endswith(".yaml"):
                lp = load_lang(os.path.join(DICT_DIR, f))
                if lp.phonemes == key:
                    _FOUND[key] = lp
                    break
    return _FOUND[key]


def consonant_frame_weights(ph_ids, ph_frames, cons_ids, n_frames: int, weight: float,
                            next_frames: int = 0, pau_id: int = 0) -> np.ndarray:
    """フレームごとの損失の重み [n_frames]（float32）。子音（cons_ids）のフレームと、その直後の音素（無音以外）の頭
    next_frames フレームを weight、ほかを 1 にする。ph_frames は音素ごとのフレーム数（合計 = n_frames）。"""
    w = np.ones(n_frames, dtype=np.float32)
    fb = np.concatenate([[0], np.cumsum(np.asarray(ph_frames, dtype=np.int64))]).clip(max=n_frames)
    ids = [int(p) for p in ph_ids]
    for k, p in enumerate(ids):
        if p not in cons_ids:
            continue
        s, e = int(fb[k]), int(fb[k + 1])
        w[s:e] = weight
        if next_frames > 0 and k + 1 < len(ids) and ids[k + 1] != pau_id:
            w[e:min(e + int(next_frames), int(fb[k + 2]))] = weight
    return w
