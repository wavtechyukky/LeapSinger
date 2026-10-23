"""音素の長さだけを持つラベルを、前処理が読む区間ごとの lab（HTK、100 ns の整数 `start end phoneme`）に変換する。
元のデータには触らず、--out のフォルダに書く。

  # Opencpop（segments/transcriptions.txt: id|歌詞|音素|音符|音符の長さ|音素の長さ|スラー）。
  # --split に学習用・評価用の一覧を渡すと、<out>/train/、<out>/test/ に分けて書く
  python tools/convert_labels.py opencpop --src /path/to/segments/transcriptions.txt --out <out> \
      --split train=/path/to/segments/train.txt test=/path/to/segments/test.txt --split_i

  # DiffSinger の transcriptions.csv（name, ph_seq, ph_dur）
  python tools/convert_labels.py csv --src /path/to/transcriptions.csv --out <out>

  # 曲ごとの HTK lab と wav（Sds-Korean-Singing-Dataset など）。境界の小さな隙間・重なりをそろえ、音声の終わりで切る。
  # --split_sec を渡すと、長い音声を休み（SP・AP など）の真ん中で、その長さ以上の片に切る（<name>_s000.wav/.lab …）
  # --fill_gaps: 音素の間の 5 ms 以上の隙間を、音声の有声・無声を見て埋める（SODAsoo の caffeinism に 57 個）
  python tools/convert_labels.py lab --src song.lab --wav song.wav --out <out> [--name NAME] [--split_sec 60] [--fill_gaps]

  # wav/ と lab/ に分かれたフォルダをまとめて（Mechol の 52 分など）。2 分を超えるものだけ 1 分ほどの片に切る
  python tools/convert_labels.py labdir --src /path/to/52min --out <out> --prefix mechol_ [--skip 2_46]

  # GTSinger の韓国語（IPA）。歌い手 1 人の Control_Group だけを、曲ごとに 1 本につないで Team CODA の lab にする
  # （つないだ曲は 10 分を超えることがあり、RMVPE がメモリを使い切るので --split_sec で 1 分ほどの片に切る）
  python tools/convert_labels.py gtsinger --src /path/to/GTSinger/Korean/KO-Tenor-1 --out <out> --prefix gts_ko_t1 \
      --split_sec 45 [--exclude <tsv>]

--split_i: Opencpop の i を、直前の声母で舌尖母音 ir（zh ch sh r）と i0（z c s）に分ける（DiffSinger の opencpop-strict と
同じ）。メリスマで韻母を繰り返した音素（スラー 1）は、同じ音節の声母を引き継ぐ。
"""
from __future__ import annotations

import argparse
import csv
import os

OPENCPOP_INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
                     "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"}
RETROFLEX, DENTAL = {"zh", "ch", "sh", "r"}, {"z", "c", "s"}


def split_apical_i(phonemes, slurs=None):
    """Opencpop の音素の並び → i を ir / i0 に分けた並び。slurs（0/1）が 1 の音素は前の音節の続き。"""
    out, initial = [], None
    for k, p in enumerate(phonemes):
        cont = bool(slurs[k]) if slurs is not None else False
        if p in OPENCPOP_INITIALS:
            initial = p
        elif p in ("SP", "AP"):
            initial = None
        else:                                                    # 韻母
            if not cont and (k == 0 or phonemes[k - 1] not in OPENCPOP_INITIALS):
                initial = None                                   # 声母の無い音節（ai, an, er ...）
            if p == "i" and initial in RETROFLEX:
                p = "ir"
            elif p == "i" and initial in DENTAL:
                p = "i0"
        out.append(p)
    return out


def lab_lines(phonemes, durations) -> list[str]:
    """音素と長さ（秒）→ lab の行。境界は長さの累積を 100 ns に丸めたもの（区間は隙間なく続く）。"""
    lines, t = [], 0.0
    start = 0
    for p, d in zip(phonemes, durations):
        t += float(d)
        end = int(round(t * 1e7))
        lines.append(f"{start} {end} {p}")
        start = end
    return lines


def _write(path: str, lines) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def convert_opencpop(src: str, out: str, splits: dict | None = None, split_i: bool = False) -> dict:
    """→ {分け方: 書いた数}。splits = {名前: id の一覧のファイル}（Opencpop の train.txt / test.txt は
    transcriptions.txt と同じ形式の行）。無ければ out に全部書く。"""
    where = {}
    for name, path in (splits or {}).items():
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                where[ln.split("|", 1)[0].strip()] = name
    n = {}
    for ln in open(src, encoding="utf-8"):
        f = ln.rstrip("\n").split("|")
        if len(f) < 7:
            continue
        uid, phs, durs, slurs = f[0], f[2].split(), f[5].split(), [int(x) for x in f[6].split()]
        if not (len(phs) == len(durs) == len(slurs)):
            raise ValueError(f"{uid}: 音素 {len(phs)}・長さ {len(durs)}・スラー {len(slurs)} の数が合いません")
        if split_i:
            phs = split_apical_i(phs, slurs)
        sub = where.get(uid, "" if not splits else None)
        if sub is None:
            raise ValueError(f"{uid} が --split のどの一覧にもありません")
        _write(os.path.join(out, sub, f"{uid}.lab"), lab_lines(phs, durs))
        n[sub] = n.get(sub, 0) + 1
    return n


def convert_csv(src: str, out: str) -> int:
    """DiffSinger の transcriptions.csv（name, ph_seq, ph_dur）→ <out>/<name>.lab。書いた数を返す。"""
    k = 0
    for row in csv.DictReader(open(src, encoding="utf-8")):
        phs, durs = row["ph_seq"].split(), row["ph_dur"].split()
        if len(phs) != len(durs):
            raise ValueError(f"{row['name']}: 音素 {len(phs)} と長さ {len(durs)} の数が合いません")
        _write(os.path.join(out, f"{row['name']}.lab"), lab_lines(phs, durs))
        k += 1
    return k


PAUSES = {"SP", "AP", "pau", "sil", "exh", "br", "trash"}


def tidy_segments(segs, audio_sec: float, snap_sec: float = 0.005):
    """[(start, end, 音素)]（秒）→ 隙間なく続く区間。snap_sec 未満の隙間・重なりは、始まりを一つ前の終わりにそろえる
    （それより大きい隙間はそのまま残し、前処理が pau で埋める）。音声の終わりを超えた分は切る。"""
    out = []
    for s, e, p in segs:
        if out and abs(s - out[-1][1]) < snap_sec:
            s = out[-1][1]
        e = min(e, audio_sec)
        if e - s > 1e-7:
            out.append((s, e, p))
    return out


def fill_gaps(segs, audio, sr: int, max_cl: float = 0.08):
    """音素と音素の間（休みの記号の隣でない所）の隙間を、音声を見て埋める（Praat の有声・無声）。
    有声が半分以上 → 真ん中で前後の音素に分ける。無声で max_cl 秒未満 → cl（閉鎖）。それより長い無声 → そのまま（前処理が pau）。
    → (区間, {"split": 数, "cl": 数, "kept": 数})"""
    import parselmouth
    pitch = parselmouth.Sound(audio.astype("float64"), sampling_frequency=sr).to_pitch_ac(
        time_step=0.005, pitch_floor=70.0, pitch_ceiling=1100.0)
    f0, ts = pitch.selected_array["frequency"], pitch.xs()
    out, n = [], {"split": 0, "cl": 0, "kept": 0}
    for s, e, p in segs:
        if out and s - out[-1][1] > 1e-6 and out[-1][2] not in PAUSES and p not in PAUSES:
            g0 = out[-1][1]
            m = (ts >= g0) & (ts < s)
            voiced = (f0[m] > 0).mean() if m.any() else 0.0
            if voiced >= 0.5:
                mid = (g0 + s) / 2
                out[-1] = (out[-1][0], mid, out[-1][2])
                s = mid
                n["split"] += 1
            elif s - g0 < max_cl:
                out.append((g0, s, "cl"))
                n["cl"] += 1
            else:
                n["kept"] += 1
        out.append((s, e, p))
    return out, n


# ── GTSinger の韓国語（IPA）→ Team CODA ─────────────────────────────────────────
# GTSinger は発音どおりの IPA（鼻音化・連音などを反映）。初声と終声を書き分けないので、位置で決める（ipa_to_coda）。
# ㅔ・ㅐ（e・ɛ）は CODA と同じく e にまとめる。의 = eu i、위 = w i（SODAsoo のラベルと同じ）。
_IPA_BASE = {
    "a": "a", "ɐ": "a", "ɑ": "a", "ʌ": "eo", "ɔ": "eo", "o": "o", "u": "u", "ʊ": "u", "ɯ": "eu", "ɨ": "eu",
    "i": "i", "e": "e", "ɛ": "e", "æ": "e",
    "j": "y", "w": "w", "ɥ": "w", "ɰ": "eu",
    "k": "g", "ɡ": "g", "g": "g", "kʰ": "k", "k͈": "kk",
    "t": "d", "d": "d", "tʰ": "t", "t͈": "tt",
    "p": "b", "b": "b", "pʰ": "p", "p͈": "pp",
    "tɕ": "j", "dʑ": "j", "ts": "j", "tɕʰ": "ch", "tɕ͈": "jj", "t͈ɕ": "jj",
    "sʰ": "s", "s": "s", "ɕ": "s", "ɕʰ": "s", "ʃ": "s", "s͈": "ss", "ɕ͈": "ss",
    "h": "h", "ç": "h",
    "n": "n", "ɲ": "n", "m": "m", "ŋ": "NG", "ɾ": "r", "l": "l", "ɭ": "l",
    "k̚": "K", "t̚": "T", "p̚": "P", "m̚": "M",
    "<SP>": "SP", "SP": "SP", "": "SP", "<AP>": "AP", "AP": "AP",
}
_IPA_SPLIT = {"ø": ("w", "e"), "kʷ": ("g", "w")}          # 1 つの区間を 2 つの音素に等分する
_KO_VOWELS = {"a", "eo", "o", "u", "eu", "i", "e"}
_KO_GLIDES = {"y", "w"}
_KO_CODA = {"n": "N", "m": "M", "g": "K", "k": "K", "kk": "K", "d": "T", "t": "T", "tt": "T", "j": "T", "ch": "T",
            "jj": "T", "s": "T", "ss": "T", "b": "P", "p": "P", "pp": "P"}


def ipa_to_coda(segs):
    """[(start, end, IPA)] → ([(start, end, CODA)], {"coda": Counter, "unknown": Counter})。
    母音の後で、次が母音・わたり音でない（子音・休み・終わり）初声の記号は終声にする（n → N、g → K、s → T …）。
    l・ɭ は次が母音・わたり音なら r、それ以外は L。ɾ は母音の後で次が母音・わたり音でなければ L、それ以外は r。"""
    from collections import Counter
    base, unknown = [], Counter()
    for s, e, p in segs:
        q = p.replace("/", "").replace("ː", "").replace("͡", "")       # /i/・長音・結合の弧
        if q in _IPA_SPLIT:
            a, b = _IPA_SPLIT[q]
            m = (s + e) / 2
            base += [(s, m, a), (m, e, b)]
        elif q in _IPA_BASE:
            base.append((s, e, _IPA_BASE[q]))
        else:
            unknown[p] += 1
            base.append((s, e, "SP"))
    out, coda = [], Counter()
    for i, (s, e, p) in enumerate(base):
        prev_v = i > 0 and base[i - 1][2] in _KO_VOWELS
        next_v = i + 1 < len(base) and base[i + 1][2] in _KO_VOWELS | _KO_GLIDES
        if p == "l":
            p = "r" if next_v else "L"
        elif p == "r" and prev_v and not next_v:
            p = "L"
        elif p in _KO_CODA and prev_v and not next_v:
            coda[f"{p}->{_KO_CODA[p]}"] += 1
            p = _KO_CODA[p]
        out.append((s, e, p))
    return out, {"coda": coda, "unknown": unknown}


def convert_gtsinger(src: str, out: str, prefix: str, group: str = "Control_Group", gap_sec: float = 0.3,
                     exclude=None, split_sec: float | None = None) -> dict:
    """GTSinger の歌い手 1 人のフォルダ（<言語>/<歌い手>/<技法>/<曲>/<group>/NNNN.{wav,json}）→ 曲ごとに区間をつないだ
    <out>/<prefix>_songNN.{wav,lab}（区間の間に gap_sec の SP）と <out>/songs.tsv（番号・元の曲名・技法のフォルダ・区間の数・秒）。
    同じ曲が複数の技法のフォルダにあれば 1 曲にまとめる（評価用に曲ごと抜けるように）。
    exclude: 除く区間の集合 {(技法, 曲, 区間の番号 "NNNN")}（ラベルと音声のずれが大きい区間など）。
    split_sec: つないだ曲を、この長さを超えた後の最初の休み（区間のつなぎ目の SP も含む）の真ん中で切り、
    <prefix>_songNN_sKKK.{wav,lab} に書く。RMVPE は音声全体を一度に処理するので、長い 1 本はメモリを使い切る
    （12〜14 分の曲で Mac が落ちかけた）。→ 集計"""
    import json
    from collections import Counter, defaultdict

    import numpy as np
    import soundfile as sf
    by_song = defaultdict(list)
    n_excluded = 0
    for tech in sorted(os.listdir(src)):
        tdir = os.path.join(src, tech)
        if not os.path.isdir(tdir):
            continue
        for song in sorted(os.listdir(tdir)):
            gdir = os.path.join(tdir, song, group)
            if not os.path.isdir(gdir):
                continue
            for f in sorted(os.listdir(gdir)):
                if f.endswith(".json") and not f.startswith("."):
                    if exclude and (tech, song, f[:-5]) in exclude:
                        n_excluded += 1
                        continue
                    by_song[song].append((tech, os.path.join(gdir, f)))
    os.makedirs(out, exist_ok=True)
    total = {"coda": Counter(), "unknown": Counter(), "songs": 0, "segments": 0, "sec": 0.0, "excluded": n_excluded}
    rows_tsv = ["id\tsong\ttechs\tsegments\tsec"]
    for k, song in enumerate(sorted(by_song)):
        sid = f"{prefix}_song{k:02d}"
        audio_parts, lab, t0, sr0, subtype = [], [], 0.0, None, None
        for tech, js in by_song[song]:
            wav = js[:-5] + ".wav"
            y, sr = sf.read(wav, always_2d=False)
            if sr0 is None:
                sr0, subtype = sr, sf.info(wav).subtype
            elif sr != sr0:
                raise ValueError(f"{wav}: {sr} Hz（{sr0} Hz と違う）")
            dur = len(y) / sr
            segs = [(ps, pe, p) for it in json.load(open(js, encoding="utf-8"))
                    for p, ps, pe in zip(it["ph"], it["ph_start"], it["ph_end"])]
            segs = tidy_segments(segs, dur)
            segs, st = ipa_to_coda(segs)
            total["coda"].update(st["coda"])
            total["unknown"].update(st["unknown"])
            if segs and segs[-1][1] < dur - 1e-4:
                segs.append((segs[-1][1], dur, "SP"))
            if lab:
                lab.append((t0, t0 + gap_sec, "SP"))
                audio_parts.append(np.zeros(int(round(gap_sec * sr))))
                t0 += len(audio_parts[-1]) / sr
            lab += [(t0 + s, t0 + e, p) for s, e, p in segs]
            audio_parts.append(y)
            t0 += dur
            total["segments"] += 1
        audio = np.concatenate(audio_parts)
        pieces = (split_at_pauses(lab, split_sec, min_pause=min(0.3, gap_sec) - 1e-6) if split_sec
                  else [(0.0, lab[-1][1], lab)])
        for j, (a, b, rows) in enumerate(pieces):
            nm = f"{sid}_s{j:03d}" if split_sec else sid
            i0 = int(round(a * sr0))
            sf.write(os.path.join(out, f"{nm}.wav"), audio[i0:i0 + int(round(rows[-1][1] * sr0))], sr0, subtype=subtype)
            _write(os.path.join(out, f"{nm}.lab"), [f"{int(round(s * 1e7))} {int(round(e * 1e7))} {p}" for s, e, p in rows])
            total["max_piece_sec"] = max(total.get("max_piece_sec", 0.0), rows[-1][1])
        techs = sorted({t for t, _ in by_song[song]})
        rows_tsv.append(f"{sid}\t{song}\t{','.join(techs)}\t{len(by_song[song])}\t{len(audio) / sr0:.1f}\t{len(pieces)}")
        total["songs"] += 1
        total["pieces"] = total.get("pieces", 0) + len(pieces)
        total["sec"] += len(audio) / sr0
    _write(os.path.join(out, "songs.tsv"), [rows_tsv[0] + "\tpieces", *rows_tsv[1:]])
    return total


def split_at_pauses(segs, min_sec: float, min_pause: float = 0.3):
    """区間の並び → 片の並び [(片の始まりの秒, 片の終わりの秒, 片の区間)]。長さが min_sec を超えた後の、最初の min_pause 以上の
    休みの真ん中で切る。区間の時刻は片の始まりからの秒。最後の片が短すぎる（min_sec の半分未満）ときは前の片につなぐ。"""
    cuts, t0 = [], segs[0][0]
    for s, e, p in segs:
        if p in PAUSES and e - s >= min_pause and (s + e) / 2 - t0 >= min_sec:
            c = (s + e) / 2
            cuts.append(c)
            t0 = c
    end = segs[-1][1]
    if cuts and end - cuts[-1] < min_sec / 2:
        cuts.pop()
    bounds = [segs[0][0], *cuts, end]
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = [(max(s, a) - a, min(e, b) - a, p) for s, e, p in segs if e > a and s < b]
        pieces.append((a, b, [r for r in rows if r[1] - r[0] > 1e-7]))
    return pieces


def convert_lab(src: str, wav: str, out: str, name: str | None = None, split_sec: float | None = None,
                snap_ms: float = 5.0, gaps: bool = False) -> list[str]:
    """曲ごとの HTK lab（100 ns）と wav → <out>/<name>.lab・.wav（split_sec があれば <name>_sNNN.*）。書いた名前の一覧。
    gaps=True: 音素の間の隙間を fill_gaps で埋める（そのままだと前処理が単語の途中に pau を入れる）。"""
    import soundfile as sf
    audio, sr = sf.read(wav, always_2d=False)
    subtype = sf.info(wav).subtype
    segs = []
    for ln in open(src, encoding="utf-8-sig"):
        t = ln.split()
        if len(t) >= 3:
            segs.append((int(t[0]) / 1e7, int(t[1]) / 1e7, t[2]))
    segs = tidy_segments(segs, len(audio) / sr, snap_ms / 1000)
    if gaps:
        segs, n = fill_gaps(segs, audio if audio.ndim == 1 else audio.mean(axis=1), sr)
        print(f"gaps: {n['split']} split between the neighbours, {n['cl']} -> cl, {n['kept']} left (-> pau)")
    name = name or os.path.splitext(os.path.basename(src))[0]
    pieces = split_at_pauses(segs, split_sec) if split_sec else [(0.0, segs[-1][1], segs)]
    os.makedirs(out, exist_ok=True)
    names = []
    for k, (a, b, rows) in enumerate(pieces):
        nm = f"{name}_s{k:03d}" if split_sec else name
        i0 = int(round(a * sr))
        i1 = i0 + int(round(rows[-1][1] * sr))
        sf.write(os.path.join(out, f"{nm}.wav"), audio[i0:i1], sr, subtype=subtype)
        _write(os.path.join(out, f"{nm}.lab"), [f"{int(round(s * 1e7))} {int(round(e * 1e7))} {p}" for s, e, p in rows])
        names.append(nm)
    return names


def convert_labdir(src: str, out: str, prefix: str = "", split_sec: float = 60.0, split_long_sec: float = 120.0,
                   skip=(), snap_ms: float = 5.0) -> dict:
    """<src>/lab/<名前>.lab と <src>/wav/<名前>.wav の組をまとめて convert_lab にかける → <out>/<prefix><名前>.{wav,lab}。
    split_long_sec より長い音声だけ split_sec で片に切る（RMVPE は 1 本を丸ごと処理するので、長い 1 本はメモリを使い切る）。
    skip: 使わない名前（ラベルが別のテイクのものなど）。→ {"files", "pieces", "skipped", "no_wav"}"""
    import soundfile as sf
    skip = set(skip)
    total = {"files": 0, "pieces": 0, "skipped": [], "no_wav": []}
    for lab in sorted(os.listdir(os.path.join(src, "lab"))):
        if not lab.endswith(".lab") or lab.startswith("."):
            continue
        stem = lab[:-4]
        if stem in skip:
            total["skipped"].append(stem)
            continue
        wav = os.path.join(src, "wav", f"{stem}.wav")
        if not os.path.exists(wav):
            total["no_wav"].append(stem)
            continue
        long = sf.info(wav).duration > split_long_sec
        names = convert_lab(os.path.join(src, "lab", lab), wav, out, name=f"{prefix}{stem}",
                            split_sec=split_sec if long else None, snap_ms=snap_ms)
        total["files"] += 1
        total["pieces"] += len(names)
    return total


def main():
    ap = argparse.ArgumentParser(description="音素の長さのラベル → 区間ごとの lab（100 ns）")
    ap.add_argument("kind", choices=["opencpop", "csv", "lab", "labdir", "gtsinger"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs="*", default=None, metavar="NAME=LIST",
                    help="opencpop: 例 train=train.txt test=test.txt（<out>/<NAME>/ に分けて書く）")
    ap.add_argument("--split_i", action="store_true", help="opencpop: i を ir（zh ch sh r）・i0（z c s）に分ける")
    ap.add_argument("--wav", default=None, help="lab: 曲の wav")
    ap.add_argument("--name", default=None, help="lab: 書き出す名前（既定は lab のファイル名）")
    ap.add_argument("--split_sec", type=float, default=None,
                    help="lab・gtsinger: 長い音声をこの長さ以上の片に、休みの真ん中で切る（RMVPE のメモリのため）")
    ap.add_argument("--snap_ms", type=float, default=5.0, help="lab: これ未満の隙間・重なりをそろえる")
    ap.add_argument("--prefix", default=None,
                    help="gtsinger・labdir: 書き出す名前の頭（例 gts_ko_t1 → gts_ko_t1_song00、mechol_ → mechol_2_0）")
    ap.add_argument("--split_long", type=float, default=120.0, help="labdir: これより長い音声だけ --split_sec で切る")
    ap.add_argument("--skip", default="", help="labdir: 使わない名前（カンマ区切り）")
    ap.add_argument("--group", default="Control_Group", help="gtsinger: 使うグループのフォルダ名")
    ap.add_argument("--exclude", default=None, help="gtsinger: 除く区間の TSV（列 tech, song, seg。見出しの行があってよい）")
    ap.add_argument("--fill_gaps", action="store_true",
                    help="lab: 音素の間の隙間を音声で埋める（有声 → 前後に分ける、80 ms 未満の無声 → cl）。要 parselmouth")
    args = ap.parse_args()
    if args.kind == "opencpop":
        splits = dict(s.split("=", 1) for s in args.split) if args.split else None
        n = convert_opencpop(args.src, args.out, splits, split_i=args.split_i)
        print(f"wrote {sum(n.values())} labs to {args.out}: " + ", ".join(f"{k or '.'} {v}" for k, v in sorted(n.items())))
    elif args.kind == "gtsinger":
        if not args.prefix:
            ap.error("gtsinger には --prefix が要ります")
        exclude = None
        if args.exclude:                                   # TSV: tech, song, seg の列（見出しの行は飛ばす）
            exclude = {tuple(ln.rstrip("\n").split("\t")[:3]) for ln in open(args.exclude, encoding="utf-8")
                       if ln.strip() and not ln.startswith("tech\t")}
        t = convert_gtsinger(args.src, args.out, args.prefix, args.group, exclude=exclude, split_sec=args.split_sec)
        print(f"wrote {t['songs']} songs as {t['pieces']} files (longest {t['max_piece_sec']:.1f} s; "
              f"{t['segments']} segments, {t['sec'] / 60:.1f} min; {t['excluded']} segments excluded) to {args.out}")
        print(f"  coda by position: {dict(t['coda'].most_common())}")
        if t["unknown"]:
            print(f"  unknown symbols -> SP: {dict(t['unknown'])}")
    elif args.kind == "labdir":
        t = convert_labdir(args.src, args.out, args.prefix or "", args.split_sec or 60.0, args.split_long,
                           [s for s in args.skip.split(",") if s], args.snap_ms)
        print(f"wrote {t['pieces']} wav+lab from {t['files']} files to {args.out}; skipped {t['skipped']}"
              + (f"; no wav: {t['no_wav']}" if t["no_wav"] else ""))
    elif args.kind == "lab":
        if not args.wav:
            ap.error("lab には --wav が要ります")
        names = convert_lab(args.src, args.wav, args.out, args.name, args.split_sec, args.snap_ms, args.fill_gaps)
        print(f"wrote {len(names)} wav+lab to {args.out}: {names[0]} … {names[-1]}")
    else:
        print(f"wrote {convert_csv(args.src, args.out)} labs to {args.out}")


if __name__ == "__main__":
    main()
