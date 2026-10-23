"""Dataset + batching for the LeapSinger acoustic model.

Reads the `shard.npz` produced by `preprocess` — each phrase carries the **raw** mel,
F0 (`f0_interp`) + voiced flag, phoneme ids, and per-phoneme durations. Speaker / style
ids are assigned per-DB at train time (not baked in). The loader derives per-phoneme frame
durations and log2 F0, and — unless disabled — silences the pau regions (fading breath
toward the mel floor). Silencing lives here, not in the data, so keeping the breath is one
`silence=False` away.

Optional GPU pitch augmentation reads the raw clip audio from the `shard_wav.npz`
sidecar (present when preprocessing with `save_wav=True`); the trainer frequency-warps it.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from preprocess.vocab import PAU_ID
from preprocess.phrase_cut import clip_pau_gain
from leapsinger.loudness import normalize_mel
from leapsinger.phonemes import consonant_frame_weights
from leapsinger.reflow import reflow_eps

_MEL_FLOOR = math.log(1e-5)     # ln-mel silence floor (matches the preprocess mel clamp)


def _song_of(stem: str) -> str:
    """Phrase name '{song}_{idx:04d}' -> song name (drop the trailing _NNNN)."""
    return re.sub(r"_\d{4,}$", "", stem)


def _gapless_f0(f0_hz: np.ndarray) -> np.ndarray:
    """Fill pitch-less frames (f0 <= 1 Hz) by holding/interpolating the real values (in Hz, like
    preprocess). Older shards store f0 = 0 on the artificial edge silence `cut_phrases` adds;
    unfilled, those frames would reach the model as 1 Hz, while inference F0 is always gap-less."""
    good = f0_hz > 1.0
    if good.all() or not good.any():
        return f0_hz
    idx = np.arange(len(f0_hz))
    return np.interp(idx, idx[good], f0_hz[good]).astype(np.float32)


def _lead_pad_frames(f0_raw: np.ndarray) -> int:
    """Leading artificial-silence frames of an older shard phrase (f0 stored as 0 there; real
    frames are always interpolated, so leading zeros can only be artificial). Its clip wav does
    not cover them, so it has to be shifted by this many frames to line up with the mel."""
    good = np.flatnonzero(f0_raw > 1.0)
    return int(good[0]) if len(good) else 0


def _phoneme_frames(dur_sec: np.ndarray, T: int, frame_rate: float):
    """Phoneme boundaries in frames fb [L+1] (the last one forced to T) and per-phoneme frames (at least 1)."""
    cum = np.concatenate([[0.0], np.cumsum(dur_sec)])
    fb = np.round(cum * frame_rate).astype(int)
    fb[-1] = T
    return fb, np.maximum(np.diff(fb), 1).astype(np.int64)


def _open(path: str, cache: dict):
    """Lazily open (and cache per process) an npz shard; None if missing. Fork-safe."""
    key = (os.getpid(), str(path))
    if key not in cache:
        cache[key] = np.load(path) if os.path.exists(path) else None
    return cache[key]


class LeapSingerDataset(Dataset):
    def __init__(self, dirs, split: str = "train", eval_songs: int = 2,
                 eval_dbs=None,
                 min_sec: float = 0.0, pitch_aug: bool = False, seed: int = 42,
                 silence: bool = True, silence_fade_sec: float = 0.05,
                 spk_map: dict | None = None, style_map: dict | None = None,
                 loudness_agc: dict | None = None, max_sec: float | None = None):
        # spk_map/style_map: {db-dir-name -> id}. Speaker/style identity is NOT baked into the
        # shard; it is assigned here at train time from these maps (keyed by the DB folder name,
        # e.g. "ritsu_soft"). Absent map -> fall back to any id stored in the shard, else 0.
        # loudness_agc: config `data.loudness_agc` (sigma_sec / target_db / k / max_gain_db /
        # max_cut_db) -> the target mel loses everything slower than ~sigma (phrase level, section
        # level, crescendi) and keeps note-level and phonetic dynamics. None = off.
        self.spk_map = spk_map
        self.style_map = style_map
        self.split = split
        self.pitch_aug = pitch_aug
        self.load_wav = pitch_aug
        self.silence = silence
        self.silence_fade_sec = silence_fade_sec
        self.loudness_agc = dict(loudness_agc) if loudness_agc else None
        self._cache: dict = {}
        self._harm_cache: dict = {}      # index i -> 決定論的倍音波形 np[float32, n]（warm_harm_cache で充填）
        self._reflow: dict | None = None  # index i -> reflow の組の教師の出力 np[float16, D, mel, T]（attach_reflow_targets）
        self._reflow_x0: dict | None = None  # 同じ形の出発点（波形の白色ノイズがあるときだけ。無ければ eps を作り直す）
        self._reflow_seed = 0
        self._cons_w: dict | None = None     # 子音のフレームの損失の重み（set_consonant_weights。None = 付けない）
        self.files: list = []            # (shard_path, phrase_name, db_dir)
        self.frame_counts: List[int] = []
        frame_rate = None
        self.hop = 256
        n_too_long: list = []            # max_sec で除外したフレーズ（名前, 秒）

        for d in dirs:
            d = Path(d)
            meta = json.load(open(d / "metadata.json", encoding="utf-8"))
            frame_rate = float(meta.get("frame_rate", frame_rate or 172.265625))
            self.hop = int(meta.get("hop_size", self.hop))
            phrases = meta["phrases"]
            names = sorted(phrases)
            songs = sorted({_song_of(n) for n in names})
            # eval_dbs: eval に使う DB をここに挙げた分だけに絞る opt-in。
            # 未指定(None)なら全 DB から抜く従来どおりの挙動。1 話者あたりの曲数が少ない
            # コーパス（NUS-48E は 4 曲）で全 DB から 1 曲ずつ抜くと eval が 25% に達し、
            # 学習データを大きく削ってしまうため。
            if eval_dbs is not None and os.path.basename(os.path.normpath(str(d))) not in eval_dbs:
                n_hold = 0                                      # この DB は全曲 train へ
            else:
                n_hold = min(eval_songs, max(0, len(songs) - 1))  # keep >=1 song in train
            hold = set(random.Random(seed).sample(songs, n_hold)) if n_hold else set()
            shard = str(d / "shard.npz")
            for n in names:
                is_eval = _song_of(n) in hold
                if (split == "eval") != is_eval:
                    continue
                T = int(phrases[n])
                if min_sec and T < min_sec * frame_rate:
                    continue
                # max_sec: 極端に長いフレーズを除外（config data.max_sec）。少数の長尺が単独で
                # 小さなバッチを作り、学習を不安定にするため。切り出しはしない（除外するだけ）
                if max_sec and T > max_sec * frame_rate:
                    n_too_long.append((n, T / frame_rate))
                    continue
                self.files.append((shard, n, str(d)))
                self.frame_counts.append(T)

        self.frame_rate = float(frame_rate or 172.265625)
        # per-phrase の均等サンプリング用キー。speaker=話者id（多スタイルは合算）／db=DBフォルダ名
        # （＝話者×スタイルのユニット。各声を等しく学習したいときはこちら）。spk_map 無し=全て0。
        self.speaker_ids = [int(self.spk_map.get(os.path.basename(os.path.normpath(db)), 0))
                            if self.spk_map else 0 for (_s, _n, db) in self.files]
        self.db_ids = [os.path.basename(os.path.normpath(db)) for (_s, _n, db) in self.files]
        self.silence_fade_frames = (max(1, int(round(silence_fade_sec * self.frame_rate)))
                                    if silence else 0)
        print(f"[dataset] {split}: {len(self.files)} phrases  frame_rate={self.frame_rate:.3f}"
              f"  silence={'off' if not silence else f'{silence_fade_sec}s fade'}")
        if n_too_long:                   # 除外は必ず名前ごと出す（黙って減らさない）
            print(f"[dataset] {split}: max_sec={max_sec} で {len(n_too_long)} 句を除外: "
                  + ", ".join(f"{n} {s:.1f}s" for n, s in sorted(n_too_long, key=lambda x: -x[1])))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        shard, name, dbdir = self.files[i]
        z = _open(shard, self._cache)

        def g(k):
            return z[f"{name}|{k}"]

        ph_ids    = g("phoneme_ids").astype(np.int64)
        dur_sec   = g("dur_sec").astype(np.float32)
        f0_raw    = g("f0_interp").astype(np.float32)
        f0_hz     = _gapless_f0(f0_raw)
        uv        = g("uv").astype(np.float32)
        mel       = g("mel").astype(np.float32)
        dbname = os.path.basename(os.path.normpath(dbdir))
        if self.spk_map is not None:
            spk_id = int(self.spk_map.get(dbname, 0))
        else:                                              # backward compat: id from shard, else 0
            spk_id = int(g("spk_id")) if f"{name}|spk_id" in z.files else 0
        if self.style_map is not None:
            style_id = int(self.style_map.get(dbname, 0))
        else:
            style_id = int(g("style_id")) if f"{name}|style_id" in z.files else 0

        T = mel.shape[1]
        fb, ph_durs = _phoneme_frames(dur_sec, T, self.frame_rate)     # boundaries / per-phoneme frames
        f0_logf0 = np.log2(np.maximum(f0_hz, 1.0)).astype(np.float32)

        sil = np.zeros(T, dtype=bool)                        # lab-derived pau frames
        for j in range(len(ph_ids)):
            if ph_ids[j] == PAU_ID:
                sil[fb[j]:fb[j + 1]] = True

        # optional slow AGC (config data.loudness_agc): take the slow loudness envelope out of
        # the target before the pau fade, estimated on sung frames only (leapsinger/loudness.py)
        if self.loudness_agc:
            mel = normalize_mel(mel, ~sil, self.frame_rate, **self.loudness_agc)

        # optional pau silencing (default on): fade pau regions toward the mel floor. Frames
        # come from the pau phonemes; cl (even long) is never silenced. Edge pau -> one-sided
        # fade, internal pau -> two-sided (clip_pau_gain).
        if self.silence and sil.any():
            g = clip_pau_gain(sil, self.silence_fade_frames)
            logg = np.log(np.clip(g, 1e-12, 1.0)).astype(mel.dtype)
            mel = np.maximum(_MEL_FLOOR, mel + logg[None, :])

        out = {
            "ph_ids": ph_ids, "ph_durs": ph_durs,
            "spk_id": np.int64(spk_id), "style_id": np.int64(style_id),
            "f0_logf0": f0_logf0, "uv": uv, "target_mel": mel, "item_name": name,
        }
        if i in self._harm_cache:                    # 事前計算した決定論的倍音波形（キャッシュ）
            out["harm_wave"] = self._harm_cache[i]
        if self._cons_w is not None:                 # 子音のフレームの損失の重み [T]（train.cons_weight）
            out["frame_w"] = consonant_frame_weights(ph_ids, ph_durs, n_frames=T, **self._cons_w)
        if self._reflow is not None:                 # reflow の生徒: D 通りの組から 1 つ
            x1 = self._reflow[i]                     # [D, mel, T] fp16（正規化域）
            if x1.shape[2] != T:
                raise ValueError(f"{self.phrase_key(i)}: reflow の組 {x1.shape[2]} フレームと mel {T} フレームが合いません")
            d = random.randrange(x1.shape[0])
            out["reflow_x1"] = x1[d]
            if self._reflow_x0 is not None:          # 出発点そのもの（波形の白色ノイズがあるとき）
                out["reflow_x0"] = self._reflow_x0[i][d]
            else:                                    # eps を種から作り直す
                out["reflow_eps"] = reflow_eps(self.phrase_key(i), d, x1.shape[1], T, self._reflow_seed).numpy()
            out["reflow_d"] = d
        if self.load_wav:
            wz = _open(os.path.join(dbdir, "shard_wav.npz"), self._cache)
            if wz is not None and f"{name}|wav" in wz.files:
                wav = wz[f"{name}|wav"].astype(np.float32) / 32767.0
                if len(wav) < T * self.hop:              # older shard: wav lacks the artificial lead
                    wav = np.pad(wav, (_lead_pad_frames(f0_raw) * self.hop, 0))
                out["wav"] = wav
        return out

    @torch.no_grad()
    def warm_harm_cache(self, *, n_harm: int, harm_decay: float, exc_hop: int,
                        use_uv: bool, device) -> None:
        """決定論的倍音波形（励起の主コスト）を全フレーズ分 GPU で生成し float32 CPU にキャッシュ。
        以降 __getitem__ が `harm_wave` を返し、モデルは毎ステップ「フレッシュノイズ+STFT」だけで済む
        （倍音和の再計算を省略）。**各フレーズ standalone（batch=1）で計算**＝バッチのパディング境界
        アーティファクトが無く、推論時（batch=1）の励起と一致。DataLoader を作る前に呼ぶこと（fork
        worker が COW で共有）。pitch_aug 時は f0 が毎step変わるので呼ばない。"""
        from leapsinger.modules.harmonic_excitation import harmonic_wave
        import time as _t
        dev = torch.device(device)
        n_tot = len(self.files); _t0 = _t.time()
        print(f"[dataset] warming harm cache: {n_tot} phrases (standalone, ~1min)...", flush=True)
        for i in range(n_tot):
            if i and i % 500 == 0:
                print(f"[dataset]   {i}/{n_tot} ({_t.time()-_t0:.0f}s)", flush=True)
            shard, name, _ = self.files[i]
            z = _open(shard, self._cache)
            f0_np = _gapless_f0(z[f"{name}|f0_interp"].astype(np.float32))   # = __getitem__
            uv_np = z[f"{name}|uv"].astype(np.float32)
            f0l = torch.as_tensor(np.log2(np.maximum(f0_np, 1.0)), device=dev)[None]   # __getitem__ と同一 np.log2
            uvu = (torch.ones_like(f0l) if not use_uv                                  # use_uv 偽なら全域有声
                   else torch.as_tensor(uv_np, device=dev)[None])
            harm = harmonic_wave(f0l, uvu, n_harm=n_harm, hop=exc_hop, harm_decay=harm_decay)  # [1,T*hop]
            self._harm_cache[i] = harm[0].float().cpu().numpy()
        gb = sum(a.nbytes for a in self._harm_cache.values()) / 1e9
        print(f"[dataset] harm cache warmed: {len(self._harm_cache)} phrases ({gb:.1f} GB float32, standalone) "
              f"in {_t.time() - _t0:.0f}s", flush=True)

    def set_consonant_weights(self, cons_ids, weight: float, next_frames: int = 0) -> None:
        """子音（cons_ids）のフレームと、その直後の音素の頭 next_frames フレームの損失の重みを weight にする
        （leapsinger/phonemes.consonant_frame_weights）。以降 __getitem__ は frame_w [T] を返す。DataLoader の前に呼ぶ。"""
        self._cons_w = dict(cons_ids=frozenset(int(i) for i in cons_ids), weight=float(weight),
                            next_frames=int(next_frames), pau_id=int(PAU_ID))

    def weighted_frame_share(self) -> float:
        """set_consonant_weights の重みが付くフレームが、全フレーズのフレームに占める割合（起動時の表示用）。"""
        n_w = n = 0
        for i, (shard, name, _db) in enumerate(self.files):
            z = _open(shard, self._cache)
            T = self.frame_counts[i]
            _fb, ph_durs = _phoneme_frames(z[f"{name}|dur_sec"].astype(np.float32), T, self.frame_rate)
            w = consonant_frame_weights(z[f"{name}|phoneme_ids"].astype(np.int64), ph_durs, n_frames=T, **self._cons_w)
            n_w += int((w != 1.0).sum())
            n += T
        return n_w / max(n, 1)

    def phrase_key(self, i: int) -> str:
        """フレーズの名前 "DB名/フレーズ名"（reflow の eps の種に使う）。"""
        _shard, name, dbdir = self.files[i]
        return f"{os.path.basename(os.path.normpath(dbdir))}/{name}"

    def attach_reflow_targets(self, targets: dict, seed: int = 0, x0: dict | None = None) -> None:
        """reflow の組（leapsinger/reflow.make_reflow_targets の出力 (x1, x0)。seed はそのときの値）を持たせる。以降
        __getitem__ は組を 1 つ選び、教師の出力 reflow_x1・番号 reflow_d と、出発点 reflow_x0（x0 を渡したとき）か
        出発点のノイズ reflow_eps（渡さないとき。種から作り直す）を返す。
        DataLoader を作る前に呼ぶこと（fork した worker が COW で共有する）。"""
        for name, tg in (("組", targets), ("組の出発点", x0)):
            missing = [] if tg is None else [self.phrase_key(i) for i in range(len(self.files)) if i not in tg]
            if missing:
                raise ValueError(f"reflow の{name}が無いフレーズがあります: {missing[:5]}（ほか {max(0, len(missing) - 5)} 本）")
        self._reflow = targets
        self._reflow_x0 = x0
        self._reflow_seed = int(seed)

    def __getstate__(self):                          # drop open npz handles across fork/spawn
        s = self.__dict__.copy()
        s["_cache"] = {}
        return s


def acoustic_collate_fn(batch: list) -> dict:
    B = len(batch)
    max_L = max(len(b["ph_ids"]) for b in batch)
    max_T = max(b["target_mel"].shape[1] for b in batch)
    mel_dim = batch[0]["target_mel"].shape[0]

    def pad_ph(key, dtype, val=0):
        o = torch.full((B, max_L), val, dtype=dtype)
        for i, b in enumerate(batch):
            t = torch.as_tensor(b[key], dtype=dtype)
            o[i, :len(t)] = t
        return o

    def pad_fr(key, dtype, val=0.0):
        o = torch.full((B, max_T), val, dtype=dtype)
        for i, b in enumerate(batch):
            t = torch.as_tensor(b[key], dtype=dtype)
            o[i, :len(t)] = t
        return o

    ph_mask = torch.ones(B, max_L, dtype=torch.bool)
    frame_mask = torch.ones(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        ph_mask[i, :len(b["ph_ids"])] = False
        frame_mask[i, :b["target_mel"].shape[1]] = False

    mel_out = torch.zeros(B, mel_dim, max_T)
    for i, b in enumerate(batch):
        m = torch.as_tensor(b["target_mel"], dtype=torch.float32)
        mel_out[i, :, :m.shape[1]] = m

    out = {
        "ph_ids": pad_ph("ph_ids", torch.long),
        "ph_durs": pad_ph("ph_durs", torch.long),
        "f0_logf0": pad_fr("f0_logf0", torch.float32),
        "uv": pad_fr("uv", torch.float32),
        **({"frame_w": pad_fr("frame_w", torch.float32)} if "frame_w" in batch[0] else {}),   # 詰め物は 0
        "target_mel": mel_out,
        "ph_mask": ph_mask, "frame_mask": frame_mask,
        "spk_id": torch.tensor([int(b["spk_id"]) for b in batch], dtype=torch.long),
        "style_id": torch.tensor([int(b["style_id"]) for b in batch], dtype=torch.long),
        "item_name": [b["item_name"] for b in batch],
    }
    if "wav" in batch[0]:
        max_W = max(len(b["wav"]) for b in batch)
        wav = torch.zeros(B, max_W)
        for i, b in enumerate(batch):
            w = torch.as_tensor(b["wav"], dtype=torch.float32)
            wav[i, :len(w)] = w
        out["wav"] = wav
    if "harm_wave" in batch[0]:                       # 事前計算倍音波形 [n_i] を [B, max_n] に zero-pad
        max_N = max(len(b["harm_wave"]) for b in batch)   # = max_T * hop（mel の max_T と整合）
        hw = torch.zeros(B, max_N)
        for i, b in enumerate(batch):
            w = torch.as_tensor(b["harm_wave"], dtype=torch.float32)
            hw[i, :len(w)] = w
        out["harm_wave"] = hw
    if "reflow_x1" in batch[0]:                       # reflow の組（正規化域）[mel, T_i] を [B, mel, max_T] に zero-pad
        # 教師の出力・出発点は保存した fp16 のまま送る（値は変わらない。GPU で float にする）。eps は float32
        for key, dt in (("reflow_x1", torch.float16), ("reflow_x0", torch.float16), ("reflow_eps", torch.float32)):
            if key not in batch[0]:
                continue
            o = torch.zeros(B, mel_dim, max_T, dtype=dt)
            for i, b in enumerate(batch):
                a = torch.as_tensor(b[key], dtype=dt)
                o[i, :, :a.shape[1]] = a
            out[key] = o
        out["reflow_d"] = torch.tensor([int(b["reflow_d"]) for b in batch], dtype=torch.long)
    return out


class FrameBasedBatchSampler(Sampler):
    """Length-sorted batches capped by total frames (B*maxT) and item count.

    weights（per-phrase 抽出重み）を渡すと、各エポックで**重み付き復元抽出**してから長さ順に詰める。
    話者均等化に使う: 少数話者に高い重みを与えると実質オーバーサンプルされ、話者ごとの学習回数が揃う。

    min_items（config train.min_batch_items）: 0 = 従来どおり短い順に詰める。この詰め方だと最後に
    最長の 1〜4 本が余り、それだけで 1 バッチになる（メモリ上限のためではなく詰め方の余り）。Adam 系は
    バッチの大きさによらずほぼ同じ幅で更新するので、1 本だけのバッチはノイズの大きい一歩になる。
    >0 にすると長い順に詰め（余りは最短側に回る）、それでも最後のバッチが min_items 未満なら直前の
    バッチと合わせて半分ずつに分け直す。"""

    def __init__(self, frame_counts, max_frames: int, max_items: int,
                 shuffle: bool = True, seed: int = 0, weights=None, min_items: int = 0):
        self.frame_counts = list(frame_counts)
        self.max_frames = max_frames
        self.max_items = max_items
        self.min_items = int(min_items or 0)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float)
        self._batches = self._build(range(len(self.frame_counts)))   # 常にビルド(__len__ と非weighted 用)

    def _fits(self, batch) -> bool:
        return (len(batch) <= self.max_items
                and len(batch) * max(self.frame_counts[i] for i in batch) <= self.max_frames)

    def _pack(self, order):
        """order の順に、本数×最長フレーム数 と本数の上限まで貪欲に詰める。"""
        batches, cur, cur_max = [], [], 0
        for idx in order:
            fc = self.frame_counts[idx]
            nmax = max(cur_max, fc)
            if cur and (nmax * (len(cur) + 1) > self.max_frames or len(cur) + 1 > self.max_items):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(idx)
            cur_max = max(cur_max, fc)
        if cur:
            batches.append(cur)
        return batches

    def _build(self, indices):
        if self.min_items <= 0:                          # 従来: 短い順（余りの最長数本が単独バッチになる）
            return self._pack(sorted(indices, key=lambda i: self.frame_counts[i]))
        batches = self._pack(sorted(indices, key=lambda i: -self.frame_counts[i]))   # 長い順
        if len(batches) >= 2 and len(batches[-1]) < self.min_items:
            merged = batches[-2] + batches[-1]           # 長い順のまま
            h = (len(merged) + 1) // 2
            a, b = merged[:h], merged[h:]
            if self._fits(a) and self._fits(b):
                batches[-2:] = [a, b]
        return batches

    def __iter__(self):
        if self.weights is not None:                     # 話者均等: 毎epoch 重み付き復元抽出→長さ順に詰める
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idxs = torch.multinomial(self.weights, len(self.frame_counts),
                                     replacement=True, generator=g).tolist()
            batches = self._build(idxs)
        else:
            batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self._batches)
