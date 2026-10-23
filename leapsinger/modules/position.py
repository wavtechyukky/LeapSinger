"""フレーム単位の相対位置の条件（config `model.pos_cond` / `model.dur_cond`）。

長さ調整で音素を引き伸ばしただけの条件には「その音素のどのあたりにいるか」が無い。ここでは音素内の
相対位置を作って cond に加算する。符号化は 2 通り（`pos_cond.encoding`）:

  sinusoidal（configs の既定、2026-10-17〜）: DiffSinger の stretch 埋め込みと同じ形。
      s = round(1000 · i / n)（i = 音素内のフレーム番号 0..n−1、n = 音素のフレーム数）を
      Transformer 型の正弦波埋め込み（hidden 次元）→ Linear(h, 4h) → GELU → Linear(4h, h) に通す。
      s は 0〜1000 の整数なので、1001 行の表を作ってから引く（DiffSinger の学習時と同じ高速化）。
      DiffSinger は続けて条件の列に GRU を通すが、ここでは入れていない（速度場ネットの dilated conv が
      前後を見る。GRU は 1 フレームずつ順に処理するので、長いフレーズで学習が遅くなる）。
  raised_cosine（旧。2026-10-16 の A/B の pos_agc）: p = i / (n − 1)（1 フレームの音素は 0.5）を
      e_j = 0.5 + 0.5·cos(2π(p − j/K))、j = 0..K−1 で符号化して線形変換（0 と 1 が同じ符号）。SynthV 2 Pro
      の解析から取った形なので新しい学習では使わない。古い ckpt を読むためだけに残す。ckpt の pos_cond に
      encoding が無ければこちら。
      unit（「核音素（母音・N）の開始 → 次の核音素の開始、または pau」を 1 単位とした単位内の位置。子音は
      前の単位に属する。SynthV 2 Pro の「ノート頭 → 次のノート頭」をなぞったもの）もこちらでだけ使える。

dur_cond: log(1 + 音素フレーム数) の線形埋め込み（DiffSinger の dur_embed）。別スイッチ。

configs の既定（2026-10-17〜）: phoneme だけ on・sinusoidal。unit は使わない（SynthV のなぞりになるため）。
dur_cond は off（A/B で測れる効果が無かった）。

ONNX: 学習では torch.searchsorted でフレーム→音素を引くが、legacy exporter は searchsorted / cummax /
scatter を出せないので、export は `frame_to_phoneme` 属性を `frame_to_phoneme_onnx`（比較の総和）に
差し替える（export/wrappers.py）。単位の開始・長さも cummax を使わず、下三角マスクの max / min で出す。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def frame_to_phoneme(durations: torch.Tensor, max_len: int) -> torch.Tensor:
    """[B,Tp] フレーム数 → [B,T] 各フレームの音素 index（学習用: searchsorted）。"""
    dur = durations.long().clamp(min=0)
    ends = dur.cumsum(dim=1)
    frame = torch.arange(max_len, device=durations.device)
    return torch.searchsorted(ends, frame.unsqueeze(0).expand(dur.shape[0], -1).contiguous(),
                              right=True).clamp(max=dur.shape[1] - 1)


def frame_to_phoneme_onnx(durations: torch.Tensor, max_len) -> torch.Tensor:
    """同じ写像を ONNX に出せる形で（比較の総和。[B,Tp,T] を作るので export 専用）。"""
    dur = durations.long()
    ends = dur.cumsum(dim=1)
    frame = torch.arange(max_len, device=durations.device)
    cmp = ends.unsqueeze(2) <= frame.view(1, 1, -1)
    return cmp.to(torch.int64).sum(dim=1).clamp(max=dur.shape[1] - 1)


def unit_bounds(durations: torch.Tensor, is_boundary: torch.Tensor):
    """音素ごとの (単位の開始フレーム, 単位のフレーム数) [B,Tp]。is_boundary[b,k]=True の音素 k が
    新しい単位を始める。cummax / scatter を使わず、三角マスクの max / min で出す（export 可）。"""
    dur = durations.long().clamp(min=0)
    ends = dur.cumsum(dim=1)
    starts = ends - dur
    total = ends[:, -1:]                                                   # [B,1]
    k = torch.arange(dur.shape[1], device=dur.device)
    lower = k.view(-1, 1) >= k.view(1, -1)                                 # [Tp(k), Tp(j)] : j <= k
    upper = k.view(-1, 1) < k.view(1, -1)                                  # j > k
    is_boundary = is_boundary.clone()
    is_boundary[:, 0] = True                                               # 先頭は必ず単位の頭
    bstart = torch.where(is_boundary, starts, torch.zeros_like(starts))    # 単位頭の開始フレーム、他は 0
    unit_start = torch.where(lower.unsqueeze(0), bstart.unsqueeze(1),
                             torch.zeros_like(bstart).unsqueeze(1)).amax(dim=2)     # max_{j<=k}: 直近の単位頭
    bend = torch.where(is_boundary, starts, total.expand_as(starts))       # 次の単位頭の開始、他は総フレーム
    unit_end = torch.where(upper.unsqueeze(0), bend.unsqueeze(1),
                           total.unsqueeze(2).expand(-1, dur.shape[1], dur.shape[1])).amin(dim=2)
    return unit_start, (unit_end - unit_start).clamp(min=1)


def relative_position(idx: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """p = idx / (length − 1)。長さ 1 は 0.5。"""
    n = length.float()
    return torch.where(n > 1, idx.float() / (n - 1).clamp(min=1), torch.full_like(n, 0.5))


def raised_cosine(p: torch.Tensor, bins: int) -> torch.Tensor:
    """[...] → [..., bins]: e_j = 0.5 + 0.5·cos(2π(p − j/bins))。"""
    j = torch.arange(bins, device=p.device, dtype=p.dtype) / bins
    return 0.5 + 0.5 * torch.cos(2.0 * math.pi * (p.unsqueeze(-1) - j))


STRETCH_STEPS = 1000                                        # DiffSinger: round(1000 · i / n)


class SinusoidalEmbedding(nn.Module):
    """x [...] → [..., dim]: [sin(x·w_k), cos(x·w_k)]、w_k = 10000^(−k/(dim/2 − 1))（Transformer 型）。"""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(torch.arange(half) * -(math.log(10000.0) / (half - 1))),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.float().unsqueeze(-1) * self.freqs
        return torch.cat([a.sin(), a.cos()], dim=-1)


def stretch_index(i: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """DiffSinger の stretch: round(1000 · i / n)（i = 音素内のフレーム番号 0..n−1）→ 0..1000 の整数。"""
    s = torch.round(STRETCH_STEPS * i.float() / n.clamp(min=1).float())
    return s.long().clamp(0, STRETCH_STEPS)


class PositionCondition(nn.Module):
    """phoneme_ids [B,Tp], durations [B,Tp], T → cond への加算項 [B,T,hidden]。"""

    def __init__(self, n_phonemes: int, hidden: int, nucleus_ids, pau_id: int = 0,
                 phoneme: bool = True, unit: bool = True, bins: int = 8, dur: bool = False,
                 encoding: str = "raised_cosine"):
        super().__init__()
        self.use_phoneme, self.use_unit, self.bins, self.use_dur = bool(phoneme), bool(unit), int(bins), bool(dur)
        self.encoding = str(encoding)
        if self.encoding not in ("sinusoidal", "raised_cosine"):
            raise ValueError(f"pos_cond.encoding='{encoding}' は不正（sinusoidal | raised_cosine）")
        if self.encoding == "sinusoidal" and self.use_unit:
            raise ValueError("pos_cond.unit は旧符号化（raised_cosine）でだけ使える")
        self.pau_id = int(pau_id)
        table = torch.zeros(n_phonemes, dtype=torch.bool)
        for i in (nucleus_ids or []):
            table[int(i)] = True
        self.register_buffer("is_nucleus", table)
        self.proj = None                                    # raised_cosine: K ビン → hidden
        self.stretch_embed = None                           # sinusoidal: DiffSinger の stretch 埋め込み
        if self.encoding == "sinusoidal":
            if self.use_phoneme:
                self.stretch_embed = nn.Sequential(SinusoidalEmbedding(hidden), nn.Linear(hidden, hidden * 4),
                                                   nn.GELU(), nn.Linear(hidden * 4, hidden))
        else:
            n_in = self.bins * (int(self.use_phoneme) + int(self.use_unit))
            self.proj = nn.Linear(n_in, hidden) if n_in > 0 else None
        self.dur_emb = nn.Linear(1, hidden) if self.use_dur else None
        self.frame_to_phoneme = frame_to_phoneme            # export がここを差し替える

    def forward(self, phoneme_ids: torch.Tensor, durations: torch.Tensor, max_len) -> torch.Tensor:
        dur = durations.long().clamp(min=0)
        ends = dur.cumsum(dim=1)
        starts = ends - dur
        ph = self.frame_to_phoneme(durations, max_len)                     # [B,T]
        frame = torch.arange(max_len, device=durations.device).unsqueeze(0).expand(dur.shape[0], -1)
        out = None
        if self.stretch_embed is not None:
            s = stretch_index(frame - torch.gather(starts, 1, ph), torch.gather(dur, 1, ph))     # [B,T] 0..1000
            table = self.stretch_embed(torch.arange(STRETCH_STEPS + 1, device=durations.device))  # [1001,hidden]
            out = table[s]
        feats = []
        if self.proj is not None and self.use_phoneme:
            i = frame - torch.gather(starts, 1, ph)
            feats.append(raised_cosine(relative_position(i, torch.gather(dur, 1, ph)), self.bins))
        if self.proj is not None and self.use_unit:
            is_b = self.is_nucleus[phoneme_ids] | (phoneme_ids == self.pau_id)
            us, ul = unit_bounds(dur, is_b)
            i = frame - torch.gather(us, 1, ph)
            feats.append(raised_cosine(relative_position(i, torch.gather(ul, 1, ph)), self.bins))
        if self.proj is not None:
            out = self.proj(torch.cat(feats, dim=-1))
        if self.dur_emb is not None:
            d = self.dur_emb(torch.log1p(torch.gather(dur, 1, ph).float()).unsqueeze(-1))
            out = d if out is None else out + d
        valid = (frame < ends[:, -1:]).unsqueeze(-1).to(out.dtype)          # 総フレーム超のパディングは 0
        return out * valid
