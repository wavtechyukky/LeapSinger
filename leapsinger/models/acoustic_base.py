"""
HarmonicAcousticBase: harmonic 音響モデルの基底（mel 直接 rectified flow + dilated conv backbone）。

phoneme(attention encoder) → length regulate → + f0 (+ uv) (+ speaker) (+ style) を条件 cond にまとめる。
実際の出発点 x0（F0倍音＋白色ノイズの励起 mel）と少数ステップ推論は harmonic サブクラス
（models/acoustic.py）が与える。CFG/guidance なし。
"""
import torch
import torch.nn as nn

from leapsinger.modules.encoders.phoneme_encoder import PhonemeEncoder
from leapsinger.modules.length_regulator import LengthRegulator
from leapsinger.modules.flow.mel_dilated_rectified_flow import MelDilatedRectifiedFlow
from leapsinger.modules.backbones.dilated_conv import DILATION_SCHEDULES
from leapsinger.modules.position import PositionCondition


class HarmonicAcousticBase(nn.Module):
    def __init__(
        self,
        n_phonemes: int,
        hidden: int = 256,
        mel_bins: int = 128,
        mel_vmin: float = -11.5,
        mel_vmax: float = 2.0,
        backbone_ch: int = 256,
        n_cycles: int = 3,
        n_speakers: int = 0,           # 0 = 単一話者（spk_emb なし）
        n_styles: int = 0,             # 0 = スタイルなし。>0 で style_emb 追加
        use_uv: bool = True,           # True=frame v/uv を uv_emb で条件に加算＋励起の有声ゲートに使う。
                                       # False=uvフリー(DiffSinger互換): uv_emb を持たず、励起は全域有声
                                       # (補間F0で倍音が全域に出る)。
        flow_loss: str = 'l1',         # 'l1'(既定) | 'l2'(openvpi 同等)。学習時のみ使用
        dilation_schedule: str = None, # backbone の dilation 列（DILATION_SCHEDULES のキー。None=n_cycles既定）
        pos_cond: dict = None,         # {phoneme, unit, bins, encoding}。フレームの相対位置を cond に加算
                                       # (leapsinger/modules/position.py)。None/全 False で無し。encoding が
                                       # 無い dict（2026-10-16 の A/B の ckpt）は旧符号化 raised_cosine
        dur_cond: bool = False,        # log(1+音素フレーム数) の埋め込みを cond に加算（DiffSinger の dur_embed）
        nucleus_ids=None,              # pos_cond.unit の単位境界になる核音素（母音・N）の id 列
        pau_id: int = 0,
    ):
        super().__init__()
        self.hidden = hidden
        self.mel_bins = mel_bins
        self.use_uv = use_uv

        self.phoneme_encoder = PhonemeEncoder(n_phonemes, hidden)
        self.length_regulator = LengthRegulator()
        pc = dict(pos_cond or {})
        self.pos_cond = None
        if pc.get("phoneme") or pc.get("unit") or dur_cond:
            self.pos_cond = PositionCondition(n_phonemes, hidden, nucleus_ids, pau_id=pau_id,
                                              phoneme=bool(pc.get("phoneme", False)),
                                              unit=bool(pc.get("unit", False)),
                                              bins=int(pc.get("bins", 8)), dur=bool(dur_cond),
                                              encoding=str(pc.get("encoding", "raised_cosine")))
        self.f0_emb = nn.Linear(1, hidden)
        self.uv_emb = nn.Embedding(2, hidden) if use_uv else None   # uvフリー時は持たない
        self.spk_emb = None
        if n_speakers > 0:
            self.spk_emb = nn.Embedding(n_speakers, hidden)
            nn.init.normal_(self.spk_emb.weight, 0, hidden ** -0.5)
        # ボーカルスタイル(例: ritsu normal/soft)。離散 style_id、または推論時に
        # style_mix [B, n_styles] の重みで埋め込みを線形ブレンド(連続スタイル)。
        self.style_emb = None
        if n_styles > 0:
            self.style_emb = nn.Embedding(n_styles, hidden)
            nn.init.normal_(self.style_emb.weight, 0, hidden ** -0.5)

        dilations = DILATION_SCHEDULES[dilation_schedule] if dilation_schedule else None
        self.flow = MelDilatedRectifiedFlow(
            vmin=mel_vmin, vmax=mel_vmax, mel_bins=mel_bins,
            dim_cond=hidden, ch=backbone_ch, n_cycles=n_cycles,
            loss_type=flow_loss, dilations=dilations,
        )

    # ── shared encode ──────────────────────────────────────────────────────────
    def _encode(
        self,
        phoneme_ids, ph_durations,
        f0_logf0, uv,
        padding_mask=None, max_frames=None, spk_id=None,
        style_id=None, style_mix=None,
    ):
        ph_feat = self.phoneme_encoder(phoneme_ids, padding_mask)     # [B, Tp, H]
        c = self.length_regulator(ph_feat, ph_durations, max_frames)  # [B, T, H]
        if self.pos_cond is not None:                                 # 音素内/単位内の相対位置 (+ log 長)
            c = c + self.pos_cond(phoneme_ids, ph_durations, c.shape[1])
        c = c + self.f0_emb(f0_logf0.unsqueeze(-1))
        if self.use_uv:                            # uvフリー時は v/uv を条件に入れない
            c = c + self.uv_emb((uv > 0.5).long())
        if self.spk_emb is not None and spk_id is not None:
            c = c + self.spk_emb(spk_id)[:, None, :]   # [B,H] → 全フレームに話者ベクトル加算
        if self.style_emb is not None:
            if style_mix is not None:                  # 連続ブレンド [B, n_styles]
                c = c + (style_mix @ self.style_emb.weight)[:, None, :]
            elif style_id is not None:                 # 離散切替 [B]
                c = c + self.style_emb(style_id)[:, None, :]
        return c.transpose(1, 2)                  # [B, H, T]


def up2_linear(x):
    """hop512 格子 [B, T] → 内部 hop256 格子 [B, 2T] への線形2倍補間（偶数=元値・奇数=中点、末尾2点は
    最終値の複製）。export が hop512 界面の f0/dur を内部 hop256 へ写すのに使う（export/wrappers.py）。"""
    mid = 0.5 * (x[:, :-1] + x[:, 1:])
    out = torch.stack([x[:, :-1], mid], dim=2).reshape(x.shape[0], -1)
    return torch.cat([out, x[:, -1:], x[:, -1:]], dim=1)
