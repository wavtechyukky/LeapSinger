"""
Mel2DDiscriminator — JCU判別器の 2D-CNN(PatchGAN)版。jcu.py(1D)のドロップイン代替。

動機:
  1D版(DiffGAN-TTS原典)は mel ビンを全チャンネル混合して**時間方向にだけ**畳み込むため、
  (周波数×時間)平面の**局所テクスチャ**(倍音 vs 倍音間の非周期成分・高域ディテール)の弁別が弱い。
  → mel を画像とみなし、**2D conv の局所受容野**でパッチ単位に判定する。

MRD(UnivNet)との比較で、多重解像度以外に効く2点を取り込む:
  ① **spectral_norm**: 全conv を Lipschitz 拘束。Dのロジット/勾配の青天井=暴走機構を潰す
     → GAN強度を上げても発散しにくい（我々の mel-GAN 発散の主因対策）。
  ② **周波数保持形状**: stride を**時間だけ**に掛け(横長 kernel (3,9))、freq=128 を保持。
     スペクトルの質感は周波数方向に細かいので、freq を早く潰さない。

仕様は jcu.py と同一 → **同じ d_loss_jcu / g_adv_fm_jcu をそのまま再利用**(JCU=cond+uncond 構造は共通):
  forward(x_t, x_clean, t, cond_id) -> (cond_feats, uncond_feats)
    各 feats = 中間層出力のリスト(末尾=ロジットのパッチマップ [B,1,F,T'])
  入力 = stack([x_clean, x_t]) の2チャンネル画像 [B,2,mel,T]
  t と条件(話者)埋め込みは **cond 枝の入口でのみ** 2Dブロードキャスト加算。init N(0,0.02)。
損失(LSGAN + feature matching)は jcu.py 側のヘルパを共用する(このファイルには損失を持たない)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

from leapsinger.modules.backbones.dilated_conv import ContinuousStepEmbedding


def _sn(conv: nn.Conv2d, use_sn: bool) -> nn.Module:
    """conv を N(0,0.02) 初期化してから(必要なら)spectral_norm で包む。
    ★init は wrap の前に。parametrization 後は conv.weight が計算値になり init が効かないため。"""
    nn.init.normal_(conv.weight, 0.0, 0.02)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return _spectral_norm(conv) if use_sn else conv


class Mel2DDiscriminator(nn.Module):
    def __init__(self, mel_bins: int = 128, n_cond: int = 0, step_dim: int = 128,
                 chs=(32, 64, 128), spectral_norm: bool = True):
        super().__init__()
        self.mel_bins = mel_bins
        self.use_sn = spectral_norm
        # 共有幹: 入力=(x_clean, x_t) の2ch画像。stride は時間だけ(周波数保持)、kernel は横長 (3,9)。
        #   1層目 stride(1,1) → 以降 stride(1,2)。freq は全層で 128 のまま、time を 1/4 に。
        c_prev = 2
        self.shared = nn.ModuleList()
        for i, c in enumerate(chs):
            stride = (1, 1) if i == 0 else (1, 2)
            self.shared.append(_sn(nn.Conv2d(c_prev, c, (3, 9), stride=stride, padding=(1, 4)), self.use_sn))
            c_prev = c
        top = chs[-1]
        # cond / uncond 枝(同仕様・重み別)。末尾 = 1ch ロジットのパッチマップ。
        def _branch():
            return nn.ModuleList([
                _sn(nn.Conv2d(top, top // 2, (3, 3), stride=1, padding=1), self.use_sn),
                _sn(nn.Conv2d(top // 2, 1, (3, 3), stride=1, padding=1), self.use_sn),
            ])
        self.cond_branch = _branch()
        self.uncond_branch = _branch()
        # t 埋め込み: 正弦波+MLP → top ch へ(jcu.py と同型)。条件埋め込みは D 専有(G と非共有)。
        # step_mlp / cond_emb は SN しない(MRD は conv のみ正規化。埋め込みは加算注入で小さい)。
        self.step_emb = ContinuousStepEmbedding(step_dim)
        self.step_mlp = nn.Sequential(nn.Linear(step_dim, top * 2), nn.SiLU(),
                                      nn.Linear(top * 2, top))
        self.cond_emb = nn.Embedding(n_cond, top) if n_cond > 0 else None

    def forward(self, x_t, x_clean, t, cond_id=None):
        """x_t/x_clean: [B, mel, T]([-1,1])、t:[B]∈[0,1]、cond_id:[B] or None
        Returns: (cond_feats, uncond_feats) — 各層出力のリスト(末尾=ロジット [B,1,F,T'])"""
        x = torch.stack([x_clean, x_t], dim=1)       # [B, 2, mel, T]
        cond_feats, uncond_feats = [], []
        for l in self.shared:
            x = F.leaky_relu(l(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)
        e = self.step_mlp(self.step_emb(t))          # [B, top]
        if self.cond_emb is not None and cond_id is not None:
            e = e + self.cond_emb(cond_id)
        e = e[:, :, None, None]                       # [B, top, 1, 1] → freq/time にブロードキャスト
        xc, xu = x + e, x                            # jcu.py どおり cond 枝の入口でのみ加算
        for l in self.cond_branch:
            xc = F.leaky_relu(l(xc), 0.2)
            cond_feats.append(xc)
        for l in self.uncond_branch:
            xu = F.leaky_relu(l(xu), 0.2)
            uncond_feats.append(xu)
        return cond_feats, uncond_feats
