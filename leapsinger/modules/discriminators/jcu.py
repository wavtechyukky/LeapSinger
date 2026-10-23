"""
JCUMelDiscriminator — DiffGAN-TTS の JCU discriminator を rectified flow mel へ翻案した軽量判別器。

原典: DiffGAN-TTS JCUDiscriminator（joint conditional-unconditional）。
  - 入力 = concat(クリーン側, x_t) の 2×mel ch（原典は (x_{t-1}, x_t)）
  - 1x1 射影 → 共有幹 1D conv 3層（→64→128→512, k3/5/5, s1/2/2, LeakyReLU 0.2）
  - cond 枝 / uncond 枝 各2層（512→128→1, k5/3, s1/1）。JCU = 条件付き+無条件の2ロジット
  - 時刻 t と条件（話者 or スタイル）の埋め込みを **cond 枝の入口でのみ** 特徴に加算
  - 重み初期化 normal_(0, 0.02)

flow への翻案:
  - DDPM の (x_t, x_{t-1}) ペア → (x_t, クリーン側)。real=(x_t, x1_norm)、
    fake=(x_t, clamp(x1_pred_norm, -1, 1))。x1_pred = x_t + (1-t)v（1-step clean 予測）。
  - t は連続 [0,1]。埋め込みは ContinuousStepEmbedding（正弦波+MLP）を流用し 512ch へ持ち上げ。
  - 判定域は flow の正規化域 [-1,1]（x_t がこの域にしか存在しないため）。
  - 条件 cond_id は **話者id**（多話者）or スタイルid。音素では条件付けしない（過平滑対策の定石）。
    D 専有の埋め込みで、G 側の spk_emb/style_emb とは共有しない。

損失（DiffGAN-TTS 準拠）:
  - LSGAN: cond/uncond ロジットの MSE 平均 ×0.5（本物→1 / 偽物→0）
  - FM: cond/uncond 両枝の全中間層 L1（最終 logit 除外）× feat_weights=4/(n_layers+1)×0.5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from leapsinger.modules.backbones.dilated_conv import ContinuousStepEmbedding


class JCUMelDiscriminator(nn.Module):
    def __init__(self, mel_bins: int = 128, n_cond: int = 0, step_dim: int = 128):
        super().__init__()
        in_ch = 2 * mel_bins                       # concat(clean, x_t)
        self.inp = nn.Conv1d(in_ch, in_ch, 1)      # 原典 LinearNorm 射影の 1D conv 版
        # 共有幹（原典 n_channels[:3]=[64,128,512], kernel [3,5,5], stride [1,2,2]）
        self.shared = nn.ModuleList([
            nn.Conv1d(in_ch, 64, 3, stride=1, padding=1),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.Conv1d(128, 512, 5, stride=2, padding=2),
        ])
        # cond / uncond 枝（原典 n_channels[3:]=[128,1], kernel [5,3], stride [1,1]。同仕様・重み別）
        def _branch():
            return nn.ModuleList([
                nn.Conv1d(512, 128, 5, stride=1, padding=2),
                nn.Conv1d(128, 1, 3, stride=1, padding=1),
            ])
        self.cond_branch = _branch()
        self.uncond_branch = _branch()
        # t 埋め込み: 正弦波+MLP → 512ch へ持ち上げ
        self.step_emb = ContinuousStepEmbedding(step_dim)
        self.step_mlp = nn.Sequential(nn.Linear(step_dim, 512), nn.SiLU(),
                                      nn.Linear(512, 512))
        # 条件（話者 or スタイル）埋め込み。D 専有 — G の埋め込みは共有しない
        self.cond_emb = nn.Embedding(n_cond, 512) if n_cond > 0 else None

        def _init(m):                              # 原典 weights_init: conv を N(0, 0.02)
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(self, x_t, x_clean, t, cond_id=None):
        """x_t/x_clean: [B, mel, T]（flow 正規化域 [-1,1]）、t: [B]∈[0,1]、cond_id: [B] or None
        Returns: (cond_feats, uncond_feats) — 各層出力のリスト（末尾=ロジット [B,1,T']）"""
        x = self.inp(torch.cat([x_clean, x_t], dim=1))
        cond_feats, uncond_feats = [], []
        for l in self.shared:
            x = F.leaky_relu(l(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)
        e = self.step_mlp(self.step_emb(t)).unsqueeze(-1)          # [B,512,1]
        if self.cond_emb is not None and cond_id is not None:
            e = e + self.cond_emb(cond_id).unsqueeze(-1)
        xc, xu = x + e, x                          # 原典どおり cond 枝の入口でのみ加算
        for l in self.cond_branch:
            xc = F.leaky_relu(l(xc), 0.2)
            cond_feats.append(xc)
        for l in self.uncond_branch:
            xu = F.leaky_relu(l(xu), 0.2)
            uncond_feats.append(xu)
        return cond_feats, uncond_feats


# ── 損失（DiffGAN-TTS 準拠。LSGAN + feature matching） ────────────────────────────

def _jcu_mse(cond_logit, uncond_logit, target: float):
    return 0.5 * (F.mse_loss(cond_logit, torch.full_like(cond_logit, target)) +
                  F.mse_loss(uncond_logit, torch.full_like(uncond_logit, target)))


def d_loss_jcu(disc, x_t, real_n, fake_n, t, cond_id=None):
    """D 更新用 LSGAN 損失。fake は内部で detach。
    Returns: (d_loss, dict(logit平均 — TB 監視用))"""
    rc, ru = disc(x_t, real_n, t, cond_id)
    fc, fu = disc(x_t, fake_n.detach(), t, cond_id)
    d_loss = _jcu_mse(rc[-1], ru[-1], 1.0) + _jcu_mse(fc[-1], fu[-1], 0.0)
    logits = {'d_real_logit': 0.5 * (rc[-1].detach().mean().item() + ru[-1].detach().mean().item()),
              'd_fake_logit': 0.5 * (fc[-1].detach().mean().item() + fu[-1].detach().mean().item())}
    return d_loss, logits


def g_adv_fm_jcu(disc, x_t, real_n, fake_n, t, cond_id=None):
    """G 更新用: adversarial(LSGAN, fake→1) + feature matching(両枝 全中間層 L1)。
    D の重みには勾配を流さない。Returns: (adv, fm)"""
    disc.requires_grad_(False)
    fc, fu = disc(x_t, fake_n, t, cond_id)
    with torch.no_grad():
        rc, ru = disc(x_t, real_n, t, cond_id)
    adv = _jcu_mse(fc[-1], fu[-1], 1.0)
    w = 4.0 / (len(fc) + 1)                        # 原典 feat_weights（n_layers=5 → 0.667）
    fm = sum(w * 0.5 * (F.l1_loss(rc[j].detach(), fc[j]) +
                        F.l1_loss(ru[j].detach(), fu[j]))
             for j in range(len(fc) - 1))          # 最終ロジット層は除外
    disc.requires_grad_(True)
    return adv, fm


# ── 監視指標: mel 鮮明度 Var_L（Revisiting Over-Smoothness, Ren+ 2022） ────────────

@torch.no_grad()
def laplacian_var_ratio(pred_mel, gt_mel, valid=None) -> float:
    """Laplacian 応答の分散（=時間/周波数エッジ量）の pred/GT 比。
    1.0 に近いほど GT 並みの鮮明さ。回帰系学習は潰れて 1.0 を下回る。
    入力: [B, mel, T]（ln-mel 域。同一スケールで比を取るので域はどちらでも可）。
    valid: [B, T] bool（True=有効フレーム）。学習バッチではパディング（損失をマスクしていて出力が
    何でもよいフレーム）を除くために渡す。3x3 の窓が全部有効なフレームだけで分散を取る。"""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=pred_mel.device, dtype=torch.float32).view(1, 1, 3, 3)
    def _r(m):
        return F.conv2d(m.unsqueeze(1).float(), k).abs()          # [B,1,mel-2,T-2]
    rp, rg = _r(pred_mel), _r(gt_mel)
    if valid is None:
        return float(rp.var() / rg.var().clamp_min(1e-8))
    v = valid.bool()
    v = v[:, 1:-1] & v[:, :-2] & v[:, 2:]                         # [B, T-2]
    m = v[:, None, None, :].expand_as(rp)
    if not bool(m.any()):
        return float("nan")
    return float(rp[m].var() / rg[m].var().clamp_min(1e-8))
