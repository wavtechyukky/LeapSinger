"""HarmonicAcousticModel / HarmonicAcousticModelMultiSpk — 単段の harmonic 音響モデル。

F0倍音＋白色ノイズの励起 mel を出発点 x0 に、rectified flow がフォルマント包絡だけを少数ステップ
Euler で精製する（前段の粗mel回帰などは無い＝単段）。

HarmonicAcousticModelMultiSpk — 低次元話者ベクトルを cond に加算する多話者版。
    - 話者: 親の full-width `spk_emb` は使わず（親を n_speakers=0 で構築）、**低次元 `spk_bank(N, spk_dim)`
      → `spk_proj(spk_dim→hidden)` → cond 加算**。ボトルネックで正則化＋話者適応/ゼロショット向き。
    - 音素/本体は親と共有。注入は `_encode` の override だけ（forward/infer は self._encode を呼ぶ）。
"""
import torch
import torch.nn as nn

from leapsinger.models.acoustic_base import HarmonicAcousticBase


class HarmonicAcousticModel(HarmonicAcousticBase):
    def __init__(self, *args, noise_ratio: float = 0.05, n_harm: int = 5, exc_scale: float = 0.15,
                 harm_decay: float = 1.0, exc_hop: int = 256, x0_sigma: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 励起 mel（flow の出発点 x0）のパラメータ
        self.noise_ratio = float(noise_ratio)
        self.n_harm = int(n_harm)
        self.exc_scale = float(exc_scale)
        self.harm_decay = float(harm_decay)     # 1.0 = -6 dB/oct; 倍音減衰スロープ
        self.exc_hop = int(exc_hop)             # excitation hop は mel hop と一致させる
        # 出発点に足すガウスノイズの標準偏差（正規化域）。0 = 従来の決定論的な擬似 mel。> 0 = reflow の教師用の
        # 確率的な出発点（複数 step で解く）。eps を渡せば、そのノイズを使う（生徒の学習・組の生成で再現するため）
        self.x0_sigma = float(x0_sigma)

    def _excitation_x0(self, f0_logf0, uv, harm_wave=None, eps=None, generator=None):
        """F0倍音＋白色ノイズ励起の正規化 mel [B,mel,T]（flow の出発点 x0）。
        harm_wave 指定時（事前計算した決定論的倍音波形）は毎ステップ「フレッシュノイズ+STFT」だけ
        （倍音和の再計算を省略）。None なら従来どおりフル生成。どちらもノイズは毎回新規。
        x0_sigma > 0 なら、正規化域でガウスノイズ x0_sigma · eps を足す（eps 省略時は毎回新規）。
        generator: 波形の白色ノイズの乱数（reflow の組を作るとき、同じ種から同じ出発点を作るため）。"""
        from leapsinger.modules.harmonic_excitation import harmonic_noise_mel_torch, harm_wave_to_mel
        if harm_wave is not None:                    # キャッシュ経路: 倍音波形→noise+STFT のみ
            exc = harm_wave_to_mel(harm_wave, noise_ratio=self.noise_ratio,
                                   scale=self.exc_scale, hop=self.exc_hop, generator=generator)   # [B,mel,T]
        else:
            if not self.use_uv:                      # uvフリー: 全域有声＝倍音が全域に出る(補間F0)
                uv = torch.ones_like(f0_logf0)
            exc = harmonic_noise_mel_torch(
                f0_logf0, uv, noise_ratio=self.noise_ratio, n_harm=self.n_harm,
                hop=self.exc_hop, scale=self.exc_scale, harm_decay=self.harm_decay,
                generator=generator)                 # [B,mel,T]
        x0 = self.flow._norm(exc)                    # [-1,1]
        if self.x0_sigma > 0:
            if eps is None:
                eps = torch.randn_like(x0)
            x0 = x0 + self.x0_sigma * eps[..., :x0.shape[-1]].to(x0)
        return x0

    def _recon_loss(self, x1_pred, target_mel, frame_mask=None):
        """x1_pred（flow の 1-step clean mel 予測）vs target の masked L1（正規化域）。
        速度場損失に加えて mel を直接罰する＝mel 忠実度を上げる補助。"""
        e = (self.flow._norm(x1_pred) - self.flow._norm(target_mel)).abs()
        if frame_mask is None:
            return e.mean()
        m = (~frame_mask)[:, None, :].to(e.dtype)
        return (e * m).sum() / m.sum().clamp(min=1.0) / e.shape[1]

    # ── training ──────────────────────────────────────────────────────────────
    def forward(
        self,
        phoneme_ids, ph_durations,
        f0_logf0, uv,
        target_mel,
        padding_mask=None, spk_id=None, style_id=None,
        frame_mask=None, harm_wave=None,
    ):
        """frame_mask [B,T] bool（True=パディング）: 指定時は損失を有効フレームのみで平均。
        harm_wave: 事前計算した決定論的倍音波形（キャッシュ）。指定時は励起の倍音和を省略。"""
        cond = self._encode(
            phoneme_ids, ph_durations, f0_logf0, uv, padding_mask,
            max_frames=target_mel.shape[2], spk_id=spk_id, style_id=style_id,
        )
        # x0 = F0倍音＋白色ノイズ励起。flow はフォルマント包絡だけを残差輸送する。
        x0_init = self._excitation_x0(f0_logf0, uv, harm_wave=harm_wave)
        flow_mask = None if frame_mask is None else (~frame_mask)
        flow_loss, x1_pred = self.flow.compute_loss(target_mel, cond, x0_init=x0_init, mask=flow_mask)
        recon = self._recon_loss(x1_pred, target_mel, frame_mask)
        return {'flow': flow_loss, 'recon': recon, 'total': flow_loss,
                'x1_pred': x1_pred, 'cond': cond}

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(
        self,
        phoneme_ids, ph_durations,
        f0_logf0, uv,
        n_frames: int,
        num_steps: int = 10,
        algorithm: str = 'euler',
        padding_mask=None, spk_id=None, style_id=None, style_mix=None,
        harm_wave=None, eps=None, x0=None,
    ):
        """Returns: pred_mel [B, mel_bins, T_frames]。harm_wave: 事前計算倍音波形（任意）。
        eps: x0_sigma > 0 のときに出発点に足すノイズ [B, mel_bins, T]（省略時は毎回新規。reflow の組の生成用）。
        x0: 出発点（正規化域 [B, mel_bins, T]）をそのまま使う（reflow の組の生成用。指定時は harm_wave・eps を使わない）。"""
        cond = self._encode(
            phoneme_ids, ph_durations, f0_logf0, uv, padding_mask,
            max_frames=n_frames, spk_id=spk_id, style_id=style_id, style_mix=style_mix,
        )
        x0_init = x0 if x0 is not None else self._excitation_x0(f0_logf0, uv, harm_wave=harm_wave, eps=eps)
        return self.flow.inference(cond, num_steps=num_steps, algorithm=algorithm, x0_init=x0_init)


class HarmonicAcousticModelMultiSpk(HarmonicAcousticModel):
    def __init__(self, *args, n_speakers: int = 1, spk_dim: int = 32, **kwargs):
        # 親（HarmonicAcousticModel→Base）には full-width 話者を作らせない（低次元 bank を使う）。
        kwargs['n_speakers'] = 0
        super().__init__(*args, **kwargs)
        assert n_speakers >= 1, n_speakers
        self.spk_n = n_speakers            # 低次元 bank の行数（config: n_speakers）
        self.spk_dim = spk_dim
        self.spk_bank = nn.Embedding(n_speakers, spk_dim)
        nn.init.normal_(self.spk_bank.weight, 0, spk_dim ** -0.5)
        self.spk_proj = nn.Linear(spk_dim, self.hidden)   # [B,spk_dim] → [B,hidden]

    def _spk_add(self, cond, spk_id):
        """cond[B,H,T] に話者ベクトルを加算。spk_id [B] long。None=未指定(加算せず)。"""
        if spk_id is None:
            return cond
        v = self.spk_proj(self.spk_bank(spk_id))          # [B, hidden]
        return cond + v.unsqueeze(-1)                      # [B, hidden, 1] → broadcast over T

    def _encode(self, *args, spk_id=None, **kwargs):
        cond = super()._encode(*args, spk_id=None, **kwargs)   # [B, H, T]
        return self._spk_add(cond, spk_id)
