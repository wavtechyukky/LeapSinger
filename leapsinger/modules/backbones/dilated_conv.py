"""
DilatedConvDenoiser: DiffWave型 gated dilated conv バックボーン（mel rectified flow の velocity_fn）。

WaveNet/DiffWave 型の gated dilated conv（tanh×sigmoid ゲート）。入力=mel[B,mel_bins,T] + 連続時刻
t∈[0,1] 埋め込み + 条件 cond。dilation 列は DILATION_SCHEDULES から選ぶ（未指定時は n_cycles×[1,3,9,27,81]）。
"""
import math
import torch
import torch.nn as nn

from leapsinger.modules.backbones.gated_res_block import GatedResBlock


# dilation スケジュール（各層の dilation 列。層数=len）。速度/容量は「層数」で決まる。
#   pow2_15: DiffWave 論文準拠の 2の冪、15層（既定で全 config が使用）。
DILATION_SCHEDULES = {
    'pow2_15': [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64],
}


class ContinuousStepEmbedding(nn.Module):
    """連続時刻 t∈[0,1] の正弦波埋め込み → MLP（rectified flow 用、離散step版の連続化）。"""

    def __init__(self, dim: int = 128, scale: float = 1000.0):
        super().__init__()
        half = dim // 2
        self.register_buffer('freqs', torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)))
        self.scale = scale
        self.dim = half * 2
        self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU(),
                                 nn.Linear(self.dim, self.dim), nn.SiLU())

    def forward(self, t):                          # t: [B] in [0,1]
        a = t[:, None] * self.freqs[None, :] * self.scale
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a)], dim=-1))   # [B, dim]


class DilatedConvDenoiser(nn.Module):
    """
    Args:
        mel_bins  : 入出力次元（128）
        ch        : 残差チャネル幅
        dim_cond  : 条件次元（content+f0+uv の hidden）
        n_cycles  : dilation 周期の繰り返し数（[1,3,9,27,81]×n_cycles 層）。dilations 未指定時のみ有効
        step_dim  : 時刻埋め込み次元
        dilations : 明示的な dilation 列。指定時は n_cycles を無視してこの列で層を構成
                    （DILATION_SCHEDULES の値を渡す想定。A/B 用）
    forward: x[B, mel_bins, T], t[B]∈[0,1], cond[B, dim_cond, T] → v[B, mel_bins, T]
    """
    DILATIONS = [1, 3, 9, 27, 81]

    def __init__(self, mel_bins: int = 80, ch: int = 256, dim_cond: int = 256,
                 n_cycles: int = 3, step_dim: int = 128, kernel: int = 3,
                 dilations=None):
        super().__init__()
        dils = list(dilations) if dilations is not None else self.DILATIONS * n_cycles
        self.in_conv = nn.Conv1d(mel_bins, ch, 1)
        self.step_emb = ContinuousStepEmbedding(step_dim)
        self.blocks = nn.ModuleList(
            GatedResBlock(ch, kernel=kernel, dilation=d, cond_dim=dim_cond, step_dim=step_dim)
            for d in dils)
        self.out = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.SiLU(), nn.Conv1d(ch, mel_bins, 1))
        # autocast（train.amp=bf16）のときに入出力と和を fp32 に残すか（config train.amp_fp32_io、train.py が
        # 設定する）。重みではないので ckpt には入らない。autocast でなければ関係ない
        self.fp32_io = True

    def forward(self, x, t, cond):
        se = self.step_emb(t)                      # [B, step_dim]
        if self.fp32_io and torch.is_autocast_enabled(x.device.type):
            return self._forward_mixed(x, se, cond)
        h = self.in_conv(x)
        skips = 0
        for blk in self.blocks:
            h, s = blk(h, cond=cond, step_emb=se)
            skips = skips + s
        return self.out(skips)                     # [B, mel_bins, T]

    def _forward_mixed(self, x, se, cond):
        """autocast（train.amp=bf16）の中で呼ばれたとき。入力の 1x1 conv・残差の流れ・skip の和・出力の層は
        fp32 に残し、15 層の中身（dilated conv・ゲート・1x1 conv）だけ bf16 で回す。丸めが積み重なる所と、
        出力（速度）そのものの精度を守る（実サイズ・乱数の重みで出力の相対誤差 0.74% → 0.44%）。
        autocast でなければ通らないので、fp32 の学習・推論・ONNX 書き出しは変わらない。"""
        dev = x.device.type
        with torch.autocast(device_type=dev, enabled=False):
            h = self.in_conv(x.float())
        skips = 0
        for blk in self.blocks:
            h, s = blk(h, cond=cond, step_emb=se)  # h は fp32 のまま（bf16 の残差と足すと fp32 になる）
            skips = skips + s.float()
        with torch.autocast(device_type=dev, enabled=False):
            return self.out(skips)
