"""harmonic_excitation — F0 から「倍音＋白色ノイズ」の励起 mel を作る（flow の出発点 x0 用）。

設計:
    x0（flow の出発点）= 決定的な倍音（F0由来・きれい）＋ 固定振幅の白色ノイズ（非周期）。
    flow は x0→target へ「フォルマント包絡」だけを残差として運ぶ。倍音を発明しないので、
    ノイズから全再生成する場合に出る倍音の粒状ちらつきを避けられる。

倍音源:
    - F0 は**無声区間も log2 線型補間した連続値**（NHVSing 励起に合わせる）→ **1サンプルごと線型補間**で
      サンプル解像度へ。位相 = 2π∫f0 dt（cumsum）。倍音 = Σ_{k=1..K} (1/k)·sin(k·phase)（-6dB/oct）。
      振幅は voiced ゲート（無声では倍音0、ただし位相は連続なので有声再開時にコヒーレント）。
    - 白色ノイズは全域・固定振幅（noise_ratio × 倍音RMS）。無声子音/息の非周期成分を担う。
mel は target と同一変換（wav_to_mel_nhv_torch, 44.1k/hop256/128/ln）。全 torch＝GPU on-the-fly。
"""
import math
import os

import torch
import torch.nn.functional as F

from leapsinger.mel import wav_to_mel_nhv_torch


def _harm_rms(n_harm: int, decay: float = 1.0) -> float:
    """RMS of Σ_{k=1..K}(1/k^decay)·sin(k·φ) = sqrt(Σ(1/k^decay)²/2) (voiced, long window)."""
    return math.sqrt(sum((1.0 / k ** decay) ** 2 for k in range(1, n_harm + 1)) / 2.0)


def _fill_unvoiced_log2(f0_hz: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    """[B,T] → 連続 F0[B,T](Hz)。有声フレームで log2 線型補間し無声を埋める（全 torch）。"""
    B, T = f0_hz.shape
    dev = f0_hz.device
    logf = torch.log2(f0_hz.clamp(min=1.0))
    ar = torch.arange(T, device=dev, dtype=f0_hz.dtype)[None].expand(B, T)
    vm = voiced > 0.5
    # 左: t 以下で最後の有声 index（cummax）／右: t 以上で最初の有声 index（reverse cummin）
    left_idx = torch.cummax(torch.where(vm, ar, torch.full_like(ar, -1.0)), dim=1).values
    right_val = torch.where(vm, ar, torch.full_like(ar, float(T)))
    right_idx = torch.flip(torch.cummin(torch.flip(right_val, [1]), dim=1).values, [1])
    li = left_idx.clamp(min=0).long()
    ri = right_idx.clamp(max=T - 1).long()
    lf = torch.gather(logf, 1, li)
    rf = torch.gather(logf, 1, ri)
    w = (ar - left_idx) / (right_idx - left_idx).clamp(min=1e-6)
    cont = lf * (1.0 - w) + rf * w
    cont = torch.where(left_idx < 0, rf, cont)          # 先頭の無声 → 最初の有声を保持
    cont = torch.where(right_idx > T - 1, lf, cont)      # 末尾の無声 → 最後の有声を保持
    any_v = vm.any(dim=1, keepdim=True)
    cont = torch.where(any_v, cont, torch.full_like(cont, math.log2(200.0)))
    return torch.exp2(cont)


# ── 倍音和（Σ_k (1/k^decay)·sin(k·phase)·[k·f0<nyq]）─────────────────────────────
# サンプルレート([B,n])の倍音和は励起の主コスト（GPU compute の ~60%）。Python の 256回
# 逐次ループは 256回のカーネル起動＝メモリ律速。ループを外し `torch.compile` で融合すると、
# 中間 [B,H,n] を実体化せずレジスタ加算する単一カーネルになり ~3-4x（GPU実測 48→12ms）。
# compile 非対応/opt-out 時はメモリ安全なループ（数値は bit 一致・Δ~1e-6=加算順のみ）。
# LEAPSINGER_EXC_COMPILE=0 で compile を切りループを使う。
_EXC_COMPILE = os.environ.get("LEAPSINGER_EXC_COMPILE", "1") != "0"


def _harm_sum(phase: torch.Tensor, f0_up: torch.Tensor, kk: torch.Tensor,
              nyq: float, decay: float) -> torch.Tensor:
    """loop-free 倍音和 -> [B,n]。phase,f0_up [B,n]; kk=[1,H,1]（=[1,2,..,H]）。
    torch.compile 前提（融合で [B,H,n] を実体化しない）。eager だと [B,H,n] を作るので大バッチは非推奨。"""
    ph = phase.unsqueeze(1)                                            # [B,1,n]
    fu = f0_up.unsqueeze(1)                                            # [B,1,n]
    aa = (kk * fu < nyq).to(phase.dtype)                              # [B,H,n] anti-alias
    return (torch.sin(kk * ph) * _amp(kk, decay) * aa).sum(dim=1)     # [B,n]


def _amp(kk: torch.Tensor, decay: float) -> torch.Tensor:
    """1/k^decay（[1,H,1]）。inductor の CPU カーネルは、整数でない pow（と exp·log、外から渡した振幅の
    テンソル）を [B,H,n] の要素ごとに計算して 2 倍遅くなる（7 秒の句で 87 → 180 ms）。既定の 1.5 は
    reciprocal·rsqrt で書く（値は pow と 1 ulp 程度の差）。"""
    if decay == 1.5:
        return kk.reciprocal() * kk.rsqrt()
    return kk.pow(-decay)


_harm_sum_c = (torch.compile(_harm_sum, dynamic=True)
               if _EXC_COMPILE and hasattr(torch, "compile") else None)


def _harm_sum_loop(phase: torch.Tensor, f0_up: torch.Tensor,
                   n_harm: int, nyq: float, decay: float) -> torch.Tensor:
    """メモリ安全なフォールバック（従来の逐次ループ）。中間 [B,H,n] を作らない。"""
    harm = torch.zeros_like(phase)
    for k in range(1, n_harm + 1):
        harm = harm + (1.0 / k ** decay) * torch.sin(k * phase) * (k * f0_up < nyq).to(phase.dtype)
    return harm


def _harmonics(phase: torch.Tensor, f0_up: torch.Tensor, n_harm: int,
               nyq: float, decay: float) -> torch.Tensor:
    """compile 融合版があればそれを、無ければ（compile 非対応/opt-out）メモリ安全なループ。
    CPU でも融合版が速い（7 秒の句で 87 ms。ループは 4 スレッドで 135 ms、1 スレッドで 400 ms）。"""
    if _harm_sum_c is not None:
        kk = torch.arange(1, n_harm + 1, device=phase.device,
                          dtype=phase.dtype).view(1, -1, 1)
        return _harm_sum_c(phase, f0_up, kk, nyq, decay)
    return _harm_sum_loop(phase, f0_up, n_harm, nyq, decay)


@torch.no_grad()
def harmonic_wave(f0_logf0: torch.Tensor, uv: torch.Tensor, *,
                  n_harm: int = 256, sr: int = 44100, hop: int = 256,
                  harm_decay: float = 1.0) -> torch.Tensor:
    """決定論的な倍音波形 [B, n=T*hop]（ノイズ・STFT 前）。f0/uv から一意に決まる＝キャッシュ対象。
    連続補間 F0 の位相積分に -6dB/oct(1/k) 倍音を重ね、unit RMS + voiced ゲート。ここが励起の主コスト。"""
    if f0_logf0.dim() == 1:
        f0_logf0 = f0_logf0[None]; uv = uv[None]
    B, T = f0_logf0.shape
    voiced = (uv > 0.5).to(f0_logf0.dtype)
    f0_hz = torch.where(voiced > 0.5, torch.exp2(f0_logf0), torch.zeros_like(f0_logf0))
    f0_cont = _fill_unvoiced_log2(f0_hz, voiced)                       # [B,T] Hz（連続）
    n = T * hop
    f0_up = F.interpolate(f0_cont[:, None], size=n, mode='linear', align_corners=False)[:, 0]  # [B,n]
    v_up = F.interpolate(voiced[:, None], size=n, mode='linear', align_corners=False)[:, 0]
    phase = torch.cumsum(2.0 * math.pi * f0_up / sr, dim=1)            # [B,n]
    harm = _harmonics(phase, f0_up, n_harm, sr / 2.0, harm_decay)      # [B,n] loop-free/compiled
    return (harm / _harm_rms(n_harm, harm_decay)) * v_up              # unit RMS + voiced gate


@torch.no_grad()
def harm_wave_to_mel(harm: torch.Tensor, *, noise_ratio: float = 0.05,
                     scale: float = 0.15, hop: int = 256, n_frames: int | None = None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """倍音波形 [B,n] → 励起 ln-mel [B,128,T]。毎ステップの安価部（フレッシュ白色ノイズ + STFT）。
    n_frames=None なら n//hop フレーム（キャッシュ波形のバッチ幅に一致）。"""
    noise = torch.randn(harm.shape, device=harm.device, dtype=harm.dtype, generator=generator)
    exc = scale * (harm + noise_ratio * noise)                        # ノイズ全域・固定振幅
    mel = wav_to_mel_nhv_torch(exc, hop=hop)                          # [B,128,T']
    T = harm.shape[-1] // hop if n_frames is None else n_frames
    return mel[:, :, :T]


@torch.no_grad()
def harmonic_noise_mel_torch(f0_logf0: torch.Tensor, uv: torch.Tensor,
                             noise_ratio: float = 0.05, n_harm: int = 5,
                             sr: int = 44100, hop: int = 256, scale: float = 0.15,
                             harm_decay: float = 1.0,
                             generator: torch.Generator | None = None) -> torch.Tensor:
    """f0_logf0[B,T], uv[B,T] -> excitation ln-mel [B,128,T]（倍音+白色ノイズ）。
    harmonic_wave（決定論・重い）→ harm_wave_to_mel（ノイズ+STFT・安価）の合成。後方互換・ビット一致。"""
    if f0_logf0.dim() == 1:
        f0_logf0 = f0_logf0[None]; uv = uv[None]
    T = f0_logf0.shape[1]
    harm = harmonic_wave(f0_logf0, uv, n_harm=n_harm, sr=sr, hop=hop, harm_decay=harm_decay)
    return harm_wave_to_mel(harm, noise_ratio=noise_ratio, scale=scale, hop=hop,
                            n_frames=T, generator=generator)
