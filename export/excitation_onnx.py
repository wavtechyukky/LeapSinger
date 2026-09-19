"""ONNX-friendly harmonic+noise excitation — a drop-in, opset-18-exportable equivalent of
`leapsinger.modules.harmonic_excitation.harmonic_noise_mel_torch` (which is left untouched).

Why a separate module: the training excitation uses three ops that do not export to ONNX —
`torch.exp2`, `torch.stft(return_complex=True)`, and `torch.cummax/cummin` (inside
`_fill_unvoiced_log2`). This module removes all three while keeping the numbers identical:

  * exp2(x)                     -> exp(ln2 * x)
  * _fill_unvoiced_log2         -> DROPPED. The excitation contract here is that F0 is already
                                   gap-less/continuous (host-supplied interpolated F0, which is
                                   exactly the uv-free / DiffSinger case), so the unvoiced-gap
                                   interpolation is the identity and cummax/cummin never run.
  * torch.stft(...).abs()       -> manual framing (F.unfold) + hann + a REAL DFT (precomputed
                                   cos/sin basis matmul) magnitude, then the same librosa mel
                                   basis. torch.stft, torch.fft.rfft AND torch.fft.fft are all
                                   blocked in the legacy ONNX exporter (`aten::fft_fft` has no
                                   opset-18 symbolic), so the STFT is a pure Gemm — exportable at
                                   any opset, runs in every ORT build, and fp16-converts cleanly.

Numerics: the phase integral is accumulated in FLOAT64, then cast back to float32. With a
float32 cumsum, torch and ONNX Runtime drift apart at the ~1e3-1e4 rad a full phrase reaches
(different summation order), and sin(k*phase) amplifies that drift into a torch-vs-ORT mel
MAE of ~0.15+ that grows with length; float64 pins the two together (~0.03). The training
excitation accumulates in float32, but the resulting x0 differs by only ~0.005 mel and is
merely the flow's starting point — details at the inline note on the cumsum below.

Noise: white noise is drawn inside the graph via `randn_like` (exports at opset 18), so the
exported graph needs NO `z` input — matching the [tokens, durations, f0] contract.
`deterministic=True` zeroes the noise (parity/regression).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _harm_rms(n_harm: int, decay: float = 1.0) -> float:
    """RMS of Sum_{k=1..K}(1/k^decay)*sin(k*phi). Identical to harmonic_excitation._harm_rms."""
    return math.sqrt(sum((1.0 / k ** decay) ** 2 for k in range(1, n_harm + 1)) / 2.0)


class HarmonicNoiseExcitationONNX(nn.Module):
    def __init__(self, n_harm: int = 50, sr: int = 44100, exc_hop: int = 256,
                 scale: float = 0.15, noise_ratio: float = 0.05, harm_decay: float = 1.0,
                 n_fft: int = 2048, win: int = 2048, n_mels: int = 128,
                 fmin: float = 40.0, fmax: float = 16000.0,
                 deterministic: bool = False):
        super().__init__()
        import librosa
        self.sr = int(sr)
        self.hop = int(exc_hop)
        self.n_fft = int(n_fft)
        self.win = int(win)
        self.n_mels = int(n_mels)
        self.scale = float(scale)
        self.noise_ratio = float(noise_ratio)
        self.harm_decay = float(harm_decay)
        self.n_harm = int(n_harm)
        self.nyq = sr / 2.0
        self.deterministic = bool(deterministic)
        self.harm_rms = _harm_rms(n_harm, harm_decay)
        self.ln2 = math.log(2.0)
        # compile-time-constant buffers (so the graph has no data-dependent shapes)
        k = torch.arange(1, n_harm + 1, dtype=torch.float32)            # [K]
        self.register_buffer("harm_k", k.view(1, n_harm, 1))            # [1,K,1]
        self.register_buffer("harm_amp", (1.0 / k ** harm_decay).view(1, n_harm, 1))
        self.register_buffer("hann", torch.hann_window(win, periodic=True))
        mb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mb.astype(np.float32)))  # [n_mels, F]
        # Real DFT basis (== torch.fft.fft magnitude for the first F bins). X[f]=Σ_t x[t]e^{-2πi ft/N}
        # -> re[f]=Σ x[t]cos(2π ft/N), im[f]=-Σ x[t]sin(2π ft/N). Precompute [n_fft, F] cos/sin.
        Fb = n_fft // 2 + 1
        t = torch.arange(n_fft, dtype=torch.float64).view(n_fft, 1)
        f = torch.arange(Fb, dtype=torch.float64).view(1, Fb)
        ang = 2.0 * math.pi * t * f / n_fft                                         # [n_fft, F]
        self.register_buffer("dft_cos", torch.cos(ang).to(torch.float32))          # [n_fft, F]
        self.register_buffer("dft_sin", (-torch.sin(ang)).to(torch.float32))       # [n_fft, F]

    # ── STFT magnitude via real DFT matmul (ONNX-safe; == torch.stft(center=False).abs()) ──
    def _stft_mag(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, L] -> |STFT| [B, n_fft//2+1, T'] (reflect-pad (n_fft-hop)//2, hann, center=False)."""
        pad = (self.n_fft - self.hop) // 2
        w = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)          # [B, L+2pad]
        # frame with im2col: [B,1,Lp,1] kernel (n_fft,1) stride (hop,1) -> [B, n_fft, T']
        frames = F.unfold(w[:, None, :, None], kernel_size=(self.n_fft, 1),
                          stride=(self.hop, 1))                                     # [B, n_fft, T']
        frames = frames.transpose(1, 2) * self.hann                                # [B, T', n_fft]
        re = torch.matmul(frames, self.dft_cos)                                    # [B, T', F]
        im = torch.matmul(frames, self.dft_sin)                                    # [B, T', F]
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-12)                                # [B, T', F]
        return mag.transpose(1, 2)                                                 # [B, F, T']

    def _harm_sum(self, phase: torch.Tensor, f0_up: torch.Tensor) -> torch.Tensor:
        """Sum_k amp_k * sin(k*phase) * antialias_k  ->  [B, n], one harmonic at a time.

        The obvious form builds `[B, n_harm, n]` and reduces it. At n_harm=256 and 44.1 kHz
        that is ~260 MB per second of audio, and several such tensors are live at once (the
        sine, the two multiplies, the anti-alias compare and its cast), so peak memory grew
        with the input: measured ~87 MB per second of audio, i.e. 2.55 GB for a 24-second
        phrase. `leapsinger.modules.harmonic_excitation` avoids this in torch by fusing under
        torch.compile, or by falling back to its own `_harm_sum_loop`; neither survives the
        export, because ORT does not fuse an elementwise chain into a reduction.

        Accumulating into `[B, n]` keeps the peak at O(n) regardless of n_harm. The loop is a
        plain Python loop on purpose: `n_harm` is a build-time constant, so tracing unrolls it into
        n_harm straight-line blocks and **never materialises `[B, n_harm, n]`**. That works
        under the legacy exporter, which is what this module must keep using — OpenUTAU ships
        an older ONNX Runtime, hence `dynamo=False` in export/cli.py, and the ONNX `Scan` that
        a dynamic loop would need is not available there.

        The trip count is unrolled, not dynamic, but `n_harm` is fixed at construction, so
        nothing about the input length is baked in.

        Sequential accumulation does not reproduce `torch.sum`'s pairwise order bit for bit:
        measured -132 dB on the excitation waveform at n_harm=256 (-136 dB at 50), constant
        from 2.3 s to 23.2 s. That is float32 rounding, 36 dB below 16-bit quantization noise.
        """
        dt = phase.dtype
        nyq = float(self.nyq)
        decay = float(self.harm_decay)
        acc = torch.zeros_like(phase)
        for i in range(int(self.n_harm)):
            k = float(i + 1)                          # Python の即値 = グラフ上は定数
            a = 1.0 / k ** decay
            aa = (k * f0_up < nyq).to(dt)             # [B, n] anti-alias gate
            acc = acc + a * torch.sin(k * phase) * aa
        return acc

    def forward(self, f0_logf0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        """f0_logf0 [B,T] (log2 Hz, CONTINUOUS/gap-less), voiced [B,T] (1=voiced) -> ln-mel [B,n_mels,T].
        For the uv-free / DiffSinger path `voiced` is all-ones; for the optional path it is the
        real voicing gate. F0 must already be continuous (no unvoiced zeros)."""
        if f0_logf0.dim() == 1:
            f0_logf0 = f0_logf0[None]
            voiced = voiced[None]
        B, T = f0_logf0.shape
        n = T * self.hop
        f0_hz = torch.exp(self.ln2 * f0_logf0)                                     # exp2 -> exp
        f0_up = F.interpolate(f0_hz[:, None], size=n, mode="linear", align_corners=False)[:, 0]
        v_up = F.interpolate(voiced[:, None], size=n, mode="linear", align_corners=False)[:, 0]
        # phase = 2*pi * cumsum(f0/sr), accumulated in FLOAT64. Over n = T*hop (~1e5) samples the
        # phase reaches ~1e3-1e4 rad; in float32 an ORT cumsum and a torch cumsum drift apart at
        # that magnitude (different summation order), and sin(k*phase) with k up to 50 amplifies the
        # drift -> torch-vs-ORT mel MAE ~0.15+ that GROWS with length. float64 pins them together
        # (torch-vs-ORT back to ~0.03). The training excitation used float32, but the resulting x0
        # differs only ~0.005 mel and it is merely the flow's starting point (the flow is robust to
        # it). The excitation stays fp32 in the exported graph (island fp16 never touches /exc), so
        # the float64 accumulation costs nothing in the fp16 conversion.
        phase = torch.cumsum((2.0 * math.pi * f0_up / self.sr).to(torch.float64), dim=1)
        phase = phase.to(f0_logf0.dtype)                                           # back to float32
        harm = self._harm_sum(phase, f0_up)                                        # [B, n]
        harm = (harm / self.harm_rms) * v_up                                       # unit-RMS + voiced gate
        noise = torch.zeros_like(harm) if self.deterministic else torch.randn_like(harm)
        exc = self.scale * (harm + self.noise_ratio * noise)                       # [B, n]
        mag = self._stft_mag(exc)                                                  # [B, F, T']
        mel = torch.matmul(self.mel_basis, mag)                                    # [B, n_mels, T']
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel[:, :, :T]
