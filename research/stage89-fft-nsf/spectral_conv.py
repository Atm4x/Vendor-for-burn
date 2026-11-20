"""Exact-topology overlap-save spectral Conv1d research reference (CPU only)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SpectralPlan:
    channels_in: int
    channels_out: int
    kernel: int
    dilation: int
    output_frames: int
    nfft: int

    @property
    def effective_kernel(self) -> int:
        return (self.kernel - 1) * self.dilation + 1

    @property
    def valid_frames(self) -> int:
        return self.nfft - self.effective_kernel + 1

    @property
    def blocks(self) -> int:
        return math.ceil(self.output_frames / self.valid_frames)

    @property
    def bins(self) -> int:
        return self.nfft // 2 + 1

    @property
    def spectral_weight_bytes(self) -> int:
        return self.channels_in * self.channels_out * self.bins * 8

    @property
    def direct_flops(self) -> int:
        return 2 * self.output_frames * self.channels_in * self.channels_out * self.kernel

    @property
    def spectral_flops(self) -> float:
        # Three-real-matmul complex multiply: three real matmuls + scalar additions.
        # An idealized count, not a measure of Tensor Core dispatch or bandwidth.
        matmul = 6 * self.blocks * self.bins * self.channels_in * self.channels_out
        recompose = 4 * self.blocks * self.bins * self.channels_out
        # Conservative real-FFT approximation: both forward and inverse transforms.
        transforms = 5 * self.blocks * self.nfft * math.log2(self.nfft) * (self.channels_in + self.channels_out)
        return matmul + recompose + transforms

    @property
    def ideal_flop_ratio(self) -> float:
        return self.direct_flops / self.spectral_flops


def candidate_plans(channels_in: int, channels_out: int, kernel: int,
                    dilation: int, output_frames: int,
                    nffts: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
                    max_spectral_weight_bytes: int | None = None) -> list[SpectralPlan]:
    if min(channels_in, channels_out, kernel, dilation, output_frames) <= 0:
        raise ValueError("dimensions must be positive")
    if kernel % 2 == 0:
        raise ValueError("same-padding research path requires an odd kernel")
    plans = []
    for nfft in nffts:
        if nfft <= 0 or nfft & (nfft - 1):
            raise ValueError("FFT length must be power of two")
        p = SpectralPlan(channels_in, channels_out, kernel, dilation, output_frames, nfft)
        if p.valid_frames <= 0:
            continue
        if max_spectral_weight_bytes is not None and p.spectral_weight_bytes > max_spectral_weight_bytes:
            continue
        plans.append(p)
    return sorted(plans, key=lambda p: (p.spectral_flops, p.spectral_weight_bytes))


class SpectralConv1d:
    """Pretransforms fixed weights and replaces cross-correlation with overlap-save FFT.

    This is a mathematical CPU reference, NOT a GPU kernel and NOT a production
    replacement.  Full float32 convolution differs in reduction order from FFT.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray | None,
                 dilation: int, nfft: int):
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 3:
            raise ValueError("weight must be [out,in,kernel]")
        o, i, k = w.shape
        plan = SpectralPlan(i, o, k, dilation, 1, nfft)
        if k % 2 == 0 or plan.valid_frames <= 0:
            raise ValueError("odd kernel and FFT length >= effective kernel required")
        if bias is not None and np.shape(bias) != (o,):
            raise ValueError("bias shape must match output channels")
        h = np.zeros((o, i, nfft), dtype=np.float32)
        # PyTorch Conv1d is cross-correlation. Reverse taps for convolution.
        h[:, :, np.arange(k) * dilation] = w[:, :, ::-1]
        self.freq_weights = np.moveaxis(
            np.fft.rfft(h, axis=-1).astype(np.complex64), -1, 0).copy()
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.channels_in = i
        self.channels_out = o
        self.kernel = k
        self.dilation = dilation
        self.nfft = nfft
        self.valid_frames = plan.valid_frames
        self.pad = (plan.effective_kernel - 1) // 2

    def forward(self, x: np.ndarray, algorithm: str = "complex",
                windows_per_batch: int = 1) -> np.ndarray:
        """Run spectral convolution with GEMM column width = windows_per_batch * batch.

        Production GPU implementations would fuse the per-frequency work into
        batched MMA dispatches rather than launching one matrix-vector GEMM for
        each (frequency, window) pair.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] != self.channels_in:
            raise ValueError("input must be [batch,in,time]")
        if algorithm not in ("complex", "three_real_gemm"):
            raise ValueError("unknown spectral matmul algorithm")
        if windows_per_batch <= 0:
            raise ValueError("windows_per_batch must be positive")
        batch, _, total = x.shape
        padded = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)))
        result = np.empty((batch, self.channels_out, total), dtype=np.float32)
        starts = list(range(0, total, self.valid_frames))
        for first in range(0, len(starts), windows_per_batch):
            run = starts[first:first + windows_per_batch]
            windows = len(run)
            segments = np.zeros((windows, batch, self.channels_in, self.nfft),
                                dtype=np.float32)
            for windex, start in enumerate(run):
                available = min(self.nfft, padded.shape[2] - start)
                if available > 0:
                    segments[windex, :, :, :available] = padded[:, :, start:start + available]
            f = np.fft.rfft(segments.reshape(windows * batch, self.channels_in,
                                                self.nfft), axis=-1).astype(np.complex64)
            # [frequency,in,windows*batch] is a true matrix (not a matrix-vector)
            # for windows_per_batch > 1.
            f = np.transpose(f, (2, 1, 0))
            w = self.freq_weights  # [frequency,out,in]
            if algorithm == "complex":
                y_freq = np.matmul(w, f)
            else:
                a, b = w.real, w.imag
                c, d = f.real, f.imag
                ac = np.matmul(a, c)
                bd = np.matmul(b, d)
                ab_cd = np.matmul(a + b, c + d)
                y_freq = (ac - bd) + 1j * (ab_cd - ac - bd)
            y_freq = np.transpose(y_freq, (2, 1, 0))
            blocks = np.fft.irfft(y_freq, self.nfft, axis=-1).real
            blocks = blocks.reshape(windows, batch, self.channels_out, self.nfft)
            for windex, start in enumerate(run):
                length = min(self.valid_frames, total - start)
                result[:, :, start:start + length] = blocks[windex, :, :,
                    self.nfft - self.valid_frames:self.nfft - self.valid_frames + length]
        if self.bias is not None:
            result += self.bias[None, :, None]
        return result
