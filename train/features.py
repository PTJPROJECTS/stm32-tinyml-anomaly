"""Feature extraction, numpy version. core/vib_features.c does the same thing in
float32 on the MCU, and the tests compare the two.

window (uint16 ADC counts)
  -> g:      (count - 2048) * LSB
  -> remove mean
  -> Hann window
  -> 512 point FFT, power of bins 1..256
  -> 32 bands of 8 bins, sum
  -> log10(band + 1e-9)
"""
import numpy as np

from . import config

EPS = 1e-9


def hann(n=config.WIN):
    # periodic Hann, same formula as the C code
    k = np.arange(n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float32)


def extract(win_adc):
    """win_adc: [..., WIN] uint16. Returns float32 [..., N_BANDS]."""
    x = (win_adc.astype(np.float32) - config.ADC_MID) * np.float32(config.ADC_LSB_G)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x * hann()
    spec = np.fft.rfft(x.astype(np.float64), axis=-1)
    p = (spec.real ** 2 + spec.imag ** 2)[..., 1:config.N_BANDS * config.BINS_PER_BAND + 1]
    p = p.reshape(*p.shape[:-1], config.N_BANDS, config.BINS_PER_BAND).sum(-1)
    return np.log10(p + EPS).astype(np.float32)
