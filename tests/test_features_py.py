import numpy as np

from train import config, cwru, features


def tone_adc(hz, amp_g=1.0):
    k = np.arange(config.WIN)
    g = amp_g * np.sin(2 * np.pi * hz * k / config.FS_HZ)
    return cwru.to_adc(g)


def test_shape_and_dtype():
    w = np.full((3, config.WIN), config.ADC_MID, np.uint16)
    f = features.extract(w)
    assert f.shape == (3, config.N_BANDS)
    assert f.dtype == np.float32


def test_tone_lands_in_its_band():
    bin_hz = config.FS_HZ / config.WIN
    for b in (4, 30, 100, 200, 250):
        f = features.extract(tone_adc(b * bin_hz))
        assert f.argmax() == (b - 1) // config.BINS_PER_BAND


def test_dc_offset_is_removed():
    w = tone_adc(30 * config.FS_HZ / config.WIN, 0.5).astype(np.int32)
    a = features.extract(w.astype(np.uint16))
    b = features.extract((w + 300).astype(np.uint16))
    assert np.allclose(a, b, atol=1e-4)


def test_adc_clips_at_full_scale():
    x = np.array([-100.0, 0.0, 100.0])
    assert cwru.to_adc(x).tolist() == [0, config.ADC_MID, 4095]


def test_windows_do_not_cross_split():
    sig = np.arange(10000)
    tr, va, te = cwru.split_recording(sig)
    assert tr[-1] < va[0] and va[-1] < te[0]
    w = cwru.windows(tr)
    assert w.shape[1] == config.WIN and w[-1, -1] <= tr[-1]
