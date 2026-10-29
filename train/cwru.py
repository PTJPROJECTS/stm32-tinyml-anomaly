"""Download and load the Case Western Reserve University (CWRU) bearing data.

I only use the 12 kHz drive-end accelerometer channel, 0.007 inch faults, and
all four motor loads (0-3 hp). That is 16 recordings, about 70 MB.

Source: https://engineering.case.edu/bearingdatacenter
"""
import os
import urllib.request

import numpy as np
import scipy.io
import scipy.signal

from . import config

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# file number -> (class index, load in hp)
FILES = {
    97: (0, 0), 98: (0, 1), 99: (0, 2), 100: (0, 3),     # normal baseline
    105: (1, 0), 106: (1, 1), 107: (1, 2), 108: (1, 3),  # inner race 0.007"
    118: (2, 0), 119: (2, 1), 120: (2, 2), 121: (2, 3),  # ball 0.007"
    130: (3, 0), 131: (3, 1), 132: (3, 2), 133: (3, 3),  # outer race 0.007" @6:00
}

# The normal baseline files are recorded at 48 kHz, the 12k drive-end fault
# files at 12 kHz. Normal gets decimated by 4 so everything is 12 kHz.
NORMAL_DECIMATE = 4


def download(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for n in FILES:
        path = os.path.join(data_dir, f"{n}.mat")
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            continue
        print(f"downloading {n}.mat")
        urllib.request.urlretrieve(BASE_URL.format(n), path)


def load_signal(n, data_dir="data"):
    """Drive-end vibration in g, resampled to 12 kHz."""
    mat = scipy.io.loadmat(os.path.join(data_dir, f"{n}.mat"))
    # 99.mat also contains the X098 arrays, so pick the key by file number.
    key = f"X{n:03d}_DE_time"
    x = mat[key].astype(np.float64).ravel()
    if FILES[n][0] == 0:
        x = scipy.signal.decimate(x, NORMAL_DECIMATE, ftype="fir", zero_phase=True)
    return x


def to_adc(x_g):
    """What a 12 bit ADC would give for this signal (uint16 counts)."""
    counts = np.round(x_g / config.ADC_LSB_G) + config.ADC_MID
    return np.clip(counts, 0, (1 << config.ADC_BITS) - 1).astype(np.uint16)


def windows(sig, win=config.WIN, hop=config.HOP):
    n = (len(sig) - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return sig[idx]


def split_recording(adc, fractions=(0.70, 0.15, 0.15)):
    """Split one recording in time: first 70 % train, then val, then test.
    Splitting before windowing means no overlapping window ends up in two
    sets."""
    n = len(adc)
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return adc[:a], adc[a:b], adc[b:]


def build_dataset(data_dir="data", hop=config.HOP):
    """Returns dict split -> (windows uint16 [N, WIN], labels, loads)."""
    out = {s: ([], [], []) for s in ("train", "val", "test")}
    for n, (label, load) in FILES.items():
        adc = to_adc(load_signal(n, data_dir))
        for split, part in zip(("train", "val", "test"), split_recording(adc)):
            w = windows(part, hop=hop)
            out[split][0].append(w)
            out[split][1].append(np.full(len(w), label, np.int64))
            out[split][2].append(np.full(len(w), load, np.int64))
    return {s: tuple(np.concatenate(v) for v in vals) for s, vals in out.items()}


def build_cross_load(data_dir="data", test_load=3, hop=config.HOP):
    """Train on three loads, test on the one held out."""
    tr, te = ([], []), ([], [])
    for n, (label, load) in FILES.items():
        w = windows(to_adc(load_signal(n, data_dir)), hop=hop)
        dst = te if load == test_load else tr
        dst[0].append(w)
        dst[1].append(np.full(len(w), label, np.int64))
    return (np.concatenate(tr[0]), np.concatenate(tr[1]),
            np.concatenate(te[0]), np.concatenate(te[1]))
