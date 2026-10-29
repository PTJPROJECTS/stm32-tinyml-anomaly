#!/usr/bin/env python3
"""Plot the mean log band energy per class (test split) to docs/band_energies.png."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import config, cwru, features  # noqa: E402


def main():
    ds = cwru.build_dataset("data")
    w, y, _ = ds["test"]
    f = features.extract(w)
    band_khz = (np.arange(config.N_BANDS) + 0.5) * config.BINS_PER_BAND * config.FS_HZ / config.WIN / 1000
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    for c, name in enumerate(config.CLASSES):
        m, s = f[y == c].mean(0), f[y == c].std(0)
        ax.plot(band_khz, m, label=name.replace("_", " "))
        ax.fill_between(band_khz, m - s, m + s, alpha=0.15)
    ax.set_xlabel("band centre (kHz)")
    ax.set_ylabel("log10 band energy")
    ax.set_title("Model input: 32 log band energies, CWRU test split (mean +/- 1 std)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig.savefig("docs/band_energies.png")
    print("wrote docs/band_energies.png")


if __name__ == "__main__":
    main()
