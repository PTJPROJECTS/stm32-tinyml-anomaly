"""Harder checks on the exported int8 model (no retraining):

1. Unseen fault size: the model only saw 0.007" faults. Here it gets the
   0.021" recordings (3x bigger defect, different vibration signature).
2. Added sensor noise: white noise on the test windows at a few SNRs,
   before the ADC quantization, like a noisier front end would give.

    python -m train.robustness
"""
import json
import os

import numpy as np
import scipy.io

from . import config, cwru, features, quantize

FILES_021 = {209: (1, 0), 210: (1, 1), 211: (1, 2), 212: (1, 3),
             222: (2, 0), 223: (2, 1), 224: (2, 2), 225: (2, 3),
             234: (3, 0), 235: (3, 1), 236: (3, 2), 237: (3, 3)}


def load_021(data_dir):
    w, y = [], []
    for n, (label, _) in FILES_021.items():
        path = os.path.join(data_dir, f"{n}.mat")
        if not os.path.exists(path):
            import urllib.request
            urllib.request.urlretrieve(cwru.BASE_URL.format(n), path)
        x = scipy.io.loadmat(path)[f"X{n:03d}_DE_time"].ravel()
        win = cwru.windows(cwru.to_adc(x))
        w.append(win)
        y.append(np.full(len(win), label))
    return np.concatenate(w), np.concatenate(y)


def main(data_dir="data", out="out"):
    tfl = open(os.path.join(out, "model_int8.tflite"), "rb").read()
    run = quantize.Int8Runner(tfl)

    # recompute the normalization the same way train.py did
    ds = cwru.build_dataset(data_dir)
    f_tr = features.extract(ds["train"][0])
    mean, inv_std = f_tr.mean(0), 1.0 / (f_tr.std(0) + 1e-6)

    def predict(win_adc):
        x = ((features.extract(win_adc) - mean) * inv_std).astype(np.float32)
        return run.run_q(run.quantize_input(x)).argmax(1)

    res = {}
    w21, y21 = load_021(data_dir)
    p = predict(w21)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y21, p), 1)
    res["unseen_0.021in"] = {
        "windows": int(len(y21)),
        "acc_fault_type": float(np.mean(p == y21)),
        "detected_as_fault": float(np.mean(p != 0)),
        "confusion_rows_true_cols_pred": cm.tolist(),
    }
    print("0.021in faults:", res["unseen_0.021in"])

    # noise test on the normal test split
    rng = np.random.default_rng(0)
    w_te, y_te, _ = ds["test"]
    g = (w_te.astype(np.float64) - config.ADC_MID) * config.ADC_LSB_G
    res["noise"] = {}
    for snr_db in (30, 20, 10, 5, 0):
        sig_rms = g.std(axis=1, keepdims=True)
        noise = rng.standard_normal(g.shape) * sig_rms / (10 ** (snr_db / 20))
        wn = cwru.to_adc(g + noise)
        pn = predict(wn)
        res["noise"][f"{snr_db}dB"] = {
            "acc": float(np.mean(pn == y_te)),
            "false_alarm_rate": float(np.mean(pn[y_te == 0] != 0)),
            "miss_rate": float(np.mean(pn[y_te != 0] == 0)),
        }
        print(f"SNR {snr_db:>2} dB:", res["noise"][f"{snr_db}dB"])
    with open("docs/robustness.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
