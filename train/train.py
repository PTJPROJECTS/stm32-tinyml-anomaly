"""Train, quantize, evaluate and export the bearing fault model.

    python -m train.train            # main run, writes core/ and host/vectors
    python -m train.train --cross-load   # also the leave-one-load-out check
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

from . import config, cwru, export_c, features, quantize
from . import model as model_lib


def confusion(y, p, n):
    m = np.zeros((n, n), np.int64)
    np.add.at(m, (y, p), 1)
    return m


def binary_stats(y, p):
    """Normal (0) vs any fault. A false alarm is a normal window called a
    fault, a miss is a fault window called normal."""
    normal = y == 0
    return {
        "false_alarm_rate": float(np.mean(p[normal] != 0)),
        "miss_rate": float(np.mean(p[~normal] == 0)),
    }


def fit(x_tr, y_tr, x_va, y_va, epochs, seed):
    tf.keras.utils.set_random_seed(seed)
    m = model_lib.build()
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True)]
    h = m.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs,
              batch_size=64, verbose=0, callbacks=cb)
    return m, len(h.history["loss"])


def evaluate(m, tfl, x_te, y_te):
    pf = m.predict(x_te, verbose=0).argmax(1)
    run = quantize.Int8Runner(tfl)
    xq = run.quantize_input(x_te)
    yq = run.run_q(xq)
    pq = yq.argmax(1)
    return pf, pq, xq, yq, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross-load", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cwru.download(args.data)
    ds = cwru.build_dataset(args.data)
    w_tr, y_tr, _ = ds["train"]
    w_va, y_va, _ = ds["val"]
    w_te, y_te, l_te = ds["test"]
    f_tr, f_va, f_te = (features.extract(w) for w in (w_tr, w_va, w_te))

    mean = f_tr.mean(0).astype(np.float32)
    inv_std = (1.0 / (f_tr.std(0) + 1e-6)).astype(np.float32)

    def norm(f):
        return ((f - mean) * inv_std).astype(np.float32)

    x_tr, x_va, x_te = norm(f_tr), norm(f_va), norm(f_te)
    print(f"windows train/val/test: {len(x_tr)}/{len(x_va)}/{len(x_te)}")

    m, ep = fit(x_tr, y_tr, x_va, y_va, args.epochs, args.seed)
    tfl = quantize.to_int8_tflite(m, x_tr)
    tfl_f = quantize.to_float_tflite(m)
    pf, pq, xq, yq, run = evaluate(m, tfl, x_te, y_te)

    n_cls = len(config.CLASSES)
    cm = confusion(y_te, pq, n_cls)
    metrics = {
        "dataset": {"train": int(len(x_tr)), "val": int(len(x_va)), "test": int(len(x_te)),
                    "test_per_class": np.bincount(y_te, minlength=n_cls).tolist()},
        "epochs_run": ep,
        "params": int(m.count_params()),
        "macs": model_lib.macs(m),
        "float_acc": float(np.mean(pf == y_te)),
        "int8_acc": float(np.mean(pq == y_te)),
        "int8_vs_float_agreement": float(np.mean(pq == pf)),
        "int8_binary": binary_stats(y_te, pq),
        "int8_confusion": cm.tolist(),
        "int8_recall_per_class": dict(zip(config.CLASSES,
                                          (cm.diagonal() / cm.sum(1)).round(4).tolist())),
        "int8_acc_per_load": {str(L): float(np.mean(pq[l_te == L] == y_te[l_te == L]))
                              for L in range(4)},
        "tflite_int8_bytes": len(tfl),
        "tflite_float_bytes": len(tfl_f),
        "input_quant": [float(run.in_scale), int(run.in_zp)],
        "output_quant": [float(run.out_scale), int(run.out_zp)],
    }

    if args.cross_load:
        res = {}
        for L in range(4):
            wa, ya, wb, yb = cwru.build_cross_load(args.data, test_load=L)
            fa, fb = features.extract(wa), features.extract(wb)
            mu, isd = fa.mean(0), 1.0 / (fa.std(0) + 1e-6)
            xa, xb = (fa - mu) * isd, (fb - mu) * isd
            # last 15 % of the training windows as val, good enough here
            k = int(len(xa) * 0.85)
            perm = np.random.default_rng(args.seed).permutation(len(xa))
            tr_i, va_i = perm[:k], perm[k:]
            mc, _ = fit(xa[tr_i], ya[tr_i], xa[va_i], ya[va_i], args.epochs, args.seed)
            tq = quantize.to_int8_tflite(mc, xa[tr_i])
            _, pqc, _, _, _ = evaluate(mc, tq, xb.astype(np.float32), yb)
            res[str(L)] = float(np.mean(pqc == yb))
            print(f"cross-load: test on {L} hp -> int8 acc {res[str(L)]:.4f}")
        metrics["cross_load_int8_acc"] = res

    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "model_int8.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(args.out, "model_float.tflite"), "wb") as f:
        f.write(tfl_f)

    # full test set for the C side (not committed, 1.1 MB)
    export_c.write_vectors(os.path.join(args.out, "test_all.bin"), w_te, f_te, xq, yq, y_te)

    if args.no_export:
        return
    export_c.write_model(tfl, mean, inv_std, config.CLASSES,
                         "core/model_data.h", "core/model_data.cc")
    # 16 windows per class for the committed host test
    pick = np.concatenate([np.flatnonzero(y_te == c)[::max(1, (y_te == c).sum() // 16)][:16]
                           for c in range(n_cls)])
    os.makedirs("host/vectors", exist_ok=True)
    export_c.write_vectors("host/vectors/test_vectors.bin", w_te[pick], f_te[pick],
                           xq[pick], yq[pick], y_te[pick])
    with open("docs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("exported core/model_data.{h,cc}, host/vectors/test_vectors.bin, docs/metrics.json")


if __name__ == "__main__":
    main()
