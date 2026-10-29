"""The numbers in train/config.py and core/vib_features.h have to agree,
otherwise the MCU computes different features than the model was trained on."""
import os
import re

from train import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def c_defines():
    src = open(os.path.join(ROOT, "core", "vib_features.h")).read()
    return dict(re.findall(r"#define\s+(FEAT_\w+)\s+(.+?)\s*(?:/\*.*)?$", src, re.M))


def test_sizes_match():
    d = c_defines()
    assert int(d["FEAT_FS_HZ"]) == config.FS_HZ
    assert int(d["FEAT_WIN"]) == config.WIN
    assert int(d["FEAT_N_BANDS"]) == config.N_BANDS
    assert int(d["FEAT_BINS_PER_BAND"]) == config.BINS_PER_BAND
    assert int(d["FEAT_ADC_MID"]) == config.ADC_MID


def test_lsb_matches():
    expr = c_defines()["FEAT_ADC_LSB_G"].replace("f", "")
    assert abs(eval(expr) - config.ADC_LSB_G) < 1e-12


def test_model_header_matches():
    src = open(os.path.join(ROOT, "core", "model_data.h")).read()
    assert f"#define MODEL_N_FEATURES {config.N_BANDS}" in src
    assert f"#define MODEL_N_CLASSES {len(config.CLASSES)}" in src
