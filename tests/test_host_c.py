"""Runs the C host programs (make all first, the Makefile's pytest target
does that)."""
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")


def run(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def test_c_feature_unit_tests():
    r = run([os.path.join(BUILD, "test_features")])
    assert r.returncode == 0, r.stdout
    assert "PASS" in r.stdout


def parse(out):
    vals = {}
    vals["feat_err"] = float(re.search(r"feature max \|err\|\s*: (\S+)", out).group(1))
    vals["mismatch"] = int(re.search(r"tflm vs tflite out : (\d+)", out).group(1))
    vals["cls_diff"] = int(re.search(r"class vs python\s*: (\d+)", out).group(1))
    vals["acc"] = float(re.search(r"accuracy \(C path\)\s*: (\S+)", out).group(1))
    vals["arena"] = int(re.search(r"arena used\s*: (\d+)", out).group(1))
    return vals


def test_committed_vectors_match_python():
    r = run([os.path.join(BUILD, "tvec_check"), "host/vectors/test_vectors.bin", "-q"])
    assert r.returncode == 0, r.stdout + r.stderr
    v = parse(r.stdout)
    assert v["mismatch"] == 0          # TFLM bit exact vs TFLite interpreter
    assert v["cls_diff"] == 0
    assert v["feat_err"] < 1e-3
    assert v["acc"] >= 0.98
    assert v["arena"] <= 3072


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "out", "test_all.bin")),
                    reason="run `make train` first for the full test set")
def test_full_test_set_in_c():
    r = run([os.path.join(BUILD, "tvec_check"), "out/test_all.bin", "-q"])
    assert r.returncode == 0, r.stdout
    assert parse(r.stdout)["acc"] >= 0.98
