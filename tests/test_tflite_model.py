"""Checks on the exported model itself: the C array is a valid int8 TFLite
model with only the ops the firmware registers, and the TFLite interpreter
still gives the outputs stored in the test vectors."""
import os
import re

import numpy as np
import pytest

from vectors_to_header import read

tf = pytest.importorskip("tensorflow")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_bytes():
    src = open(os.path.join(ROOT, "core", "model_data.cc")).read()
    body = src.split("g_model_tflite[] = {")[1].split("};")[0]
    data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})", body))
    n = int(re.search(r"g_model_tflite_len = (\d+);", src).group(1))
    assert len(data) == n
    return data


def interp():
    it = tf.lite.Interpreter(
        model_content=model_bytes(),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    return it


def test_only_ops_the_firmware_registers():
    ops = {o["op_name"] for o in interp()._get_ops_details()}
    assert ops == {"FULLY_CONNECTED", "SOFTMAX"}


def test_int8_in_and_out():
    it = interp()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    assert i["dtype"] == np.int8 and o["dtype"] == np.int8
    assert tuple(i["shape"]) == (1, 32) and tuple(o["shape"]) == (1, 4)


def test_model_is_small():
    assert len(model_bytes()) < 8 * 1024


def test_vectors_reproduce():
    it = interp()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    v = read(os.path.join(ROOT, "host", "vectors", "test_vectors.bin"))
    for r in v:
        it.set_tensor(i["index"], r["inp"][None, :])
        it.invoke()
        assert np.array_equal(it.get_tensor(o["index"])[0], r["out"])
