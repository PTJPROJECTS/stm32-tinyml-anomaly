"""Full int8 post-training quantization, and a small helper to run the
resulting .tflite on the host with the TFLite interpreter."""
import numpy as np
import tensorflow as tf


def to_int8_tflite(model, rep_x, n_rep=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rep_x), size=min(n_rep, len(rep_x)), replace=False)

    def rep():
        for i in idx:
            yield [rep_x[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def to_float_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


class Int8Runner:
    def __init__(self, tflite_bytes):
        # Reference kernels so the numbers are the same kind of integer math
        # TFLM does on the MCU.
        self.it = tf.lite.Interpreter(
            model_content=tflite_bytes,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def quantize_input(self, x):
        q = np.round(x / self.in_scale) + self.in_zp
        return np.clip(q, -128, 127).astype(np.int8)

    def run_q(self, xq):
        """xq: [N, n_in] int8 -> [N, n_out] int8"""
        res = np.empty((len(xq), self.out["shape"][-1]), np.int8)
        for i in range(len(xq)):
            self.it.set_tensor(self.inp["index"], xq[i:i + 1])
            self.it.invoke()
            res[i] = self.it.get_tensor(self.out["index"])[0]
        return res

    def dequant_out(self, yq):
        return (yq.astype(np.float32) - self.out_zp) * self.out_scale
