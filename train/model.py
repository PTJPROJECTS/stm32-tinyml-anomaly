import tensorflow as tf

from . import config


def build(n_in=config.N_BANDS, n_out=len(config.CLASSES), hidden=(24, 16)):
    """Small MLP. Three FullyConnected ops after conversion, all of them
    have CMSIS-NN int8 kernels."""
    inp = tf.keras.Input(shape=(n_in,), name="bands")
    x = inp
    for i, h in enumerate(hidden):
        x = tf.keras.layers.Dense(h, activation="relu", name=f"fc{i + 1}")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(n_out, name="logits")(x)
    out = tf.keras.layers.Softmax(name="prob")(out)
    return tf.keras.Model(inp, out)


def macs(model):
    n = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            k = layer.kernel.shape
            n += int(k[0]) * int(k[1])
    return n
