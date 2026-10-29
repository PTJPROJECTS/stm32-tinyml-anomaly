#!/usr/bin/env bash
# Generate a flat TFLite Micro source tree (with the CMSIS-NN kernels) into
# third_party/tflm. Pinned to one tflite-micro commit so builds don't drift.
# The same tree is used for the host test build and the STM32 firmware.
set -euo pipefail

TFLM_SHA=4bb5a53e417b597ce5c2af2079aa8c42eb5fe9da
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/tflite-micro"
OUT="$ROOT/third_party/tflm"

if [ -f "$OUT/.sha" ] && [ "$(cat "$OUT/.sha")" = "$TFLM_SHA" ]; then
    echo "third_party/tflm already at $TFLM_SHA"
    exit 0
fi

if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/tensorflow/tflite-micro.git
fi
git -C "$SRC" fetch -q --depth 1 origin "$TFLM_SHA"
git -C "$SRC" checkout -q FETCH_HEAD

rm -rf "$OUT"
# create_tflm_tree downloads flatbuffers, CMSIS, CMSIS-NN etc. by itself
(cd "$SRC" && python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
    --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn" \
    "$OUT")
echo "$TFLM_SHA" > "$OUT/.sha"

# PlatformIO picks the library up from firmware/lib/tflm
mkdir -p "$ROOT/firmware/lib"
ln -sfn ../../third_party/tflm "$ROOT/firmware/lib/tflm"
cp "$ROOT/tools/tflm_library.json" "$OUT/library.json"
echo "done: $OUT"
