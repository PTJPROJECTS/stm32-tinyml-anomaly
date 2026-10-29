#ifndef ANOMALY_MODEL_H
#define ANOMALY_MODEL_H

/* Thin C API around TFLite Micro so main.cpp and the host test don't need
 * to know about the interpreter. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 0 on success. Sets up the op resolver, interpreter and tensor arena. */
int model_init(void);

/* Bytes of the arena actually used after AllocateTensors(). */
size_t model_arena_used(void);
size_t model_arena_size(void);

float model_input_scale(void);
int model_input_zero_point(void);

/* in: MODEL_N_FEATURES int8, out: MODEL_N_CLASSES int8 scores (softmax,
 * scale 1/256, zero point -128). Returns argmax, or -1 on error. */
int model_invoke(const int8_t *in, int8_t *out);

/* Full pipeline: raw ADC window -> features -> int8 -> model. */
int model_classify_window(const uint16_t *win, int8_t *scores);

#ifdef __cplusplus
}
#endif

#endif
