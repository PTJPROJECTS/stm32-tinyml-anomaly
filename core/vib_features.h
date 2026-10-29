#ifndef VIB_FEATURES_H
#define VIB_FEATURES_H

/* Vibration window -> 32 log band energies. Same math as train/features.py.
 * Plain C99, float32, no malloc. Used on the host and on the STM32. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FEAT_FS_HZ          12000
#define FEAT_WIN            512
#define FEAT_N_BANDS        32
#define FEAT_BINS_PER_BAND  8
#define FEAT_ADC_MID        2048
#define FEAT_ADC_LSB_G      (8.0f / 2048.0f)   /* +/-8 g over 12 bits */

/* Builds the Hann and twiddle tables. Call once before anything else. */
void features_init(void);

/* win: FEAT_WIN raw 12 bit ADC counts. bands: FEAT_N_BANDS outputs. */
void features_extract(const uint16_t *win, float *bands);

/* (bands - mean) * inv_std, then quantized with the model's input scale and
 * zero point, clamped to int8. */
void features_quantize(const float *bands, const float *mean,
                       const float *inv_std, float scale, int zero_point,
                       int8_t *out);

#ifdef __cplusplus
}
#endif

#endif
