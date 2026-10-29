#include "vib_features.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG2_WIN 9

static float s_hann[FEAT_WIN];
static float s_cos[FEAT_WIN / 2];
static float s_sin[FEAT_WIN / 2];
static uint16_t s_bitrev[FEAT_WIN];

/* work buffers, 4 KB. static so they don't sit on a task stack */
static float s_re[FEAT_WIN];
static float s_im[FEAT_WIN];

void features_init(void)
{
    for (int k = 0; k < FEAT_WIN; k++) {
        /* periodic Hann, same as numpy side */
        s_hann[k] = 0.5f - 0.5f * (float)cos(2.0 * M_PI * k / FEAT_WIN);
        unsigned r = 0, v = (unsigned)k;
        for (int b = 0; b < LOG2_WIN; b++) {
            r = (r << 1) | (v & 1u);
            v >>= 1;
        }
        s_bitrev[k] = (uint16_t)r;
    }
    for (int k = 0; k < FEAT_WIN / 2; k++) {
        /* tables in double then rounded, keeps the float FFT closer to numpy */
        s_cos[k] = (float)cos(2.0 * M_PI * k / FEAT_WIN);
        s_sin[k] = (float)sin(2.0 * M_PI * k / FEAT_WIN);
    }
}

/* In-place iterative radix-2 FFT on s_re/s_im (already bit-reversed). */
static void fft512(void)
{
    for (int len = 2; len <= FEAT_WIN; len <<= 1) {
        int half = len >> 1;
        int step = FEAT_WIN / len;
        for (int i = 0; i < FEAT_WIN; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = s_cos[j * step];
                float wi = -s_sin[j * step];
                int a = i + j, b = a + half;
                float tr = s_re[b] * wr - s_im[b] * wi;
                float ti = s_re[b] * wi + s_im[b] * wr;
                s_re[b] = s_re[a] - tr;
                s_im[b] = s_im[a] - ti;
                s_re[a] += tr;
                s_im[a] += ti;
            }
        }
    }
}

void features_extract(const uint16_t *win, float *bands)
{
    /* mean in integer, exact */
    int32_t sum = 0;
    for (int k = 0; k < FEAT_WIN; k++)
        sum += win[k];
    float mean_counts = (float)sum / FEAT_WIN;

    for (int k = 0; k < FEAT_WIN; k++) {
        float g = ((float)win[k] - mean_counts) * FEAT_ADC_LSB_G;
        int r = s_bitrev[k];
        s_re[r] = g * s_hann[k];
        s_im[r] = 0.0f;
    }
    fft512();

    for (int b = 0; b < FEAT_N_BANDS; b++) {
        float acc = 0.0f;
        for (int j = 0; j < FEAT_BINS_PER_BAND; j++) {
            int bin = 1 + b * FEAT_BINS_PER_BAND + j;
            acc += s_re[bin] * s_re[bin] + s_im[bin] * s_im[bin];
        }
        bands[b] = log10f(acc + 1e-9f);
    }
}

void features_quantize(const float *bands, const float *mean,
                       const float *inv_std, float scale, int zero_point,
                       int8_t *out)
{
    for (int i = 0; i < FEAT_N_BANDS; i++) {
        float x = (bands[i] - mean[i]) * inv_std[i];
        float q = roundf(x / scale) + (float)zero_point;
        if (q < -128.0f) q = -128.0f;
        if (q > 127.0f) q = 127.0f;
        out[i] = (int8_t)q;
    }
}
