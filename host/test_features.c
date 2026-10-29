/* Small unit tests for core/features.c that don't need Python:
 * a pure tone has to land in the right band, a flat (DC) signal has no
 * energy, and quantize clamps. */
#include <math.h>
#include <stdio.h>

#include "vib_features.h"

static int fails;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            fails++;                                                \
        }                                                           \
    } while (0)

static uint16_t win[FEAT_WIN];
static float bands[FEAT_N_BANDS];

static void tone(double hz, double amp_g)
{
    for (int k = 0; k < FEAT_WIN; k++) {
        double g = amp_g * sin(2.0 * M_PI * hz * k / FEAT_FS_HZ);
        win[k] = (uint16_t)lround(FEAT_ADC_MID + g / FEAT_ADC_LSB_G);
    }
}

static int argmax(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b])
            b = i;
    return b;
}

int main(void)
{
    features_init();

    /* band width = 8 bins * 12000/512 Hz = 187.5 Hz, band b covers bins
     * 8b+1 .. 8b+8 */
    const double bin_hz = (double)FEAT_FS_HZ / FEAT_WIN;
    int test_bins[] = {4, 30, 100, 200, 250};
    for (unsigned i = 0; i < sizeof test_bins / sizeof test_bins[0]; i++) {
        int bin = test_bins[i];
        tone(bin * bin_hz, 1.0);
        features_extract(win, bands);
        int expect = (bin - 1) / FEAT_BINS_PER_BAND;
        CHECK(argmax(bands, FEAT_N_BANDS) == expect);
    }

    /* constant input: mean removal leaves nothing, every band at log10(1e-9) */
    for (int k = 0; k < FEAT_WIN; k++)
        win[k] = 3000;
    features_extract(win, bands);
    for (int b = 0; b < FEAT_N_BANDS; b++)
        CHECK(fabsf(bands[b] - (-9.0f)) < 1e-3f);

    /* doubling the amplitude adds log10(4) to the tone's band */
    tone(30 * bin_hz, 0.5);
    features_extract(win, bands);
    float a = bands[3];
    tone(30 * bin_hz, 1.0);
    features_extract(win, bands);
    CHECK(fabsf((bands[3] - a) - log10f(4.0f)) < 0.01f);

    /* quantize clamps to int8 */
    float mean[FEAT_N_BANDS] = {0}, inv[FEAT_N_BANDS];
    float in[FEAT_N_BANDS];
    int8_t q[FEAT_N_BANDS];
    for (int i = 0; i < FEAT_N_BANDS; i++) {
        inv[i] = 1.0f;
        in[i] = (i % 2) ? 1000.0f : -1000.0f;
    }
    features_quantize(in, mean, inv, 0.1f, 5, q);
    CHECK(q[0] == -128 && q[1] == 127);
    in[0] = 0.0f;
    features_quantize(in, mean, inv, 0.1f, 5, q);
    CHECK(q[0] == 5);

    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
