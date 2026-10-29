// Runs exported test vectors through the C pipeline and compares every
// stage with what Python produced:
//   1. C features (float FFT)  vs numpy features
//   2. C int8 input            vs Python int8 input
//   3. TFLM output on Python's int8 input vs TFLite interpreter output
//      (must be bit exact, same int8 math)
//   4. full C pipeline class   vs label
//
// usage: tvec_check <vectors.bin> [-q]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#include "anomaly_model.h"
#include "vib_features.h"
#include "model_data.h"

struct Vec {
    std::vector<uint16_t> win;
    float feat[FEAT_N_BANDS];
    int8_t in[FEAT_N_BANDS];
    int8_t out[MODEL_N_CLASSES];
    uint8_t label;
};

static bool load(const char *path, std::vector<Vec> &v)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return false;
    }
    char magic[4];
    uint32_t hdr[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TVEC", 4) != 0 ||
        fread(hdr, 4, 4, f) != 4) {
        fprintf(stderr, "%s: bad header\n", path);
        fclose(f);
        return false;
    }
    if (hdr[1] != FEAT_WIN || hdr[2] != FEAT_N_BANDS || hdr[3] != MODEL_N_CLASSES) {
        fprintf(stderr, "%s: shape mismatch win=%u feat=%u out=%u\n", path,
                hdr[1], hdr[2], hdr[3]);
        fclose(f);
        return false;
    }
    v.resize(hdr[0]);
    bool ok = true;
    for (auto &e : v) {
        e.win.resize(FEAT_WIN);
        ok = ok && fread(e.win.data(), 2, FEAT_WIN, f) == FEAT_WIN;
        ok = ok && fread(e.feat, 4, FEAT_N_BANDS, f) == FEAT_N_BANDS;
        ok = ok && fread(e.in, 1, FEAT_N_BANDS, f) == FEAT_N_BANDS;
        ok = ok && fread(e.out, 1, MODEL_N_CLASSES, f) == MODEL_N_CLASSES;
        ok = ok && fread(&e.label, 1, 1, f) == 1;
    }
    fclose(f);
    if (!ok)
        fprintf(stderr, "%s: short read\n", path);
    return ok;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s vectors.bin [-q]\n", argv[0]);
        return 2;
    }
    bool quiet = argc > 2 && strcmp(argv[2], "-q") == 0;
    std::vector<Vec> v;
    if (!load(argv[1], v))
        return 2;

    int rc = model_init();
    if (rc != 0) {
        fprintf(stderr, "model_init failed: %d\n", rc);
        return 1;
    }

    double feat_max_err = 0;
    int in_off_by_one = 0, in_worse = 0;
    int out_mismatch = 0, cls_vs_python = 0, correct = 0;
    double t_feat = 0, t_model = 0;
    int cm[MODEL_N_CLASSES][MODEL_N_CLASSES] = {{0}};

    for (size_t i = 0; i < v.size(); i++) {
        Vec &e = v[i];
        float feat[FEAT_N_BANDS];
        int8_t q[FEAT_N_BANDS], out[MODEL_N_CLASSES], out_py_in[MODEL_N_CLASSES];

        double t0 = now_us();
        features_extract(e.win.data(), feat);
        features_quantize(feat, g_feat_mean, g_feat_inv_std, model_input_scale(),
                          model_input_zero_point(), q);
        double t1 = now_us();
        int cls = model_invoke(q, out);
        t_model += now_us() - t1;
        t_feat += t1 - t0;

        for (int k = 0; k < FEAT_N_BANDS; k++) {
            double d = fabs((double)feat[k] - e.feat[k]);
            if (d > feat_max_err)
                feat_max_err = d;
            int dq = abs(q[k] - e.in[k]);
            if (dq == 1)
                in_off_by_one++;
            else if (dq > 1)
                in_worse++;
        }

        // model alone, fed exactly what Python fed the interpreter
        model_invoke(e.in, out_py_in);
        if (memcmp(out_py_in, e.out, MODEL_N_CLASSES) != 0) {
            out_mismatch++;
            if (!quiet)
                printf("vec %zu: tflm out differs from tflite\n", i);
        }

        int py_cls = 0;
        for (int k = 1; k < MODEL_N_CLASSES; k++)
            if (e.out[k] > e.out[py_cls])
                py_cls = k;
        if (cls != py_cls)
            cls_vs_python++;
        if (cls == e.label)
            correct++;
        cm[e.label][cls]++;
    }

    size_t n = v.size();
    printf("vectors            : %zu\n", n);
    printf("arena used         : %zu of %zu bytes\n", model_arena_used(), model_arena_size());
    printf("feature max |err|  : %.3g (log10 units)\n", feat_max_err);
    printf("int8 input off by 1: %d of %zu values, worse: %d\n", in_off_by_one,
           n * FEAT_N_BANDS, in_worse);
    printf("tflm vs tflite out : %d mismatching vectors\n", out_mismatch);
    printf("class vs python    : %d differ\n", cls_vs_python);
    printf("accuracy (C path)  : %.4f (%d/%zu)\n", (double)correct / n, correct, n);
    printf("host time / window : features %.1f us, model %.1f us\n", t_feat / n, t_model / n);
    printf("confusion (rows=true):\n");
    for (int r = 0; r < MODEL_N_CLASSES; r++) {
        printf("  %-11s", g_class_names[r]);
        for (int c = 0; c < MODEL_N_CLASSES; c++)
            printf(" %5d", cm[r][c]);
        printf("\n");
    }

    bool pass = feat_max_err < 1e-3 && in_worse == 0 && out_mismatch == 0 &&
                cls_vs_python == 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
