#include "anomaly_model.h"

#include <string.h>

#include "vib_features.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {

// arena_used_bytes() is 1904 on the 64 bit host build. On 32 bit ARM it
// should be a little less (smaller pointers), the firmware prints the real
// number at boot. 3 KB leaves some headroom.
#ifndef MODEL_ARENA_SIZE
#define MODEL_ARENA_SIZE 3072
#endif
alignas(16) uint8_t g_arena[MODEL_ARENA_SIZE];

tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_in = nullptr;
TfLiteTensor *g_out = nullptr;

}  // namespace

extern "C" int model_init(void)
{
    if (g_interp != nullptr)
        return 0;
    features_init();

    const tflite::Model *model = tflite::GetModel(g_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
        return -1;

    // Only the two ops the model has. The full AllOpsResolver would pull
    // every kernel into flash.
    static tflite::MicroMutableOpResolver<2> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk)
        return -2;

    static tflite::MicroInterpreter interp(model, resolver, g_arena, sizeof(g_arena));
    if (interp.AllocateTensors() != kTfLiteOk)
        return -3;

    g_in = interp.input(0);
    g_out = interp.output(0);
    if (g_in->type != kTfLiteInt8 || g_out->type != kTfLiteInt8 ||
        g_in->bytes != MODEL_N_FEATURES || g_out->bytes != MODEL_N_CLASSES)
        return -4;

    g_interp = &interp;
    return 0;
}

extern "C" size_t model_arena_used(void)
{
    return g_interp ? g_interp->arena_used_bytes() : 0;
}

extern "C" size_t model_arena_size(void) { return sizeof(g_arena); }

extern "C" float model_input_scale(void) { return g_in->params.scale; }

extern "C" int model_input_zero_point(void) { return g_in->params.zero_point; }

extern "C" int model_invoke(const int8_t *in, int8_t *out)
{
    if (g_interp == nullptr)
        return -1;
    memcpy(g_in->data.int8, in, MODEL_N_FEATURES);
    if (g_interp->Invoke() != kTfLiteOk)
        return -1;
    int best = 0;
    for (int i = 0; i < MODEL_N_CLASSES; i++) {
        out[i] = g_out->data.int8[i];
        if (out[i] > out[best])
            best = i;
    }
    return best;
}

extern "C" int model_classify_window(const uint16_t *win, int8_t *scores)
{
    float bands[FEAT_N_BANDS];
    int8_t q[FEAT_N_BANDS];
    features_extract(win, bands);
    features_quantize(bands, g_feat_mean, g_feat_inv_std, model_input_scale(),
                      model_input_zero_point(), q);
    return model_invoke(q, scores);
}
