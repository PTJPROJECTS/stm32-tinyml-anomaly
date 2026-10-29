// Bearing fault / anomaly detection on a NUCLEO-F446RE.
// First step: just get TFLM running on the board and check it against the
// Python output with a few fixed windows from flash.
#include <Arduino.h>

#include "anomaly_model.h"
#include "model_data.h"
#include "selftest_vectors.h"
#include "vib_features.h"

void tflm_port_init(void);

static float cyc_to_us(uint32_t c) { return c * 1e6f / SystemCoreClock; }

static void selftest(void)
{
    int bad = 0;
    int8_t scores[MODEL_N_CLASSES];
    uint32_t worst = 0;
    for (int i = 0; i < SELFTEST_N; i++) {
        uint32_t t0 = DWT->CYCCNT;
        int cls = model_classify_window(selftest_win[i], scores);
        uint32_t dt = DWT->CYCCNT - t0;
        if (dt > worst) worst = dt;
        bool same = memcmp(scores, selftest_out[i], MODEL_N_CLASSES) == 0;
        if (cls != selftest_label[i])
            bad++;
        Serial.printf("selftest %d: class %d (label %d) scores %s\n", i, cls,
                      selftest_label[i], same ? "match" : "differ");
    }
    Serial.printf("selftest: %d/%d wrong, worst window %lu cycles = %.1f us\n", bad,
                  SELFTEST_N, (unsigned long)worst, cyc_to_us(worst));
}

void setup()
{
    Serial.begin(115200);
    delay(200);
    tflm_port_init();
    int rc = model_init();
    Serial.printf("model_init=%d, arena used %u of %u bytes\n", rc,
                  (unsigned)model_arena_used(), (unsigned)model_arena_size());
    if (rc == 0)
        selftest();
}

void loop() { delay(1000); }
