// Bearing fault / anomaly detection on a NUCLEO-F446RE.
//
// TIM ISR at 12 kHz reads one ADC sample (accelerometer on PA0 / A0) into a
// ping-pong buffer. Every 512 samples it hands the full buffer to the
// inference task. The inference task runs features + TFLM and prints the
// result. A stats task prints sampling jitter and latency every 5 s.
//
//   ISR (TIM, 12 kHz)  --notify-->  infer task (prio 3)  --> UART
//                                    stats task (prio 1)  --> UART
#include <Arduino.h>
#include <STM32FreeRTOS.h>

#include "anomaly_model.h"
#include "model_data.h"
#include "selftest_vectors.h"
#include "vib_features.h"

void tflm_port_init(void);

#define ADC_PIN_CHANNEL 0   // PA0 = ADC1_IN0 = Arduino A0

// ---- sampling -------------------------------------------------------------

static uint16_t s_buf[2][FEAT_WIN];
static volatile uint8_t s_fill = 0;       // buffer the ISR is writing
static volatile uint16_t s_idx = 0;
static volatile uint8_t s_ready = 0xFF;   // buffer waiting for inference
static volatile uint32_t s_overruns = 0;
static TaskHandle_t s_infer_task;
static HardwareTimer *s_tim;

// jitter stats, in CPU cycles, between consecutive ISR entries
struct Jitter {
    uint32_t last, min, max, n;
    uint64_t sum, sumsq;
};
static volatile Jitter s_jit;
static uint32_t s_period_cycles;

static void cycle_counter_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

// Register level ADC1 setup: single channel, software start, 12 bit.
// analogRead() reconfigures the ADC on every call, far too slow for 12 kHz.
static void adc_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
    GPIOA->MODER |= (3u << (ADC_PIN_CHANNEL * 2));       // analog mode
    ADC->CCR = (ADC->CCR & ~ADC_CCR_ADCPRE) | ADC_CCR_ADCPRE_0;  // PCLK2/4 = 22.5 MHz
    ADC1->CR1 = 0;                                       // 12 bit, no scan
    ADC1->SMPR2 = (4u << (ADC_PIN_CHANNEL * 3));         // 84 cycles, ~4.3 us total
    ADC1->SQR1 = 0;                                      // 1 conversion
    ADC1->SQR3 = ADC_PIN_CHANNEL;
    ADC1->CR2 = ADC_CR2_ADON;
    delayMicroseconds(5);
    ADC1->CR2 |= ADC_CR2_SWSTART;                        // first conversion
}

static void sample_isr(void)
{
    uint32_t now = DWT->CYCCNT;
    // Read the conversion started on the previous tick, then start the next
    // one. Adds one sample of delay but the ISR never waits on the ADC.
    uint16_t v = (uint16_t)ADC1->DR;
    ADC1->CR2 |= ADC_CR2_SWSTART;

    if (s_jit.last != 0) {
        uint32_t d = now - s_jit.last;
        if (d < s_jit.min) s_jit.min = d;
        if (d > s_jit.max) s_jit.max = d;
        s_jit.sum += d;
        s_jit.sumsq += (uint64_t)d * d;
        s_jit.n++;
    }
    s_jit.last = now;

    s_buf[s_fill][s_idx++] = v;
    if (s_idx == FEAT_WIN) {
        s_idx = 0;
        if (s_ready != 0xFF)
            s_overruns++;            // inference still busy with the last one
        s_ready = s_fill;
        s_fill ^= 1;
        BaseType_t woken = pdFALSE;
        vTaskNotifyGiveFromISR(s_infer_task, &woken);
        portYIELD_FROM_ISR(woken);
    }
}

// ---- inference ------------------------------------------------------------

static volatile uint32_t s_feat_cyc_max, s_model_cyc_max, s_feat_cyc_last, s_model_cyc_last;
static volatile uint32_t s_windows, s_class_count[MODEL_N_CLASSES];

static void infer_task(void *)
{
    float bands[FEAT_N_BANDS];
    int8_t q[FEAT_N_BANDS], scores[MODEL_N_CLASSES];
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        uint8_t b = s_ready;
        if (b == 0xFF)
            continue;

        uint32_t t0 = DWT->CYCCNT;
        features_extract(s_buf[b], bands);
        features_quantize(bands, g_feat_mean, g_feat_inv_std, model_input_scale(),
                          model_input_zero_point(), q);
        uint32_t t1 = DWT->CYCCNT;
        int cls = model_invoke(q, scores);
        uint32_t t2 = DWT->CYCCNT;
        s_ready = 0xFF;

        s_feat_cyc_last = t1 - t0;
        s_model_cyc_last = t2 - t1;
        if (s_feat_cyc_last > s_feat_cyc_max) s_feat_cyc_max = s_feat_cyc_last;
        if (s_model_cyc_last > s_model_cyc_max) s_model_cyc_max = s_model_cyc_last;
        s_windows++;
        if (cls >= 0) {
            s_class_count[cls]++;
            // p(fault) = 1 - p(normal); score is int8 with scale 1/256, zp -128
            float p_normal = (scores[0] + 128) / 256.0f;
            Serial.printf("win %lu %-10s p_fault=%.2f\n", (unsigned long)s_windows,
                          g_class_names[cls], 1.0f - p_normal);
        }
    }
}

static float cyc_to_us(uint32_t c) { return c * 1e6f / SystemCoreClock; }

static void stats_task(void *)
{
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        taskENTER_CRITICAL();
        Jitter j;
        j.min = s_jit.min; j.max = s_jit.max; j.n = s_jit.n;
        j.sum = s_jit.sum; j.sumsq = s_jit.sumsq;
        s_jit.min = UINT32_MAX; s_jit.max = 0; s_jit.n = 0; s_jit.sum = 0; s_jit.sumsq = 0;
        taskEXIT_CRITICAL();
        if (j.n == 0)
            continue;
        double mean = (double)j.sum / j.n;
        double var = (double)j.sumsq / j.n - mean * mean;
        Serial.printf("[stats] period mean %.2f us (target %.2f), min %.2f, max %.2f, "
                      "std %.3f us, n=%lu\n",
                      mean * 1e6 / SystemCoreClock, cyc_to_us(s_period_cycles),
                      cyc_to_us(j.min), cyc_to_us(j.max),
                      sqrt(var > 0 ? var : 0) * 1e6 / SystemCoreClock, (unsigned long)j.n);
        Serial.printf("[stats] features %.1f us (max %.1f), model %.1f us (max %.1f), "
                      "overruns %lu, windows %lu\n",
                      cyc_to_us(s_feat_cyc_last), cyc_to_us(s_feat_cyc_max),
                      cyc_to_us(s_model_cyc_last), cyc_to_us(s_model_cyc_max),
                      (unsigned long)s_overruns, (unsigned long)s_windows);
    }
}

// ---- boot -----------------------------------------------------------------

// Same check as host/tvec_check, on the board: fixed windows in flash,
// compare against the TFLite interpreter output from Python.
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
        // a 1 LSB input difference from the float FFT is allowed, so only
        // the class has to match exactly
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
    cycle_counter_init();
    tflm_port_init();

    int rc = model_init();
    Serial.printf("\nstm32-tinyml-anomaly, core %lu Hz\n", (unsigned long)SystemCoreClock);
    Serial.printf("model_init=%d, model %u bytes, arena used %u of %u bytes\n", rc,
                  g_model_tflite_len, (unsigned)model_arena_used(),
                  (unsigned)model_arena_size());
    if (rc != 0)
        for (;;) {}
    selftest();

    s_jit.min = UINT32_MAX;
    s_period_cycles = SystemCoreClock / FEAT_FS_HZ;
    adc_init();

    xTaskCreate(infer_task, "infer", 1024, nullptr, 3, &s_infer_task);
    xTaskCreate(stats_task, "stats", 512, nullptr, 1, nullptr);

    s_tim = new HardwareTimer(TIM2);
    s_tim->setOverflow(FEAT_FS_HZ, HERTZ_FORMAT);
    // must be numerically >= configMAX_SYSCALL_INTERRUPT_PRIORITY to call
    // FreeRTOS FromISR functions
    s_tim->setInterruptPriority(6, 0);
    s_tim->attachInterrupt(sample_isr);
    s_tim->resume();

    vTaskStartScheduler();
}

void loop() {}
