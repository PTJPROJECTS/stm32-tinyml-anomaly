// Target hooks TFLM needs on the STM32: a tick source (DWT cycle counter)
// and somewhere to print errors (Serial).
#include <Arduino.h>

#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
namespace micro {
uint32_t ticks_per_second() { return SystemCoreClock; }
uint32_t GetCurrentTimeTicks() { return DWT->CYCCNT; }
}  // namespace micro
}  // namespace tflite

static void log_to_serial(const char *s) { Serial.print(s); }

void tflm_port_init(void) { RegisterDebugLogCallback(log_to_serial); }
