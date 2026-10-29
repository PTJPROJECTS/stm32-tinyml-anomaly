// Host (x86/Linux) versions of the few functions TFLM expects the target
// to provide. On the STM32 these come from cortex_m_generic/debug_log.cc and
// firmware/src/tflm_port.cpp.
#include <stdio.h>
#include <time.h>

#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_time.h"

extern "C" void DebugLog(const char *format, va_list args)
{
    vfprintf(stderr, format, args);
}

extern "C" int DebugVsnprintf(char *buffer, size_t buf_size, const char *format,
                              va_list vlist)
{
    return vsnprintf(buffer, buf_size, format, vlist);
}

namespace tflite {
namespace micro {
uint32_t ticks_per_second() { return 1000000; }
uint32_t GetCurrentTimeTicks()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000000ull + ts.tv_nsec / 1000);
}
}  // namespace micro
}  // namespace tflite
