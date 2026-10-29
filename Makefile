# Host build: TFLM (+ CMSIS-NN C fallbacks) as a static lib, the shared core
# code, and two test programs. Firmware is built with PlatformIO, see README.

CC       ?= gcc
CXX      ?= g++
TFLM     := third_party/tflm
BUILD    := build
PY       ?= python3

# -isystem so TFLM's own headers don't trip -Werror in our code
TFLM_INC := -isystem $(TFLM) -isystem $(TFLM)/third_party/flatbuffers/include \
            -isystem $(TFLM)/third_party/cmsis_nn -isystem $(TFLM)/third_party/cmsis_nn/Include
DEFS     := -DTF_LITE_STATIC_MEMORY -DCMSIS_NN
WARN     := -Wall -Wextra
OPT      ?= -O2 -g
CFLAGS_C   = -std=c99 $(WARN) -Werror $(OPT) -Icore
CFLAGS_CXX = -std=c++17 $(WARN) $(OPT) $(DEFS) -Icore $(TFLM_INC) -fno-exceptions -fno-rtti
SAN      ?=

TFLM_SRCS := $(shell find $(TFLM)/tensorflow -name '*.cc' -not -path '*cortex_m_generic*' 2>/dev/null) \
             $(shell find $(TFLM)/third_party/cmsis_nn/Source -name '*.c' 2>/dev/null)
TFLM_OBJS := $(patsubst $(TFLM)/%,$(BUILD)/tflm/%.o,$(TFLM_SRCS))

CORE_OBJS := $(BUILD)/core/vib_features.o $(BUILD)/core/anomaly_model.o $(BUILD)/core/model_data.o \
             $(BUILD)/host/tflm_host_port.o

.PHONY: all tflm test pytest train robustness firmware clean distclean

all: $(BUILD)/tvec_check $(BUILD)/test_features

tflm:
	./tools/fetch_tflm.sh

$(TFLM)/.sha:
	./tools/fetch_tflm.sh

$(BUILD)/tflm/%.cc.o: $(TFLM)/%.cc | $(TFLM)/.sha
	@mkdir -p $(dir $@)
	@$(CXX) $(CFLAGS_CXX) -w -c $< -o $@

$(BUILD)/tflm/%.c.o: $(TFLM)/%.c | $(TFLM)/.sha
	@mkdir -p $(dir $@)
	@$(CC) -std=c99 $(OPT) $(DEFS) $(TFLM_INC) -w -c $< -o $@

$(BUILD)/libtflm.a: $(TFLM_OBJS)
	@rm -f $@
	ar rcs $@ $^

$(BUILD)/core/vib_features.o: core/vib_features.c core/vib_features.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_C) $(SAN) -c $< -o $@

$(BUILD)/core/%.o: core/%.cc core/*.h | $(TFLM)/.sha
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS_CXX) -Werror $(SAN) -c $< -o $@

$(BUILD)/host/%.o: host/%.cc | $(TFLM)/.sha
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS_CXX) -Werror $(SAN) -c $< -o $@

$(BUILD)/tvec_check: $(BUILD)/host/tvec_check.o $(CORE_OBJS) $(BUILD)/libtflm.a
	$(CXX) $(SAN) $^ -o $@ -lm

$(BUILD)/test_features: host/test_features.c core/vib_features.c core/vib_features.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_C) -D_DEFAULT_SOURCE $(SAN) host/test_features.c core/vib_features.c -o $@ -lm

test: all
	./$(BUILD)/test_features
	./$(BUILD)/tvec_check host/vectors/test_vectors.bin

pytest: all
	$(PY) -m pytest -q tests

train:
	$(PY) -m train.train --cross-load

robustness:
	$(PY) -m train.robustness

firmware:
	cd firmware && pio run

clean:
	rm -rf $(BUILD)/core $(BUILD)/host $(BUILD)/tvec_check $(BUILD)/test_features

distclean:
	rm -rf $(BUILD) out firmware/.pio
