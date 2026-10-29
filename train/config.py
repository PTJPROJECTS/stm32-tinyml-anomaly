"""Settings shared by the training scripts. The C side has the same numbers in
core/features.h, and tests/test_config_sync.py checks they still match."""

FS_HZ = 12000          # sample rate the MCU runs the ADC at
WIN = 512              # samples per window (42.7 ms)
HOP = 256              # 50 % overlap for training windows
N_BANDS = 32           # log band energies fed to the model
BINS_PER_BAND = 8      # 32 * 8 = 256 FFT bins = 0 .. 6 kHz

# Simulated ADC front end. Accelerometer in g -> 12 bit ADC counts.
# +/-8 g full scale over 0..4095, mid-scale 2048.
ADC_BITS = 12
ADC_MID = 2048
ADC_FULL_SCALE_G = 8.0
ADC_LSB_G = ADC_FULL_SCALE_G / ADC_MID   # 3.906 mg per count

CLASSES = ["normal", "inner_race", "ball", "outer_race"]
