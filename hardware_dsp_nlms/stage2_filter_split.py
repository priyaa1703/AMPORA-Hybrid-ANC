import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import os

# ============================================================
# STAGE 2
# 80 Hz - 8 kHz Bandpass
# + 1.5 kHz Frequency Split
# + FFT Frequency Verification
#
# Sampling Rate = 24 kHz
#
# LOW BAND  -> 80 Hz - 1.5 kHz -> NLMS
# HIGH BAND -> 1.5 kHz - 8 kHz -> AI
# ============================================================


# ============================================================
# FILES
# ============================================================

primary_file = "results/primary_mic.wav"
reference_file = "results/reference_mic.wav"

# WAV outputs
primary_bandpass_file = "results/primary_80_8000.wav"
primary_low_file = "results/primary_low_80_1500.wav"
primary_high_file = "results/primary_high_1500_8000.wav"
reference_low_file = "results/reference_low_80_1500.wav"

# Exact floating-point outputs for later processing
primary_low_npy = "results/primary_low_80_1500.npy"
primary_high_npy = "results/primary_high_1500_8000.npy"
reference_low_npy = "results/reference_low_80_1500.npy"


# ============================================================
# PARAMETERS
# ============================================================

FS = 24000

LOW_CUTOFF = 80
HIGH_CUTOFF = 8000
CROSSOVER = 1500

FILTER_ORDER = 4


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD PRIMARY MICROPHONE
# ============================================================

print("Loading primary microphone...")

fs_primary, primary = wavfile.read(primary_file)

primary = primary.astype(np.float64)

if primary.ndim > 1:
    primary = np.mean(primary, axis=1)

# Convert int16 range to floating point
primary = primary / 32768.0


# ============================================================
# LOAD REFERENCE MICROPHONE
# ============================================================

print("Loading reference microphone...")

fs_reference, reference = wavfile.read(reference_file)

reference = reference.astype(np.float64)

if reference.ndim > 1:
    reference = np.mean(reference, axis=1)

# Convert int16 range to floating point
reference = reference / 32768.0


# ============================================================
# CHECK SAMPLING RATES
# ============================================================

print()
print("Primary sampling rate  :", fs_primary, "Hz")
print("Reference sampling rate:", fs_reference, "Hz")

if fs_primary != FS:
    raise ValueError(
        f"Primary microphone is {fs_primary} Hz, "
        f"but expected {FS} Hz."
    )

if fs_reference != FS:
    raise ValueError(
        f"Reference microphone is {fs_reference} Hz, "
        f"but expected {FS} Hz."
    )


# ============================================================
# MAKE SIGNALS SAME LENGTH
# ============================================================

length = min(len(primary), len(reference))

primary = primary[:length]
reference = reference[:length]

print()
print("Signal length:", length)
print("Duration:", round(length / FS, 2), "seconds")


# ============================================================
# DESIGN 80 Hz - 8 kHz BANDPASS FILTER
# ============================================================

print()
print("Designing 80 Hz - 8 kHz Butterworth filter...")

sos_bandpass = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass",
    fs=FS,
    output="sos"
)


# ============================================================
# FILTER PRIMARY MICROPHONE
# ============================================================

print("Filtering primary microphone...")

primary_bandpass = sosfiltfilt(
    sos_bandpass,
    primary
)


# ============================================================
# DESIGN 1.5 kHz LOW-PASS FILTER
# ============================================================

sos_low = butter(
    FILTER_ORDER,
    CROSSOVER,
    btype="lowpass",
    fs=FS,
    output="sos"
)


# ============================================================
# DESIGN 1.5 kHz HIGH-PASS FILTER
# ============================================================

sos_high = butter(
    FILTER_ORDER,
    CROSSOVER,
    btype="highpass",
    fs=FS,
    output="sos"
)


# ============================================================
# CREATE LOW BAND
# ============================================================

print("Creating low-frequency band...")

primary_low = sosfiltfilt(
    sos_low,
    primary_bandpass
)


# ============================================================
# CREATE HIGH BAND
# ============================================================

print("Creating high-frequency band...")

primary_high = sosfiltfilt(
    sos_high,
    primary_bandpass
)


# ============================================================
# FILTER REFERENCE MICROPHONE
# ============================================================

print("Filtering reference microphone...")

reference_low = sosfiltfilt(
    sos_low,
    reference
)


# ============================================================
# IMPORTANT:
# COMMON SCALING FOR WAV FILES
#
# We do NOT independently normalize each signal.
#
# This preserves the amplitude relationship between:
#
# Primary low band
# Reference low band
#
# which is important for NLMS.
# ============================================================

max_primary_bandpass = np.max(np.abs(primary_bandpass))
max_primary_low = np.max(np.abs(primary_low))
max_primary_high = np.max(np.abs(primary_high))
max_reference_low = np.max(np.abs(reference_low))

common_max = max(
    max_primary_bandpass,
    max_primary_low,
    max_primary_high,
    max_reference_low
)

if common_max == 0:
    raise ValueError("All processed signals are zero.")


# Small safety margin so we don't reach int16 maximum
COMMON_SCALE = 0.95 / common_max

print()
print("Common WAV scaling factor:", COMMON_SCALE)


# ============================================================
# SAVE WAV FILES
# ============================================================

wavfile.write(
    primary_bandpass_file,
    FS,
    np.int16(
        np.clip(
            primary_bandpass * COMMON_SCALE,
            -1,
            1
        ) * 32767
    )
)

wavfile.write(
    primary_low_file,
    FS,
    np.int16(
        np.clip(
            primary_low * COMMON_SCALE,
            -1,
            1
        ) * 32767
    )
)

wavfile.write(
    primary_high_file,
    FS,
    np.int16(
        np.clip(
            primary_high * COMMON_SCALE,
            -1,
            1
        ) * 32767
    )
)

wavfile.write(
    reference_low_file,
    FS,
    np.int16(
        np.clip(
            reference_low * COMMON_SCALE,
            -1,
            1
        ) * 32767
    )
)


# ============================================================
# SAVE FLOATING-POINT DATA
#
# These files will be used later by NLMS/AI.
# No WAV quantization is involved.
# ============================================================

np.save(primary_low_npy, primary_low)
np.save(primary_high_npy, primary_high)
np.save(reference_low_npy, reference_low)


# ============================================================
# VERIFY WAV FILES
# ============================================================

fs_check1, _ = wavfile.read(primary_bandpass_file)
fs_check2, _ = wavfile.read(primary_low_file)
fs_check3, _ = wavfile.read(primary_high_file)
fs_check4, _ = wavfile.read(reference_low_file)


# ============================================================
# STAGE 2 SUMMARY
# ============================================================

print()
print("============================================================")
print("                 STAGE 2 COMPLETE")
print("============================================================")

print()
print("Sampling rate:", FS, "Hz")
print("Nyquist frequency:", FS / 2, "Hz")

print()
print("Frequency bands:")
print("----------------------------")
print("Overall band     : 80 Hz - 8 kHz")
print("Low band         : 80 Hz - 1.5 kHz")
print("High band        : 1.5 kHz - 8 kHz")
print("Crossover        :", CROSSOVER, "Hz")

print()
print("WAV output files:")
print("----------------------------")
print(primary_bandpass_file)
print(primary_low_file)
print(primary_high_file)
print(reference_low_file)

print()
print("Floating-point files:")
print("----------------------------")
print(primary_low_npy)
print(primary_high_npy)
print(reference_low_npy)

print()
print("WAV verification:")
print("----------------------------")
print("80-8k primary    :", fs_check1, "Hz")
print("Low primary      :", fs_check2, "Hz")
print("High primary     :", fs_check3, "Hz")
print("Low reference    :", fs_check4, "Hz")


# ============================================================
# TIME-DOMAIN PLOTS
# ============================================================

time = np.arange(length) / FS

display_time = 0.05
display_samples = int(display_time * FS)


# ------------------------------------------------------------
# Plot 1: Primary microphone
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time[:display_samples],
    primary[:display_samples]
)

plt.title("Primary Microphone - Speech + Noise")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 2: 80 Hz - 8 kHz
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time[:display_samples],
    primary_bandpass[:display_samples]
)

plt.title("Primary Signal - 80 Hz to 8 kHz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 3: Low band
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time[:display_samples],
    primary_low[:display_samples]
)

plt.title("Low Band - 80 Hz to 1.5 kHz → NLMS")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot 4: High band
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time[:display_samples],
    primary_high[:display_samples]
)

plt.title("High Band - 1.5 kHz to 8 kHz → AI")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# FFT FUNCTION
# ============================================================

def plot_fft(signal, title, x_max, cutoff_lines):

    N = len(signal)

    # FFT
    fft_result = np.fft.rfft(signal)

    # Frequency axis
    frequencies = np.fft.rfftfreq(
        N,
        d=1 / FS
    )

    # Magnitude
    magnitude = np.abs(fft_result)

    # Normalize magnitude only for easier visual comparison
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)

    # Convert to dB
    magnitude_db = 20 * np.log10(
        magnitude + 1e-12
    )

    # Plot
    plt.figure(figsize=(12, 5))

    plt.plot(
        frequencies,
        magnitude_db
    )

    # Frequency boundaries
    for cutoff in cutoff_lines:
        plt.axvline(
            cutoff,
            linestyle="--",
            linewidth=1.5
        )

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, x_max)
    plt.ylim(-100, 5)
    plt.grid()

    plt.tight_layout()
    plt.show()


# ============================================================
# FFT PLOTS
# ============================================================

print()
print("Generating frequency-domain FFT plots...")


# ------------------------------------------------------------
# FFT 1: Primary 80 Hz - 8 kHz
# ------------------------------------------------------------

plot_fft(
    primary_bandpass,
    "FFT - Primary Signal (80 Hz to 8 kHz)",
    12000,
    [80, 8000]
)


# ------------------------------------------------------------
# FFT 2: Low band 80 Hz - 1.5 kHz
# ------------------------------------------------------------

plot_fft(
    primary_low,
    "FFT - Low Band (80 Hz to 1.5 kHz → NLMS)",
    4000,
    [80, 1500]
)


# ------------------------------------------------------------
# FFT 3: High band 1.5 kHz - 8 kHz
# ------------------------------------------------------------

plot_fft(
    primary_high,
    "FFT - High Band (1.5 kHz to 8 kHz → AI)",
    10000,
    [1500, 8000]
)


# ------------------------------------------------------------
# FFT 4: Reference 80 Hz - 1.5 kHz
# ------------------------------------------------------------

plot_fft(
    reference_low,
    "FFT - Reference Low Band (80 Hz to 1.5 kHz → NLMS)",
    4000,
    [80, 1500]
)


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================================")
print("          STAGE 2 + FFT VERIFICATION COMPLETE")
print("============================================================")

print()
print("Time-domain filtering: DONE")
print("Frequency-domain verification: DONE")
print("Common signal scaling: DONE")
print("Floating-point files for NLMS: DONE")

print()
print("Next stage:")
print("NLMS processing of the 80 Hz - 1.5 kHz band.")