import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 3A - CONTROLLED NLMS VALIDATION
# ============================================================

FS = 24000

LOW_CUTOFF = 80
CROSSOVER = 1500

NUM_TAPS = 128
MU = 0.1
EPSILON = 1e-8

REFERENCE_FILE = "results/reference_low_80_1500.npy"
CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_FILE = "results/nlms_test_output.wav"


# ============================================================
# LOAD REFERENCE NOISE
# ============================================================

print()
print("Loading reference noise...")

reference = np.load(
    REFERENCE_FILE
)


# ============================================================
# LOAD CLEAN SPEECH
# ============================================================

print("Loading clean speech...")

clean_fs, clean_speech = wavfile.read(
    CLEAN_SPEECH_FILE
)

clean_speech = clean_speech.astype(
    np.float64
)

clean_speech /= (
    np.max(np.abs(clean_speech))
    + 1e-12
)


# Resample speech to 24 kHz

if clean_fs != FS:

    clean_speech = resample_poly(
        clean_speech,
        FS,
        clean_fs
    )


# ============================================================
# BAND-LIMIT CLEAN SPEECH
# ============================================================

sos_clean = butter(
    4,
    [LOW_CUTOFF, CROSSOVER],
    btype="bandpass",
    fs=FS,
    output="sos"
)

clean_low = sosfiltfilt(
    sos_clean,
    clean_speech
)


# ============================================================
# MATCH LENGTH
# ============================================================

length = min(
    len(clean_low),
    len(reference)
)

clean_low = clean_low[:length]
reference = reference[:length]


# ============================================================
# CREATE REALISTIC PRIMARY NOISE PATH
# ============================================================

DELAY = 20
NOISE_GAIN = 0.8

delayed_noise = np.zeros_like(
    reference
)

delayed_noise[DELAY:] = (
    reference[:-DELAY]
)

delayed_noise *= NOISE_GAIN


# ============================================================
# CREATE PRIMARY SIGNAL
# ============================================================

primary = (
    clean_low
    + delayed_noise
)


# ============================================================
# CORRELATION
# ============================================================

# Compare reference with the actual noise
# that was inserted into the primary signal.

actual_noise = delayed_noise

correlation = np.corrcoef(
    actual_noise[DELAY:],
    reference[:-DELAY]
)[0, 1]


# ============================================================
# NLMS
# ============================================================

def nlms(
    d,
    x,
    taps,
    mu,
    epsilon
):

    weights = np.zeros(taps)

    buffer = np.zeros(taps)

    error = np.zeros(len(d))

    estimated_noise = np.zeros(len(d))

    weight_history = []

    for n in range(len(d)):

        buffer[1:] = buffer[:-1]

        buffer[0] = x[n]

        y = np.dot(
            weights,
            buffer
        )

        e = d[n] - y

        estimated_noise[n] = y

        error[n] = e

        power = np.dot(
            buffer,
            buffer
        )

        weights += (
            mu
            * e
            * buffer
            / (epsilon + power)
        )

        if n % 1000 == 0:

            weight_history.append(
                np.linalg.norm(weights)
            )

    return (
        error,
        estimated_noise,
        weights,
        weight_history
    )


# ============================================================
# RUN NLMS
# ============================================================

print()
print("Running controlled NLMS test...")

cleaned, estimated_noise, weights, weight_history = nlms(
    primary,
    reference,
    NUM_TAPS,
    MU,
    EPSILON
)


# ============================================================
# SNR FUNCTION
# ============================================================

def calculate_snr(
    clean,
    test
):

    error = test - clean

    signal_power = np.mean(
        clean ** 2
    )

    error_power = np.mean(
        error ** 2
    )

    return 10 * np.log10(
        signal_power /
        (error_power + 1e-12)
    )


# ============================================================
# SNR
# ============================================================

input_snr = calculate_snr(
    clean_low,
    primary
)

output_snr = calculate_snr(
    clean_low,
    cleaned
)

improvement = (
    output_snr - input_snr
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print("       STAGE 3A CONTROLLED NLMS TEST")
print("==============================================")

print()
print("Noise delay      :", DELAY, "samples")
print("Noise gain       :", NOISE_GAIN)
print("Number of taps   :", NUM_TAPS)
print("Step size        :", MU)

print()
print("Reference/noise correlation:")
print(f"{correlation:.4f}")

print()
print("SNR:")
print(f"Input SNR        : {input_snr:.2f} dB")
print(f"Output SNR       : {output_snr:.2f} dB")
print(f"Improvement      : {improvement:.2f} dB")

print()
print("Final coefficient norm:")
print(
    f"{np.linalg.norm(weights):.6f}"
)


# ============================================================
# SAVE OUTPUT
# ============================================================

max_value = np.max(
    np.abs(cleaned)
)

wav_output = (
    0.95
    * cleaned
    / (max_value + 1e-12)
)

wavfile.write(
    OUTPUT_FILE,
    FS,
    np.int16(
        np.clip(
            wav_output,
            -1,
            1
        )
        * 32767
    )
)

print()
print("Output:")
print(OUTPUT_FILE)


# ============================================================
# PLOTS
# ============================================================

time = (
    np.arange(length)
    / FS
)

samples = min(
    int(0.1 * FS),
    length
)


# ------------------------------------------------------------
# PRIMARY
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    primary[:samples]
)

plt.title(
    "Controlled Primary Signal - Speech + Delayed Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


# ------------------------------------------------------------
# ESTIMATED NOISE
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    estimated_noise[:samples]
)

plt.title(
    "NLMS Estimated Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


# ------------------------------------------------------------
# CLEANED OUTPUT
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    cleaned[:samples]
)

plt.title(
    "NLMS Cleaned Output"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


# ------------------------------------------------------------
# COEFFICIENT CONVERGENCE
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    np.arange(len(weight_history))
    * 1000,
    weight_history
)

plt.title(
    "NLMS Coefficient Convergence"
)

plt.xlabel("Sample Number")
plt.ylabel("Coefficient Norm")

plt.grid()
plt.tight_layout()


plt.show()