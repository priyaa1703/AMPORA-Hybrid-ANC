import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 3 - NLMS DEBUG / VALIDATION
# ============================================================

FS = 24000

LOW_CUTOFF = 80
CROSSOVER = 1500

NUM_TAPS = 128
MU = 0.1
EPSILON = 1e-8

PRIMARY_FILE = "results/primary_low_80_1500.npy"
REFERENCE_FILE = "results/reference_low_80_1500.npy"

CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_WAV = "results/nlms_clean_low_80_1500.wav"
OUTPUT_NPY = "results/nlms_clean_low_80_1500.npy"


# ============================================================
# LOAD STAGE 2 SIGNALS
# ============================================================

print()
print("Loading Stage 2 signals...")

primary = np.load(PRIMARY_FILE)
reference = np.load(REFERENCE_FILE)

common_length = min(len(primary), len(reference))

primary = primary[:common_length]
reference = reference[:common_length]

print("Primary samples  :", len(primary))
print("Reference samples:", len(reference))


# ============================================================
# CORRECT REFERENCE BAND
# ============================================================

print()
print("Band-limiting reference...")
print("Reference band: 80 Hz - 1.5 kHz")

sos_reference = butter(
    4,
    [LOW_CUTOFF, CROSSOVER],
    btype="bandpass",
    fs=FS,
    output="sos"
)

reference = sosfiltfilt(
    sos_reference,
    reference
)


# ============================================================
# CORRELATION CHECK
# ============================================================

correlation = np.corrcoef(
    primary,
    reference
)[0, 1]

print()
print("Primary/reference correlation:")
print(f"{correlation:.4f}")


# ============================================================
# NLMS FUNCTION
# ============================================================

def nlms_noise_canceller(d, x, num_taps, mu, epsilon):

    weights = np.zeros(num_taps)

    x_buffer = np.zeros(num_taps)

    error = np.zeros(len(d))

    estimated_noise = np.zeros(len(d))

    weight_history = []

    for n in range(len(d)):

        # Shift reference samples
        x_buffer[1:] = x_buffer[:-1]

        x_buffer[0] = x[n]

        # Estimated noise
        y = np.dot(weights, x_buffer)

        # Cleaned signal
        e = d[n] - y

        estimated_noise[n] = y
        error[n] = e

        # NLMS normalization
        power = np.dot(
            x_buffer,
            x_buffer
        )

        weights += (
            mu
            * e
            * x_buffer
            / (epsilon + power)
        )

        # Store coefficient magnitude occasionally
        if n % 1000 == 0:
            weight_history.append(
                np.linalg.norm(weights)
            )

    return error, estimated_noise, weights, weight_history


# ============================================================
# RUN NLMS
# ============================================================

print()
print("Running NLMS...")

print()
print("NLMS parameters:")
print("Sampling rate :", FS, "Hz")
print("Number of taps:", NUM_TAPS)
print("Step size     :", MU)
print("Epsilon       :", EPSILON)


cleaned_low, estimated_noise, final_weights, weight_history = (
    nlms_noise_canceller(
        primary,
        reference,
        NUM_TAPS,
        MU,
        EPSILON
    )
)


# ============================================================
# SAVE OUTPUT
# ============================================================

np.save(
    OUTPUT_NPY,
    cleaned_low
)

max_value = np.max(
    np.abs(cleaned_low)
)

if max_value > 0:
    wav_signal = (
        0.95
        * cleaned_low
        / max_value
    )
else:
    wav_signal = cleaned_low

wav_signal_int16 = np.int16(
    np.clip(wav_signal, -1, 1)
    * 32767
)

wavfile.write(
    OUTPUT_WAV,
    FS,
    wav_signal_int16
)


# ============================================================
# CLEAN SPEECH REFERENCE
# ============================================================

print()
print("Preparing clean speech reference...")

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


if clean_fs != FS:

    clean_speech = resample_poly(
        clean_speech,
        FS,
        clean_fs
    )


clean_speech = clean_speech[
    :common_length
]


# Same 80 Hz - 1.5 kHz band
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
# SCALE-MATCH FOR SNR MEASUREMENT
# ============================================================

def calculate_snr(clean, test):

    # Find the gain that best matches test to clean
    gain = np.dot(test, clean) / (
        np.dot(clean, clean) + 1e-12
    )

    aligned_clean = gain * clean

    noise = test - aligned_clean

    signal_power = np.mean(
        aligned_clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    return 10 * np.log10(
        signal_power /
        (noise_power + 1e-12)
    )


input_snr = calculate_snr(
    clean_low,
    primary
)

output_snr = calculate_snr(
    clean_low,
    cleaned_low
)

snr_improvement = (
    output_snr - input_snr
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print("           STAGE 3 NLMS RESULT")
print("==============================================")

print()
print("Primary/reference correlation:")
print(f"{correlation:.4f}")

print()
print("NLMS:")
print("Taps       :", NUM_TAPS)
print("Step size  :", MU)

print()
print("SNR:")
print(f"Input SNR  : {input_snr:.2f} dB")
print(f"Output SNR : {output_snr:.2f} dB")
print(
    f"Improvement: {snr_improvement:.2f} dB"
)

print()
print("Final coefficient norm:")
print(
    f"{np.linalg.norm(final_weights):.6f}"
)

print()
print("Output:")
print(OUTPUT_WAV)
print(OUTPUT_NPY)


# ============================================================
# PLOTS
# ============================================================

time = np.arange(common_length) / FS

plot_samples = min(
    int(0.1 * FS),
    common_length
)


# ------------------------------------------------------------
# PRIMARY SIGNAL
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:plot_samples],
    primary[:plot_samples]
)

plt.title(
    "Primary Low-Band Signal"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


# ------------------------------------------------------------
# REFERENCE SIGNAL
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:plot_samples],
    reference[:plot_samples]
)

plt.title(
    "Reference Noise Signal"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


# ------------------------------------------------------------
# CLEANED SIGNAL
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(
    time[:plot_samples],
    cleaned_low[:plot_samples]
)

plt.title(
    "NLMS Cleaned Low-Band Signal"
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
    time[:plot_samples],
    estimated_noise[:plot_samples]
)

plt.title(
    "NLMS Estimated Noise"
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
    np.arange(len(weight_history)) * 1000,
    weight_history
)

plt.title(
    "NLMS Coefficient Norm During Adaptation"
)

plt.xlabel("Sample Number")
plt.ylabel("Coefficient Norm")

plt.grid()
plt.tight_layout()


plt.show()