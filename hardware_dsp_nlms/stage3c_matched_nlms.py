import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 3C - MATCHED TWO-MICROPHONE NLMS SIMULATION
# ============================================================

FS = 24000

LOW_CUTOFF = 80
CROSSOVER = 1500

NUM_TAPS = 128
MU = 0.1
EPSILON = 1e-8

REFERENCE_FILE = "results/reference_low_80_1500.npy"
CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_FILE = "results/nlms_matched_clean_low.wav"


# ============================================================
# LOAD REFERENCE NOISE
# ============================================================

print()
print("Loading reference noise...")

noise = np.load(
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


# ============================================================
# RESAMPLE SPEECH
# ============================================================

if clean_fs != FS:

    clean_speech = resample_poly(
        clean_speech,
        FS,
        clean_fs
    )


# ============================================================
# BANDPASS CLEAN SPEECH
# ============================================================

sos = butter(
    4,
    [LOW_CUTOFF, CROSSOVER],
    btype="bandpass",
    fs=FS,
    output="sos"
)

clean_low = sosfiltfilt(
    sos,
    clean_speech
)


# ============================================================
# MATCH LENGTH
# ============================================================

length = min(
    len(clean_low),
    len(noise)
)

clean_low = clean_low[:length]
noise = noise[:length]


# ============================================================
# CREATE TWO MICROPHONE PATHS
# ============================================================

# Reference microphone:
# captures noise with a particular gain and delay.

REFERENCE_DELAY = 5
REFERENCE_GAIN = 0.8


# Primary microphone:
# captures the same noise through a slightly
# different acoustic path.

PRIMARY_DELAY = 20
PRIMARY_GAIN = 0.65


reference = np.zeros_like(noise)

reference[REFERENCE_DELAY:] = (
    REFERENCE_GAIN
    * noise[:-REFERENCE_DELAY]
)


primary_noise = np.zeros_like(noise)

primary_noise[PRIMARY_DELAY:] = (
    PRIMARY_GAIN
    * noise[:-PRIMARY_DELAY]
)


# ============================================================
# SCALE SPEECH TO GIVE APPROXIMATELY 5 dB INPUT SNR
# ============================================================

noise_power = np.mean(
    primary_noise ** 2
)

speech_power = np.mean(
    clean_low ** 2
)


TARGET_SNR_DB = 5

target_ratio = (
    10 ** (TARGET_SNR_DB / 10)
)


desired_speech_power = (
    noise_power
    * target_ratio
)


speech_scale = np.sqrt(
    desired_speech_power
    / (speech_power + 1e-12)
)


clean_low = (
    clean_low
    * speech_scale
)


# ============================================================
# PRIMARY MICROPHONE SIGNAL
# ============================================================

primary = (
    clean_low
    + primary_noise
)


# ============================================================
# CORRELATION
# ============================================================

correlation = np.corrcoef(
    primary_noise,
    reference
)[0, 1]


print()
print("Microphone simulation:")
print("Reference delay :", REFERENCE_DELAY)
print("Reference gain  :", REFERENCE_GAIN)
print("Primary delay   :", PRIMARY_DELAY)
print("Primary gain    :", PRIMARY_GAIN)

print()
print("Primary noise / reference correlation:")
print(f"{correlation:.4f}")


# ============================================================
# NLMS
# ============================================================

def nlms(d, x, taps, mu, epsilon):

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
print("Running NLMS...")

cleaned, estimated_noise, weights, weight_history = nlms(
    primary,
    reference,
    NUM_TAPS,
    MU,
    EPSILON
)


# ============================================================
# SNR
# ============================================================

def calculate_snr(clean, test):

    error = test - clean

    signal_power = np.mean(
        clean ** 2
    )

    error_power = np.mean(
        error ** 2
    )

    return 10 * np.log10(
        signal_power
        / (error_power + 1e-12)
    )


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
print("       STAGE 3C MATCHED NLMS RESULT")
print("==============================================")

print()
print("Input SNR       :", f"{input_snr:.2f} dB")
print("Output SNR      :", f"{output_snr:.2f} dB")
print("SNR improvement :", f"{improvement:.2f} dB")

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


plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    primary[:samples]
)

plt.title(
    "Primary Microphone - Speech + Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    reference[:samples]
)

plt.title(
    "Reference Microphone - Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


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


plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    cleaned[:samples]
)

plt.title(
    "NLMS Cleaned Low-Band Speech"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


plt.figure(figsize=(12, 4))

plt.plot(
    np.arange(len(weight_history)) * 1000,
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