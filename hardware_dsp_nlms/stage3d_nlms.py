import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 3D
# Proper controlled validation of NLMS
# ============================================================

# -----------------------------
# Configuration
# -----------------------------

FS = 24000

LOW_CUTOFF = 80
CROSSOVER = 1500
HIGH_CUTOFF = 8000

FILTER_ORDER = 4

NUM_TAPS = 128
MU = 0.1
EPSILON = 1e-8

TARGET_SNR_DB = 5.0

REFERENCE_DELAY = 5
REFERENCE_GAIN = 0.8

PRIMARY_DELAY = 20
PRIMARY_GAIN = 0.65

REFERENCE_FILE = "results/reference_low_80_1500.npy"
CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_FILE = "results/nlms_stage3d_clean_low.wav"


# ============================================================
# Helper functions
# ============================================================

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=4):
    """
    Offline zero-phase Butterworth bandpass filter.
    """

    sos = butter(
        order,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def lowpass_filter(signal, cutoff, fs, order=4):
    """
    Offline zero-phase Butterworth low-pass filter.
    """

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def calculate_power(signal):
    """
    Mean-square signal power.
    """

    return np.mean(signal ** 2)


def calculate_snr(clean, test):
    """
    SNR = power of clean signal / power of error.
    """

    error = test - clean

    clean_power = calculate_power(clean)
    error_power = calculate_power(error)

    snr = 10 * np.log10(
        (clean_power + 1e-12) /
        (error_power + 1e-12)
    )

    return snr


def calculate_noise_reduction(input_noise, output_noise):
    """
    Noise reduction based on noise power.
    """

    input_power = calculate_power(input_noise)
    output_power = calculate_power(output_noise)

    reduction = 10 * np.log10(
        (input_power + 1e-12) /
        (output_power + 1e-12)
    )

    return reduction


# ============================================================
# Load reference noise
# ============================================================

print("Loading reference noise...")

noise = np.load(REFERENCE_FILE)

noise = noise.astype(np.float64)

print("Reference noise samples:", len(noise))


# ============================================================
# Load clean speech
# ============================================================

print("\nLoading clean speech...")

clean_speech, speech_fs = sf.read(CLEAN_SPEECH_FILE)

# Convert stereo to mono if necessary
if clean_speech.ndim > 1:
    clean_speech = np.mean(clean_speech, axis=1)

clean_speech = clean_speech.astype(np.float64)


print("Original speech sample rate:", speech_fs)


# ============================================================
# Resample speech to 24 kHz if required
# ============================================================

if speech_fs != FS:

    print("Resampling clean speech to 24 kHz...")

    clean_speech = resample_poly(
        clean_speech,
        FS,
        speech_fs
    )


print("Speech sample rate after resampling:", FS)


# ============================================================
# Match signal lengths
# ============================================================

common_length = min(
    len(noise),
    len(clean_speech)
)

noise = noise[:common_length]
clean_speech = clean_speech[:common_length]


print("Common signal length:", common_length)

print(
    "Duration:",
    round(common_length / FS, 2),
    "seconds"
)


# ============================================================
# Create CLEAN ground truth
#
# IMPORTANT:
# Use EXACTLY the same primary low-band filter path
# that the microphone signal will use.
#
# clean speech
#      ↓
# 80 Hz - 8 kHz bandpass
#      ↓
# 1.5 kHz lowpass
#      ↓
# CLEAN LOW GROUND TRUTH
# ============================================================

print("\nCreating clean low-band ground truth...")

clean_band = bandpass_filter(
    clean_speech,
    LOW_CUTOFF,
    HIGH_CUTOFF,
    FS,
    FILTER_ORDER
)

clean_low = lowpass_filter(
    clean_band,
    CROSSOVER,
    FS,
    FILTER_ORDER
)


# ============================================================
# Prepare underlying noise
#
# The noise is already 80 Hz - 1.5 kHz from Stage 2.
# ============================================================

noise = noise / (
    np.max(np.abs(noise)) + 1e-12
)


# ============================================================
# Create reference microphone path
#
# Reference microphone:
#
# underlying noise
#       ↓
# delay 5 samples
#       ↓
# gain 0.8
#       ↓
# reference microphone
# ============================================================

reference = np.zeros_like(noise)

if REFERENCE_DELAY > 0:

    reference[REFERENCE_DELAY:] = (
        REFERENCE_GAIN *
        noise[:-REFERENCE_DELAY]
    )

else:

    reference = REFERENCE_GAIN * noise


# ============================================================
# Create primary microphone noise path
#
# Primary noise:
#
# underlying noise
#       ↓
# delay 20 samples
#       ↓
# gain 0.65
#       ↓
# primary microphone noise
# ============================================================

primary_noise = np.zeros_like(noise)

if PRIMARY_DELAY > 0:

    primary_noise[PRIMARY_DELAY:] = (
        PRIMARY_GAIN *
        noise[:-PRIMARY_DELAY]
    )

else:

    primary_noise = PRIMARY_GAIN * noise


# ============================================================
# Scale primary noise to achieve target SNR
# ============================================================

clean_power = calculate_power(clean_low)
noise_power = calculate_power(primary_noise)

target_noise_power = (
    clean_power /
    (10 ** (TARGET_SNR_DB / 10))
)

noise_scale = np.sqrt(
    target_noise_power /
    (noise_power + 1e-12)
)

primary_noise = primary_noise * noise_scale


# ============================================================
# Recalculate actual input SNR
# ============================================================

primary = clean_low + primary_noise

input_snr = calculate_snr(
    clean_low,
    primary
)


# ============================================================
# Display microphone simulation information
# ============================================================

print("\nMicrophone simulation:")

print(
    "Reference delay :",
    REFERENCE_DELAY,
    "samples"
)

print(
    "Reference gain  :",
    REFERENCE_GAIN
)

print(
    "Primary delay   :",
    PRIMARY_DELAY,
    "samples"
)

print(
    "Primary gain    :",
    PRIMARY_GAIN
)

print(
    "Relative delay  :",
    PRIMARY_DELAY - REFERENCE_DELAY,
    "samples"
)

print(
    "Relative delay  :",
    round(
        (PRIMARY_DELAY - REFERENCE_DELAY)
        / FS * 1000,
        3
    ),
    "ms"
)

print(
    "\nInput SNR before NLMS:",
    round(input_snr, 2),
    "dB"
)


# ============================================================
# Check maximum correlation over different delays
#
# This is more meaningful than zero-lag correlation because
# the two microphones intentionally have different delays.
# ============================================================

correlation_values = []

max_lag = 100

for lag in range(-max_lag, max_lag + 1):

    if lag < 0:

        a = primary_noise[-lag:]
        b = reference[:len(reference) + lag]

    elif lag > 0:

        a = primary_noise[:-lag]
        b = reference[lag:]

    else:

        a = primary_noise
        b = reference

    if len(a) > 10:

        correlation = np.corrcoef(a, b)[0, 1]

        correlation_values.append(
            (lag, correlation)
        )


best_lag, best_correlation = max(
    correlation_values,
    key=lambda item: abs(item[1])
)


print(
    "\nMaximum absolute noise correlation:"
)

print(
    "Best lag:",
    best_lag,
    "samples"
)

print(
    "Correlation:",
    round(best_correlation, 4)
)


# ============================================================
# NLMS
# ============================================================

print("\nRunning NLMS...")

x = reference
d = primary

N = len(d)

weights = np.zeros(NUM_TAPS)

output = np.zeros(N)

estimated_noise = np.zeros(N)

coefficient_norm = np.zeros(N)


for n in range(NUM_TAPS, N):

    # --------------------------------------------------------
    # Reference vector
    #
    # [x(n), x(n-1), ..., x(n-M+1)]
    # --------------------------------------------------------

    x_vector = x[
        n - NUM_TAPS + 1 :
        n + 1
    ][::-1]


    # --------------------------------------------------------
    # Estimate noise
    # --------------------------------------------------------

    y = np.dot(
        weights,
        x_vector
    )

    estimated_noise[n] = y


    # --------------------------------------------------------
    # Error / cleaned signal
    # --------------------------------------------------------

    e = d[n] - y

    output[n] = e


    # --------------------------------------------------------
    # NLMS normalization
    # --------------------------------------------------------

    input_energy = np.dot(
        x_vector,
        x_vector
    )

    normalization = (
        EPSILON +
        input_energy
    )


    # --------------------------------------------------------
    # Update adaptive coefficients
    # --------------------------------------------------------

    weights = (
        weights +
        MU *
        e *
        x_vector /
        normalization
    )


    # --------------------------------------------------------
    # Track coefficient norm
    # --------------------------------------------------------

    coefficient_norm[n] = np.linalg.norm(
        weights
    )


# ============================================================
# Remove initial adaptation region for evaluation
#
# The first part contains filter startup/transient behavior.
# ============================================================

EVALUATION_START = int(0.5 * FS)

if EVALUATION_START >= N:

    EVALUATION_START = NUM_TAPS


clean_eval = clean_low[EVALUATION_START:]
primary_eval = primary[EVALUATION_START:]
output_eval = output[EVALUATION_START:]

noise_eval = primary_noise[EVALUATION_START:]

estimated_noise_eval = (
    estimated_noise[EVALUATION_START:]
)


# ============================================================
# Calculate SNR
# ============================================================

input_snr_eval = calculate_snr(
    clean_eval,
    primary_eval
)

output_snr_eval = calculate_snr(
    clean_eval,
    output_eval
)

snr_improvement = (
    output_snr_eval -
    input_snr_eval
)


# ============================================================
# Calculate direct noise reduction
#
# Since we know the exact simulated primary noise,
# we can directly evaluate how much of it was removed.
#
# residual_noise =
# cleaned output - clean speech
# ============================================================

residual_noise = (
    output_eval -
    clean_eval
)

noise_reduction = calculate_noise_reduction(
    noise_eval,
    residual_noise
)


# ============================================================
# Final coefficient information
# ============================================================

final_coefficient_norm = np.linalg.norm(
    weights
)


largest_coefficient_index = np.argmax(
    np.abs(weights)
)

largest_coefficient_value = (
    weights[largest_coefficient_index]
)


# ============================================================
# Print results
# ============================================================

print("\n")
print("==============================================")
print("       STAGE 3D PROPER NLMS VALIDATION")
print("==============================================")

print(
    "\nEvaluation starts after:",
    round(EVALUATION_START / FS, 3),
    "seconds"
)

print(
    "\nInput SNR       :",
    round(input_snr_eval, 2),
    "dB"
)

print(
    "Output SNR      :",
    round(output_snr_eval, 2),
    "dB"
)

print(
    "SNR improvement :",
    round(snr_improvement, 2),
    "dB"
)

print(
    "\nDirect noise reduction:",
    round(noise_reduction, 2),
    "dB"
)

print(
    "\nFinal coefficient norm:"
)

print(
    round(final_coefficient_norm, 6)
)

print(
    "\nLargest coefficient:"
)

print(
    "Tap index :",
    largest_coefficient_index
)

print(
    "Value     :",
    round(largest_coefficient_value, 6)
)


# ============================================================
# Save cleaned output
# ============================================================

# Normalize only for WAV storage.
# The processing itself remains floating point.

max_output = np.max(
    np.abs(output)
)

if max_output > 0:

    output_wav = (
        0.95 *
        output /
        max_output
    )

else:

    output_wav = output


sf.write(
    OUTPUT_FILE,
    output_wav,
    FS
)


print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# Save floating-point output
# ============================================================

NPY_OUTPUT_FILE = (
    "results/nlms_stage3d_clean_low.npy"
)

np.save(
    NPY_OUTPUT_FILE,
    output
)


print(
    "Floating-point output:"
)

print(
    NPY_OUTPUT_FILE
)


# ============================================================
# Plot 1
# Primary microphone signal
# ============================================================

plot_duration = 0.05

plot_samples = min(
    int(plot_duration * FS),
    N
)

time = (
    np.arange(plot_samples) /
    FS
)

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    primary[:plot_samples]
)

plt.title(
    "Stage 3D - Primary Microphone"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.tight_layout()


# ============================================================
# Plot 2
# Clean speech vs primary
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    clean_low[:plot_samples],
    label="Clean speech"
)

plt.plot(
    time,
    primary[:plot_samples],
    label="Primary microphone",
    alpha=0.7
)

plt.title(
    "Stage 3D - Clean Speech vs Primary Microphone"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# Plot 3
# Estimated noise
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    primary_noise[:plot_samples],
    label="Actual primary noise"
)

plt.plot(
    time,
    estimated_noise[:plot_samples],
    label="NLMS estimated noise",
    alpha=0.7
)

plt.title(
    "Stage 3D - Actual Noise vs NLMS Estimated Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# Plot 4
# Clean speech vs NLMS output
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    clean_low[:plot_samples],
    label="Clean speech"
)

plt.plot(
    time,
    output[:plot_samples],
    label="NLMS output",
    alpha=0.7
)

plt.title(
    "Stage 3D - Clean Speech vs NLMS Output"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ============================================================
# Plot 5
# Coefficient convergence
# ============================================================

plt.figure(figsize=(12, 5))

coefficient_time = (
    np.arange(N) /
    FS
)

plt.plot(
    coefficient_time,
    coefficient_norm
)

plt.title(
    "Stage 3D - NLMS Coefficient Norm"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Coefficient Norm")

plt.grid(True)

plt.tight_layout()


# ============================================================
# Show all plots
# ============================================================

plt.show()


print("\nSTAGE 3D COMPLETE")