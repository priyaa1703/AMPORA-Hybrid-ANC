import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 3E
# NLMS LEARNING WITH FROZEN COEFFICIENTS
#
# Part 1:
#   Learn the relationship between reference noise
#   and primary noise using noise only.
#
# Part 2:
#   Freeze the learned coefficients.
#
# Part 3:
#   Apply the frozen filter to speech + noise.
#
# ============================================================


# ============================================================
# Configuration
# ============================================================

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

# Learn using first 2 seconds of noise
LEARNING_TIME = 2.0

REFERENCE_FILE = "results/reference_low_80_1500.npy"
CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_FILE = "results/nlms_stage3e_frozen_clean_low.wav"
COEFFICIENT_FILE = "results/nlms_stage3e_frozen_coefficients.npy"


# ============================================================
# Helper functions
# ============================================================

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=4):

    sos = butter(
        order,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def lowpass_filter(signal, cutoff, fs, order=4):

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def power(signal):

    return np.mean(signal ** 2)


def calculate_snr(clean, test):

    error = test - clean

    clean_power = power(clean)
    error_power = power(error)

    return 10 * np.log10(
        (clean_power + 1e-12) /
        (error_power + 1e-12)
    )


def calculate_noise_reduction(input_noise, output_noise):

    input_power = power(input_noise)
    output_power = power(output_noise)

    return 10 * np.log10(
        (input_power + 1e-12) /
        (output_power + 1e-12)
    )


# ============================================================
# Load reference noise
# ============================================================

print("Loading reference noise...")

noise = np.load(
    REFERENCE_FILE
).astype(np.float64)

print(
    "Reference noise samples:",
    len(noise)
)


# ============================================================
# Load clean speech
# ============================================================

print("\nLoading clean speech...")

clean_speech, speech_fs = sf.read(
    CLEAN_SPEECH_FILE
)

if clean_speech.ndim > 1:
    clean_speech = np.mean(
        clean_speech,
        axis=1
    )

clean_speech = clean_speech.astype(
    np.float64
)

print(
    "Original speech sample rate:",
    speech_fs
)


# ============================================================
# Resample speech
# ============================================================

if speech_fs != FS:

    print(
        "Resampling clean speech to 24 kHz..."
    )

    clean_speech = resample_poly(
        clean_speech,
        FS,
        speech_fs
    )


print(
    "Speech sample rate after resampling:",
    FS
)


# ============================================================
# Match lengths
# ============================================================

common_length = min(
    len(noise),
    len(clean_speech)
)

noise = noise[:common_length]

clean_speech = clean_speech[:common_length]


print(
    "Common signal length:",
    common_length
)

print(
    "Duration:",
    round(common_length / FS, 2),
    "seconds"
)


# ============================================================
# Create clean low-band ground truth
#
# Same processing path as the primary signal:
#
# Clean speech
#      ↓
# 80 Hz - 8 kHz bandpass
#      ↓
# 1.5 kHz lowpass
#      ↓
# Clean low-band speech
# ============================================================

print(
    "\nCreating clean low-band ground truth..."
)

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
# Normalize underlying noise
# ============================================================

noise = noise / (
    np.max(np.abs(noise)) + 1e-12
)


# ============================================================
# Create reference microphone
# ============================================================

reference = np.zeros_like(noise)

if REFERENCE_DELAY > 0:

    reference[REFERENCE_DELAY:] = (
        REFERENCE_GAIN *
        noise[:-REFERENCE_DELAY]
    )

else:

    reference = (
        REFERENCE_GAIN *
        noise
    )


# ============================================================
# Create primary microphone noise
# ============================================================

primary_noise = np.zeros_like(noise)

if PRIMARY_DELAY > 0:

    primary_noise[PRIMARY_DELAY:] = (
        PRIMARY_GAIN *
        noise[:-PRIMARY_DELAY]
    )

else:

    primary_noise = (
        PRIMARY_GAIN *
        noise
    )


# ============================================================
# Scale primary noise to target SNR
# ============================================================

clean_power = power(
    clean_low
)

noise_power = power(
    primary_noise
)

target_noise_power = (
    clean_power /
    (10 ** (TARGET_SNR_DB / 10))
)

noise_scale = np.sqrt(
    target_noise_power /
    (noise_power + 1e-12)
)

primary_noise = (
    primary_noise *
    noise_scale
)


# ============================================================
# Create primary microphone signal
# ============================================================

primary = (
    clean_low +
    primary_noise
)


# ============================================================
# Display microphone configuration
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


# ============================================================
# INPUT SNR
# ============================================================

input_snr = calculate_snr(
    clean_low,
    primary
)

print(
    "\nInput SNR:",
    round(input_snr, 2),
    "dB"
)


# ============================================================
# PART 1
# NOISE-ONLY NLMS LEARNING
# ============================================================

print("\n")
print("==============================================")
print("       PART 1 - NOISE ONLY LEARNING")
print("==============================================")


# Use only the first LEARNING_TIME seconds
learning_samples = int(
    LEARNING_TIME * FS
)

learning_samples = min(
    learning_samples,
    common_length
)


print(
    "\nLearning duration:",
    LEARNING_TIME,
    "seconds"
)

print(
    "Learning samples:",
    learning_samples
)


# ------------------------------------------------------------
# Learning signals
# ------------------------------------------------------------

x_learning = reference[:learning_samples]

d_learning = primary_noise[:learning_samples]


# ------------------------------------------------------------
# Initialize filter
# ------------------------------------------------------------

weights = np.zeros(
    NUM_TAPS
)

learning_output = np.zeros(
    learning_samples
)

coefficient_norm = np.zeros(
    learning_samples
)


# ------------------------------------------------------------
# NLMS learning
# ------------------------------------------------------------

for n in range(
    NUM_TAPS,
    learning_samples
):

    x_vector = x_learning[
        n - NUM_TAPS + 1 :
        n + 1
    ][::-1]


    # Estimate primary noise
    y = np.dot(
        weights,
        x_vector
    )

    learning_output[n] = y


    # Noise-only error
    e = (
        d_learning[n] -
        y
    )


    # Input energy
    input_energy = np.dot(
        x_vector,
        x_vector
    )


    # NLMS update
    weights = (
        weights +
        MU *
        e *
        x_vector /
        (
            EPSILON +
            input_energy
        )
    )


    coefficient_norm[n] = (
        np.linalg.norm(weights)
    )


# ============================================================
# Check noise-only learning
# ============================================================

noise_input_eval = d_learning[
    NUM_TAPS:
]

noise_estimation_error = (
    d_learning[NUM_TAPS:]
    -
    learning_output[NUM_TAPS:]
)


noise_reduction_learning = (
    calculate_noise_reduction(
        noise_input_eval,
        noise_estimation_error
    )
)


print(
    "\nNoise-only learning result:"
)

print(
    "Noise reduction:",
    round(
        noise_reduction_learning,
        2
    ),
    "dB"
)

print(
    "Coefficient norm:",
    round(
        np.linalg.norm(weights),
        6
    )
)


# ============================================================
# PART 2
# FREEZE COEFFICIENTS
# ============================================================

frozen_weights = weights.copy()


np.save(
    COEFFICIENT_FILE,
    frozen_weights
)


print(
    "\nCoefficients frozen."
)

print(
    "Saved:",
    COEFFICIENT_FILE
)


largest_tap = np.argmax(
    np.abs(frozen_weights)
)

print(
    "Largest coefficient tap:",
    largest_tap
)

print(
    "Largest coefficient value:",
    round(
        frozen_weights[largest_tap],
        6
    )
)


# ============================================================
# PART 3
# APPLY FROZEN FILTER TO SPEECH + NOISE
# ============================================================

print("\n")
print("==============================================")
print("       PART 3 - FROZEN FILTER TEST")
print("==============================================")


N = len(primary)

output = np.zeros(
    N
)

estimated_noise = np.zeros(
    N
)


for n in range(
    NUM_TAPS,
    N
):

    x_vector = reference[
        n - NUM_TAPS + 1 :
        n + 1
    ][::-1]


    # --------------------------------------------------------
    # Estimate noise using FROZEN coefficients
    # --------------------------------------------------------

    y = np.dot(
        frozen_weights,
        x_vector
    )


    estimated_noise[n] = y


    # --------------------------------------------------------
    # Subtract estimated noise
    # --------------------------------------------------------

    output[n] = (
        primary[n] -
        y
    )


# ============================================================
# Evaluation after startup
# ============================================================

EVALUATION_START = int(
    0.5 * FS
)

clean_eval = clean_low[
    EVALUATION_START:
]

primary_eval = primary[
    EVALUATION_START:
]

output_eval = output[
    EVALUATION_START:
]

actual_noise_eval = primary_noise[
    EVALUATION_START:
]


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
# Direct noise reduction
# ============================================================

residual_noise = (
    output_eval -
    clean_eval
)

noise_reduction = (
    calculate_noise_reduction(
        actual_noise_eval,
        residual_noise
    )
)


# ============================================================
# Print final result
# ============================================================

print("\n")
print("==============================================")
print("       STAGE 3E FROZEN NLMS RESULT")
print("==============================================")


print(
    "\nInput SNR       :",
    round(
        input_snr_eval,
        2
    ),
    "dB"
)

print(
    "Output SNR      :",
    round(
        output_snr_eval,
        2
    ),
    "dB"
)

print(
    "SNR improvement :",
    round(
        snr_improvement,
        2
    ),
    "dB"
)

print(
    "\nDirect noise reduction:",
    round(
        noise_reduction,
        2
    ),
    "dB"
)

print(
    "\nFinal coefficient norm:"
)

print(
    round(
        np.linalg.norm(
            frozen_weights
        ),
        6
    )
)

print(
    "\nLargest coefficient:"
)

print(
    "Tap index :",
    largest_tap
)

print(
    "Value     :",
    round(
        frozen_weights[
            largest_tap
        ],
        6
    )
)


# ============================================================
# Save output
# ============================================================

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
    "results/nlms_stage3e_frozen_clean_low.npy"
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
# PLOTS
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


# ------------------------------------------------------------
# Plot 1 - Primary microphone
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    primary[:plot_samples]
)

plt.title(
    "Stage 3E - Primary Microphone"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 2 - Noise-only learning
# ------------------------------------------------------------

learning_plot_samples = min(
    int(0.05 * FS),
    learning_samples
)

learning_time = (
    np.arange(
        learning_plot_samples
    ) / FS
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    learning_time,
    d_learning[
        :learning_plot_samples
    ],
    label="Actual primary noise"
)

plt.plot(
    learning_time,
    learning_output[
        :learning_plot_samples
    ],
    label="NLMS estimated noise",
    alpha=0.7
)

plt.title(
    "Stage 3E - Noise-Only Learning"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 3 - Clean speech vs primary
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    clean_low[:plot_samples],
    label="Clean speech"
)

plt.plot(
    time,
    primary[:plot_samples],
    label="Speech + noise",
    alpha=0.7
)

plt.title(
    "Stage 3E - Speech + Noise"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 4 - Clean speech vs frozen NLMS output
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    clean_low[:plot_samples],
    label="Clean speech"
)

plt.plot(
    time,
    output[:plot_samples],
    label="Frozen NLMS output",
    alpha=0.7
)

plt.title(
    "Stage 3E - Clean Speech vs Frozen NLMS Output"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 5 - Coefficient convergence during learning
# ------------------------------------------------------------

coefficient_time = (
    np.arange(
        learning_samples
    ) / FS
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    coefficient_time,
    coefficient_norm
)

plt.title(
    "Stage 3E - NLMS Coefficient Norm During Learning"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Coefficient Norm"
)

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 6 - Final filter coefficients
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.stem(
    np.arange(NUM_TAPS),
    frozen_weights
)

plt.title(
    "Stage 3E - Frozen NLMS Filter Coefficients"
)

plt.xlabel(
    "Tap Index"
)

plt.ylabel(
    "Coefficient Value"
)

plt.grid(True)

plt.tight_layout()


# ============================================================
# Show plots
# ============================================================

plt.show()


print("\nSTAGE 3E COMPLETE")