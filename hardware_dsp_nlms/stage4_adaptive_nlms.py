import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample_poly


# ============================================================
# STAGE 4
# CONTROLLED ADAPTIVE NLMS WITH ADAPTATION GATING
#
# Purpose:
#   1. Learn a changing noise path.
#   2. Freeze adaptation during speech-dominated periods.
#   3. Allow adaptation when the reference noise is dominant.
#   4. Test whether NLMS can track a changing microphone path.
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

# First noise path
PRIMARY_DELAY_1 = 20
PRIMARY_GAIN_1 = 0.65

# Second noise path
PRIMARY_DELAY_2 = 35
PRIMARY_GAIN_2 = 0.75

# Noise path changes halfway
CHANGE_TIME = 5.0

# Adaptation parameters
ADAPTATION_THRESHOLD = 0.15

# Small adaptation step during uncertain periods
MU_SLOW = 0.01

REFERENCE_FILE = "results/reference_low_80_1500.npy"
CLEAN_SPEECH_FILE = "audio/clean_speech.wav"

OUTPUT_FILE = "results/stage4_adaptive_nlms_clean_low.wav"
NPY_OUTPUT_FILE = "results/stage4_adaptive_nlms_clean_low.npy"


# ============================================================
# Helper functions
# ============================================================

def bandpass_filter(
    signal,
    low_cutoff,
    high_cutoff,
    fs,
    order=4
):

    sos = butter(
        order,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def lowpass_filter(
    signal,
    cutoff,
    fs,
    order=4
):

    sos = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def power(signal):

    return np.mean(
        signal ** 2
    )


def calculate_snr(
    clean,
    test
):

    error = (
        test -
        clean
    )

    return 10 * np.log10(
        (
            power(clean) +
            1e-12
        )
        /
        (
            power(error) +
            1e-12
        )
    )


def calculate_noise_reduction(
    input_noise,
    output_noise
):

    return 10 * np.log10(
        (
            power(input_noise) +
            1e-12
        )
        /
        (
            power(output_noise) +
            1e-12
        )
    )


# ============================================================
# Load reference noise
# ============================================================

print(
    "Loading reference noise..."
)

noise = np.load(
    REFERENCE_FILE
).astype(
    np.float64
)

print(
    "Reference noise samples:",
    len(noise)
)


# ============================================================
# Load clean speech
# ============================================================

print(
    "\nLoading clean speech..."
)

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

noise = noise[
    :common_length
]

clean_speech = clean_speech[
    :common_length
]

print(
    "Common signal length:",
    common_length
)

print(
    "Duration:",
    round(
        common_length / FS,
        2
    ),
    "seconds"
)


# ============================================================
# Create clean low-band ground truth
#
# Same primary signal path:
#
# clean speech
#      ↓
# 80 Hz - 8 kHz
#      ↓
# 1.5 kHz lowpass
#      ↓
# clean_low
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
    np.max(
        np.abs(noise)
    )
    +
    1e-12
)


# ============================================================
# Create reference microphone
# ============================================================

reference = np.zeros_like(
    noise
)

if REFERENCE_DELAY > 0:

    reference[
        REFERENCE_DELAY:
    ] = (
        REFERENCE_GAIN *
        noise[
            :-REFERENCE_DELAY
        ]
    )

else:

    reference = (
        REFERENCE_GAIN *
        noise
    )


# ============================================================
# Create TWO primary noise paths
#
# Path 1:
#   delay = 20 samples
#   gain  = 0.65
#
# Path 2:
#   delay = 35 samples
#   gain  = 0.75
#
# This simulates a changing acoustic/environmental path.
# ============================================================

primary_noise_1 = np.zeros_like(
    noise
)

primary_noise_2 = np.zeros_like(
    noise
)


if PRIMARY_DELAY_1 > 0:

    primary_noise_1[
        PRIMARY_DELAY_1:
    ] = (
        PRIMARY_GAIN_1 *
        noise[
            :-PRIMARY_DELAY_1
        ]
    )

else:

    primary_noise_1 = (
        PRIMARY_GAIN_1 *
        noise
    )


if PRIMARY_DELAY_2 > 0:

    primary_noise_2[
        PRIMARY_DELAY_2:
    ] = (
        PRIMARY_GAIN_2 *
        noise[
            :-PRIMARY_DELAY_2
        ]
    )

else:

    primary_noise_2 = (
        PRIMARY_GAIN_2 *
        noise
    )


# ============================================================
# Create changing primary noise
# ============================================================

change_sample = int(
    CHANGE_TIME * FS
)

primary_noise = np.zeros_like(
    noise
)

primary_noise[
    :change_sample
] = primary_noise_1[
    :change_sample
]

primary_noise[
    change_sample:
] = primary_noise_2[
    change_sample:
]


# ============================================================
# Scale primary noise for target SNR
# ============================================================

clean_power = power(
    clean_low
)

noise_power = power(
    primary_noise
)

target_noise_power = (
    clean_power /
    (
        10 **
        (
            TARGET_SNR_DB /
            10
        )
    )
)

noise_scale = np.sqrt(
    target_noise_power /
    (
        noise_power +
        1e-12
    )
)

primary_noise *= (
    noise_scale
)


# ============================================================
# Create primary microphone
# ============================================================

primary = (
    clean_low +
    primary_noise
)


# ============================================================
# Input SNR
# ============================================================

input_snr = calculate_snr(
    clean_low,
    primary
)


print(
    "\nMicrophone simulation:"
)

print(
    "Reference delay:",
    REFERENCE_DELAY
)

print(
    "Reference gain:",
    REFERENCE_GAIN
)

print(
    "\nNoise path 1:"
)

print(
    "Delay:",
    PRIMARY_DELAY_1
)

print(
    "Gain:",
    PRIMARY_GAIN_1
)

print(
    "\nNoise path 2:"
)

print(
    "Delay:",
    PRIMARY_DELAY_2
)

print(
    "Gain:",
    PRIMARY_GAIN_2
)

print(
    "\nPath change time:",
    CHANGE_TIME,
    "seconds"
)

print(
    "\nInput SNR:",
    round(
        input_snr,
        2
    ),
    "dB"
)


# ============================================================
# NLMS INITIALIZATION
# ============================================================

weights = np.zeros(
    NUM_TAPS
)

output = np.zeros(
    common_length
)

estimated_noise = np.zeros(
    common_length
)

coefficient_norm = np.zeros(
    common_length
)

adaptation_state = np.zeros(
    common_length
)


# ============================================================
# Running energy estimates
#
# These help us decide whether the current block is suitable
# for adaptation.
# ============================================================

reference_energy = 0.0
primary_energy = 0.0

ENERGY_ALPHA = 0.995


# ============================================================
# ADAPTIVE NLMS
# ============================================================

print(
    "\nRunning adaptive NLMS..."
)


for n in range(
    NUM_TAPS,
    common_length
):

    # --------------------------------------------------------
    # Reference vector
    # --------------------------------------------------------

    x_vector = reference[
        n - NUM_TAPS + 1:
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
    # Error / cleaned output
    # --------------------------------------------------------

    e = (
        primary[n] -
        y
    )

    output[n] = e


    # --------------------------------------------------------
    # Energy estimation
    # --------------------------------------------------------

    x_energy = np.mean(
        x_vector ** 2
    )

    d_energy = (
        primary[n] ** 2
    )


    reference_energy = (
        ENERGY_ALPHA *
        reference_energy
        +
        (1 - ENERGY_ALPHA) *
        x_energy
    )

    primary_energy = (
        ENERGY_ALPHA *
        primary_energy
        +
        (1 - ENERGY_ALPHA) *
        d_energy
    )


    # --------------------------------------------------------
    # Adaptation decision
    #
    # Reference energy must be meaningful.
    #
    # When the reference signal is strong relative to the
    # primary signal, allow normal adaptation.
    #
    # Otherwise use slow adaptation.
    # --------------------------------------------------------

    energy_ratio = (
        reference_energy /
        (
            primary_energy +
            1e-12
        )
    )


    if energy_ratio > ADAPTATION_THRESHOLD:

        current_mu = MU

        adaptation_state[n] = 1

    else:

        current_mu = MU_SLOW

        adaptation_state[n] = 0.25


    # --------------------------------------------------------
    # NLMS coefficient update
    # --------------------------------------------------------

    input_energy = np.dot(
        x_vector,
        x_vector
    )

    weights = (
        weights
        +
        current_mu *
        e *
        x_vector
        /
        (
            EPSILON +
            input_energy
        )
    )


    # --------------------------------------------------------
    # Coefficient norm
    # --------------------------------------------------------

    coefficient_norm[n] = (
        np.linalg.norm(
            weights
        )
    )


# ============================================================
# Evaluation
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
# Overall SNR
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
# Evaluate BEFORE noise-path change
# ============================================================

before_start = int(
    1.0 * FS
)

before_end = int(
    CHANGE_TIME * FS
)

clean_before = clean_low[
    before_start:
    before_end
]

primary_before = primary[
    before_start:
    before_end
]

output_before = output[
    before_start:
    before_end
]

noise_before = primary_noise[
    before_start:
    before_end
]


snr_before_input = calculate_snr(
    clean_before,
    primary_before
)

snr_before_output = calculate_snr(
    clean_before,
    output_before
)

snr_before_improvement = (
    snr_before_output -
    snr_before_input
)

noise_reduction_before = (
    calculate_noise_reduction(
        noise_before,
        output_before -
        clean_before
    )
)


# ============================================================
# Evaluate AFTER noise-path change
# ============================================================

after_start = int(
    (CHANGE_TIME + 0.5) *
    FS
)

after_end = common_length

clean_after = clean_low[
    after_start:
    after_end
]

primary_after = primary[
    after_start:
    after_end
]

output_after = output[
    after_start:
    after_end
]

noise_after = primary_noise[
    after_start:
    after_end
]


snr_after_input = calculate_snr(
    clean_after,
    primary_after
)

snr_after_output = calculate_snr(
    clean_after,
    output_after
)

snr_after_improvement = (
    snr_after_output -
    snr_after_input
)

noise_reduction_after = (
    calculate_noise_reduction(
        noise_after,
        output_after -
        clean_after
    )
)


# ============================================================
# Final coefficient information
# ============================================================

final_norm = np.linalg.norm(
    weights
)

largest_tap = np.argmax(
    np.abs(weights)
)

largest_value = (
    weights[
        largest_tap
    ]
)


# ============================================================
# Print results
# ============================================================

print("\n")
print("==============================================")
print("       STAGE 4 ADAPTIVE NLMS RESULT")
print("==============================================")


print(
    "\nOVERALL:"
)

print(
    "Input SNR       :",
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
    "Noise reduction :",
    round(
        noise_reduction,
        2
    ),
    "dB"
)


print(
    "\nBEFORE NOISE PATH CHANGE:"
)

print(
    "Input SNR       :",
    round(
        snr_before_input,
        2
    ),
    "dB"
)

print(
    "Output SNR      :",
    round(
        snr_before_output,
        2
    ),
    "dB"
)

print(
    "SNR improvement :",
    round(
        snr_before_improvement,
        2
    ),
    "dB"
)

print(
    "Noise reduction :",
    round(
        noise_reduction_before,
        2
    ),
    "dB"
)


print(
    "\nAFTER NOISE PATH CHANGE:"
)

print(
    "Input SNR       :",
    round(
        snr_after_input,
        2
    ),
    "dB"
)

print(
    "Output SNR      :",
    round(
        snr_after_output,
        2
    ),
    "dB"
)

print(
    "SNR improvement :",
    round(
        snr_after_improvement,
        2
    ),
    "dB"
)

print(
    "Noise reduction :",
    round(
        noise_reduction_after,
        2
    ),
    "dB"
)


print(
    "\nFINAL FILTER:"
)

print(
    "Coefficient norm:",
    round(
        final_norm,
        6
    )
)

print(
    "Largest tap:",
    largest_tap
)

print(
    "Largest coefficient:",
    round(
        largest_value,
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


np.save(
    NPY_OUTPUT_FILE,
    output
)


print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
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
    int(
        plot_duration *
        FS
    ),
    common_length
)

time = (
    np.arange(
        plot_samples
    ) / FS
)


# ------------------------------------------------------------
# Plot 1
# Primary microphone
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    primary[
        :plot_samples
    ]
)

plt.title(
    "Stage 4 - Primary Microphone"
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
# Plot 2
# Actual noise path
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

full_time = (
    np.arange(
        common_length
    ) / FS
)

plt.plot(
    full_time,
    primary_noise
)

plt.axvline(
    CHANGE_TIME,
    linestyle="--"
)

plt.title(
    "Stage 4 - Changing Primary Noise Path"
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
# Plot 3
# Clean vs output
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    clean_low[
        :plot_samples
    ],
    label="Clean speech"
)

plt.plot(
    time,
    output[
        :plot_samples
    ],
    label="Adaptive NLMS output",
    alpha=0.7
)

plt.title(
    "Stage 4 - Clean Speech vs Adaptive NLMS Output"
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
# Plot 4
# Coefficient norm
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    full_time,
    coefficient_norm
)

plt.axvline(
    CHANGE_TIME,
    linestyle="--"
)

plt.title(
    "Stage 4 - Adaptive Filter Coefficient Norm"
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
# Plot 5
# Adaptation state
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    full_time,
    adaptation_state
)

plt.axvline(
    CHANGE_TIME,
    linestyle="--"
)

plt.title(
    "Stage 4 - NLMS Adaptation State"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Adaptation Level"
)

plt.ylim(
    0,
    1.1
)

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# Plot 6
# Final filter coefficients
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.stem(
    np.arange(
        NUM_TAPS
    ),
    weights
)

plt.title(
    "Stage 4 - Final Adaptive Filter Coefficients"
)

plt.xlabel(
    "Tap Index"
)

plt.ylabel(
    "Coefficient"

)

plt.grid(True)

plt.tight_layout()


# ============================================================
# Show plots
# ============================================================

plt.show()


print(
    "\nSTAGE 4 COMPLETE"
)