import numpy as np
from scipy.io import wavfile


# ============================================================
# SETTINGS
# ============================================================

FS = 24000

NUM_TAPS = 128

MU_ADAPT = 0.1
MU_FREEZE = 0.0

EPSILON = 1e-8

# 20 ms analysis frame
FRAME_LENGTH = int(0.020 * FS)

# 10 ms hop
HOP_LENGTH = int(0.010 * FS)

# Search for relative microphone delay
MAX_LAG = 64

# Correlation required before NLMS is allowed to adapt
CORRELATION_THRESHOLD = 0.35

# Minimum reference energy
REFERENCE_ENERGY_THRESHOLD = 1e-5

# Evaluation starts after initial transient
EVAL_START = int(0.5 * FS)


# ============================================================
# LOAD SIGNALS
# ============================================================

primary = np.load(
    "results/primary_low_80_1500.npy"
).astype(np.float64)

reference = np.load(
    "results/reference_low_80_1500.npy"
).astype(np.float64)


N = min(
    len(primary),
    len(reference)
)

primary = primary[:N]
reference = reference[:N]


# ============================================================
# NORMALIZED CROSS-CORRELATION
# ============================================================

def maximum_normalized_correlation(
    primary_frame,
    reference_frame,
    max_lag
):

    primary_frame = (
        primary_frame -
        np.mean(primary_frame)
    )

    reference_frame = (
        reference_frame -
        np.mean(reference_frame)
    )

    primary_norm = np.linalg.norm(
        primary_frame
    )

    reference_norm = np.linalg.norm(
        reference_frame
    )

    if (
        primary_norm < EPSILON
        or reference_norm < EPSILON
    ):
        return 0.0

    best_correlation = 0.0

    # Search positive and negative relative delays
    for lag in range(
        -max_lag,
        max_lag + 1
    ):

        if lag < 0:

            p = primary_frame[-lag:]
            r = reference_frame[:len(p)]

        elif lag > 0:

            p = primary_frame[:-lag]
            r = reference_frame[lag:]

        else:

            p = primary_frame
            r = reference_frame

        if len(p) < 10:
            continue

        denominator = (
            np.linalg.norm(p)
            * np.linalg.norm(r)
            + EPSILON
        )

        correlation = (
            np.dot(p, r)
            / denominator
        )

        best_correlation = max(
            best_correlation,
            abs(correlation)
        )

    return best_correlation


# ============================================================
# FRAME-BASED CORRELATION DETECTOR
# ============================================================

correlation_history = np.zeros(N)
reference_energy = np.zeros(N)

speech_detected = np.zeros(
    N,
    dtype=bool
)


for start in range(
    0,
    N - FRAME_LENGTH,
    HOP_LENGTH
):

    end = start + FRAME_LENGTH

    primary_frame = primary[start:end]
    reference_frame = reference[start:end]

    # Reference RMS
    ref_rms = np.sqrt(
        np.mean(
            reference_frame ** 2
        )
    )

    correlation = (
        maximum_normalized_correlation(
            primary_frame,
            reference_frame,
            MAX_LAG
        )
    )

    correlation_history[start:end] = (
        correlation
    )

    reference_energy[start:end] = (
        ref_rms
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------
    #
    # Strong primary/reference correlation:
    #     likely reference-correlated noise
    #
    # Weak correlation:
    #     possible speech / uncorrelated content
    #

    if (
        correlation <
        CORRELATION_THRESHOLD
        and
        ref_rms >
        REFERENCE_ENERGY_THRESHOLD
    ):

        speech_detected[start:end] = True


# ============================================================
# ADAPTATION DECISION
# ============================================================

# NLMS adapts only when:
#
# 1. Primary and reference are sufficiently correlated
# 2. Reference contains enough energy

adaptation_allowed = (
    (correlation_history >=
     CORRELATION_THRESHOLD)
    &
    (reference_energy >
     REFERENCE_ENERGY_THRESHOLD)
)


# ============================================================
# NLMS
# ============================================================

weights = np.zeros(
    NUM_TAPS
)

reference_buffer = np.zeros(
    NUM_TAPS
)

output = np.zeros(N)

mu_history = np.zeros(N)


for n in range(N):

    # --------------------------------------------------------
    # Update reference buffer
    # --------------------------------------------------------

    reference_buffer[1:] = (
        reference_buffer[:-1]
    )

    reference_buffer[0] = (
        reference[n]
    )


    # --------------------------------------------------------
    # Estimate correlated noise
    # --------------------------------------------------------

    predicted_noise = np.dot(
        weights,
        reference_buffer
    )


    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    error = (
        primary[n]
        - predicted_noise
    )

    output[n] = error


    # --------------------------------------------------------
    # ADAPTATION CONTROL
    # --------------------------------------------------------

    if adaptation_allowed[n]:

        mu = MU_ADAPT

    else:

        mu = MU_FREEZE


    mu_history[n] = mu


    # --------------------------------------------------------
    # NLMS UPDATE
    # --------------------------------------------------------

    if mu > 0:

        reference_power = np.dot(
            reference_buffer,
            reference_buffer
        )

        weights += (
            mu
            * error
            * reference_buffer
            /
            (
                EPSILON
                + reference_power
            )
        )


# ============================================================
# LOAD CLEAN SPEECH FOR EVALUATION ONLY
# ============================================================

clean_reference = np.load(
    "results/nlms_stage3e_frozen_clean_low.npy"
).astype(np.float64)


N_eval = min(
    len(clean_reference),
    len(output)
)

clean_reference = (
    clean_reference[:N_eval]
)

primary_eval = (
    primary[:N_eval]
)

output_eval = (
    output[:N_eval]
)


# ============================================================
# SNR FUNCTION
# ============================================================

def calculate_snr(
    clean,
    signal
):

    noise = (
        signal - clean
    )

    signal_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    return 10 * np.log10(
        signal_power
        /
        (noise_power + 1e-15)
    )


# ============================================================
# EVALUATION
# ============================================================

clean_eval = (
    clean_reference[EVAL_START:]
)

primary_eval = (
    primary_eval[EVAL_START:]
)

output_eval = (
    output_eval[EVAL_START:]
)


input_snr = calculate_snr(
    clean_eval,
    primary_eval
)

output_snr = calculate_snr(
    clean_eval,
    output_eval
)

improvement = (
    output_snr -
    input_snr
)


# ============================================================
# STATISTICS
# ============================================================

speech_percentage = (
    100 *
    np.mean(speech_detected)
)

adaptation_percentage = (
    100 *
    np.mean(adaptation_allowed)
)

mean_correlation = np.mean(
    correlation_history[
        EVAL_START:
    ]
)

max_correlation = np.max(
    correlation_history[
        EVAL_START:
    ]
)

min_correlation = np.min(
    correlation_history[
        EVAL_START:
    ]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print(
    "STAGE 4D CORRELATION-BASED "
    "DOUBLE-TALK DETECTION"
)
print("=" * 60)

print()

print(
    f"Correlation threshold  : "
    f"{CORRELATION_THRESHOLD:.2f}"
)

print(
    f"Maximum search lag     : "
    f"{MAX_LAG} samples"
)

print()

print(
    f"Mean correlation       : "
    f"{mean_correlation:.4f}"
)

print(
    f"Maximum correlation    : "
    f"{max_correlation:.4f}"
)

print(
    f"Minimum correlation    : "
    f"{min_correlation:.4f}"
)

print()

print(
    f"Speech/uncorrelated est: "
    f"{speech_percentage:.2f}%"
)

print(
    f"NLMS adaptation time   : "
    f"{adaptation_percentage:.2f}%"
)

print()

print(
    f"Input SNR              : "
    f"{input_snr:.2f} dB"
)

print(
    f"Output SNR             : "
    f"{output_snr:.2f} dB"
)

print(
    f"SNR improvement        : "
    f"{improvement:.2f} dB"
)

print()

print(
    f"Final coefficient norm : "
    f"{np.linalg.norm(weights):.6f}"
)

largest_tap = np.argmax(
    np.abs(weights)
)

print(
    f"Largest coefficient tap: "
    f"{largest_tap}"
)

print(
    f"Largest coefficient     : "
    f"{weights[largest_tap]:.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

max_output = (
    np.max(
        np.abs(output)
    )
    + 1e-12
)

wavfile.write(
    "results/nlms_stage4d_correlation.wav",
    FS,
    np.int16(
        np.clip(
            output / max_output,
            -1,
            1
        )
        * 32767
    )
)


np.save(
    "results/nlms_stage4d_correlation.npy",
    output
)

np.save(
    "results/nlms_stage4d_correlation_history.npy",
    correlation_history
)

np.save(
    "results/nlms_stage4d_speech_activity.npy",
    speech_detected
)

np.save(
    "results/nlms_stage4d_coefficients.npy",
    weights
)


print()

print("Output files:")

print(
    "results/nlms_stage4d_correlation.wav"
)

print(
    "results/nlms_stage4d_correlation.npy"
)

print(
    "results/nlms_stage4d_correlation_history.npy"
)

print(
    "results/nlms_stage4d_speech_activity.npy"
)

print(
    "results/nlms_stage4d_coefficients.npy"
)

print()

print("=" * 60)