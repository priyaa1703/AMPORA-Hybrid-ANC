import numpy as np
from scipy.io import wavfile


# ============================================================
# SETTINGS
# ============================================================

FS = 24000

NUM_TAPS = 128

MU_ACTIVE = 0.0
MU_SILENCE = 0.1

EPSILON = 1e-8

# Frame settings
FRAME_LENGTH = int(0.020 * FS)   # 20 ms
HOP_LENGTH = int(0.010 * FS)     # 10 ms

# Detector settings
ENERGY_RATIO_THRESHOLD = 2.0

# Minimum reference energy required before adaptation
REFERENCE_ENERGY_THRESHOLD = 1e-5

# Keep NLMS frozen for this long after speech detection
HANGOVER_MS = 100
HANGOVER_FRAMES = int(
    HANGOVER_MS / 10
)

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
# FRAME ENERGY
# ============================================================

primary_energy = np.zeros(N)
reference_energy = np.zeros(N)


for start in range(
    0,
    N - FRAME_LENGTH,
    HOP_LENGTH
):

    end = start + FRAME_LENGTH

    primary_frame = primary[start:end]
    reference_frame = reference[start:end]

    primary_rms = np.sqrt(
        np.mean(primary_frame ** 2)
    )

    reference_rms = np.sqrt(
        np.mean(reference_frame ** 2)
    )

    primary_energy[start:end] = primary_rms
    reference_energy[start:end] = reference_rms


# ============================================================
# PRACTICAL DOUBLE-TALK DETECTOR
# ============================================================

energy_ratio = (
    primary_energy /
    (reference_energy + 1e-8)
)


# Initial speech decision
speech_detected = (
    energy_ratio >
    ENERGY_RATIO_THRESHOLD
)


# Very quiet region
very_quiet = (
    primary_energy < REFERENCE_ENERGY_THRESHOLD
)

speech_detected[very_quiet] = False


# ============================================================
# HANGOVER
# ============================================================

# Once speech is detected, keep the NLMS frozen for a short
# period even if the detector immediately sees a gap.

speech_active = speech_detected.copy()

hangover_counter = 0

for n in range(N):

    if speech_detected[n]:

        hangover_counter = HANGOVER_FRAMES

    elif hangover_counter > 0:

        speech_active[n] = True

        hangover_counter -= 1


# ============================================================
# ADD REFERENCE-ENERGY CONDITION
# ============================================================

# If the reference microphone has almost no useful signal,
# there is little reason to adapt the filter.

reference_available = (
    reference_energy >
    REFERENCE_ENERGY_THRESHOLD
)


# Adaptation is allowed only when:
#
# 1. Speech is NOT detected
# 2. Reference noise is sufficiently strong

adaptation_allowed = (
    (~speech_active) &
    reference_available
)


# ============================================================
# NLMS
# ============================================================

weights = np.zeros(NUM_TAPS)

reference_buffer = np.zeros(NUM_TAPS)

output = np.zeros(N)

mu_history = np.zeros(N)


for n in range(N):

    # --------------------------------------------------------
    # Update reference buffer
    # --------------------------------------------------------

    reference_buffer[1:] = (
        reference_buffer[:-1]
    )

    reference_buffer[0] = reference[n]


    # --------------------------------------------------------
    # Predict noise
    # --------------------------------------------------------

    predicted_noise = np.dot(
        weights,
        reference_buffer
    )


    # --------------------------------------------------------
    # Cleaned signal
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

        mu = MU_SILENCE

    else:

        mu = MU_ACTIVE


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
            / (
                EPSILON
                + reference_power
            )
        )


# ============================================================
# LOAD CLEAN SPEECH
# ============================================================

# IMPORTANT:
# This clean signal is ONLY used for evaluation.
# The detector and NLMS never see it.

clean_reference = np.load(
    "results/nlms_stage3e_frozen_clean_low.npy"
).astype(np.float64)


N_eval = min(
    len(clean_reference),
    len(output)
)

clean_reference = clean_reference[:N_eval]
output_eval = output[:N_eval]
primary_eval = primary[:N_eval]


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
        signal_power /
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
    np.mean(speech_active)
)

adaptation_percentage = (
    100 *
    np.mean(adaptation_allowed)
)

reference_percentage = (
    100 *
    np.mean(reference_available)
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("STAGE 4C PRACTICAL DOUBLE-TALK DETECTION")
print("=" * 60)

print()

print(
    f"Energy ratio threshold : "
    f"{ENERGY_RATIO_THRESHOLD:.2f}"
)

print(
    f"Hangover time          : "
    f"{HANGOVER_MS} ms"
)

print()

print(
    f"Speech-active estimate : "
    f"{speech_percentage:.2f}%"
)

print(
    f"Reference available    : "
    f"{reference_percentage:.2f}%"
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
# SAVE OUTPUT
# ============================================================

max_output = (
    np.max(
        np.abs(output)
    )
    + 1e-12
)

wavfile.write(
    "results/nlms_stage4c_practical.wav",
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
    "results/nlms_stage4c_practical.npy",
    output
)

np.save(
    "results/nlms_stage4c_speech_activity.npy",
    speech_active
)

np.save(
    "results/nlms_stage4c_coefficients.npy",
    weights
)


print()

print("Output files:")

print(
    "results/nlms_stage4c_practical.wav"
)

print(
    "results/nlms_stage4c_practical.npy"
)

print(
    "results/nlms_stage4c_speech_activity.npy"
)

print(
    "results/nlms_stage4c_coefficients.npy"
)

print()

print("=" * 60)