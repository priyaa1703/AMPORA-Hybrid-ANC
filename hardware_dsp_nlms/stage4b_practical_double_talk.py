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

FRAME_LENGTH = int(0.020 * FS)   # 20 ms
HOP_LENGTH = int(0.010 * FS)     # 10 ms

ENERGY_RATIO_THRESHOLD = 2.0

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


N = min(len(primary), len(reference))

primary = primary[:N]
reference = reference[:N]


# ============================================================
# FRAME ENERGY
# ============================================================

primary_energy = np.zeros(N)
reference_energy = np.zeros(N)

for start in range(0, N - FRAME_LENGTH, HOP_LENGTH):

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
# PRACTICAL SPEECH ACTIVITY ESTIMATE
# ============================================================

# When primary energy is considerably larger than reference
# energy, we assume speech is present.
#
# This is only a baseline detector.
#
# It does NOT know the clean speech signal.

energy_ratio = (
    primary_energy /
    (reference_energy + 1e-8)
)


speech_active = (
    energy_ratio > ENERGY_RATIO_THRESHOLD
)


# Ignore regions where both microphones have extremely
# low energy.

minimum_energy = 1e-5

very_quiet = (
    primary_energy < minimum_energy
)

speech_active[very_quiet] = False


# ============================================================
# NLMS
# ============================================================

weights = np.zeros(NUM_TAPS)

reference_buffer = np.zeros(NUM_TAPS)

output = np.zeros(N)

mu_history = np.zeros(N)


for n in range(N):

    # --------------------------------------------------------
    # Reference buffer
    # --------------------------------------------------------

    reference_buffer[1:] = reference_buffer[:-1]

    reference_buffer[0] = reference[n]

    # --------------------------------------------------------
    # Estimate noise
    # --------------------------------------------------------

    predicted_noise = np.dot(
        weights,
        reference_buffer
    )

    # --------------------------------------------------------
    # Error / cleaned output
    # --------------------------------------------------------

    error = primary[n] - predicted_noise

    output[n] = error

    # --------------------------------------------------------
    # ADAPTATION CONTROL
    # --------------------------------------------------------

    if speech_active[n]:

        mu = MU_ACTIVE

    else:

        mu = MU_SILENCE

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
# EVALUATION
# ============================================================

# For evaluation only, we use the known clean signal.
#
# The algorithm itself NEVER used clean speech.

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


def calculate_snr(clean, signal):

    noise = signal - clean

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


clean_eval = clean_reference[EVAL_START:]
primary_eval = primary_eval[EVAL_START:]
output_eval = output_eval[EVAL_START:]


input_snr = calculate_snr(
    clean_eval,
    primary_eval
)

output_snr = calculate_snr(
    clean_eval,
    output_eval
)

improvement = output_snr - input_snr


# ============================================================
# STATISTICS
# ============================================================

speech_percentage = (
    100 * np.mean(speech_active)
)

adaptation_percentage = (
    100 * np.mean(mu_history > 0)
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("STAGE 4B PRACTICAL DOUBLE-TALK DETECTION")
print("=" * 60)

print()

print(
    f"Energy ratio threshold : "
    f"{ENERGY_RATIO_THRESHOLD:.2f}"
)

print(
    f"Speech-active estimate : "
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
# SAVE OUTPUT
# ============================================================

wavfile.write(
    "results/nlms_stage4b_practical.wav",
    FS,
    np.int16(
        np.clip(
            output /
            (np.max(np.abs(output)) + 1e-12),
            -1,
            1
        )
        * 32767
    )
)

np.save(
    "results/nlms_stage4b_practical.npy",
    output
)

np.save(
    "results/nlms_stage4b_speech_activity.npy",
    speech_active
)

np.save(
    "results/nlms_stage4b_coefficients.npy",
    weights
)


print()
print("Output files:")
print(
    "results/nlms_stage4b_practical.wav"
)
print(
    "results/nlms_stage4b_practical.npy"
)
print(
    "results/nlms_stage4b_speech_activity.npy"
)
print(
    "results/nlms_stage4b_coefficients.npy"
)

print()
print("=" * 60)