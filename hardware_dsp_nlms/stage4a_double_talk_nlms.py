import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


# ============================================================
# SETTINGS
# ============================================================

FS = 24000

NUM_TAPS = 128
MU_ACTIVE = 0.0       # Freeze adaptation during speech
MU_SILENCE = 0.1      # Adapt when speech is absent
EPSILON = 1e-8

TARGET_SNR_DB = 5.0

REFERENCE_DELAY = 5
REFERENCE_GAIN = 0.8

PRIMARY_DELAY_1 = 20
PRIMARY_GAIN_1 = 0.65

PRIMARY_DELAY_2 = 35
PRIMARY_GAIN_2 = 0.75

PATH_CHANGE_TIME = 5.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_npy(filename):
    return np.load(filename).astype(np.float64)


def delayed_signal(signal, delay):
    output = np.zeros_like(signal)

    if delay == 0:
        return signal.copy()

    output[delay:] = signal[:-delay]

    return output


def calculate_snr(clean, signal):
    noise = signal - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(signal_power / (noise_power + 1e-15))


def create_noise_path(reference, delay, gain):
    return gain * delayed_signal(reference, delay)


# ============================================================
# LOAD STAGE 2 SIGNALS
# ============================================================

clean_low = load_npy(
    "results/primary_low_80_1500.npy"
)

reference_low = load_npy(
    "results/reference_low_80_1500.npy"
)

N = min(len(clean_low), len(reference_low))

clean_low = clean_low[:N]
reference_low = reference_low[:N]


# ============================================================
# CREATE CONTROLLED SPEECH SIGNAL
# ============================================================

# The Stage 2 primary-low signal contains speech + noise.
# For this controlled experiment we need clean speech.
#
# We use the original primary-low file only as a reference
# for the speech waveform here, so we construct a clean
# speech approximation by using the known simulation source.
#
# IMPORTANT:
# This is ONLY a controlled validation experiment.


speech_source = load_npy(
    "results/nlms_stage3e_frozen_clean_low.npy"
)

# The Stage 3E clean-low file is already the known clean
# low-band speech used during previous validation.

N = min(N, len(speech_source))

clean_speech = speech_source[:N]
reference_low = reference_low[:N]


# ============================================================
# CREATE CHANGING NOISE PATH
# ============================================================

noise_path_1 = create_noise_path(
    reference_low,
    PRIMARY_DELAY_1,
    PRIMARY_GAIN_1
)

noise_path_2 = create_noise_path(
    reference_low,
    PRIMARY_DELAY_2,
    PRIMARY_GAIN_2
)


change_index = int(PATH_CHANGE_TIME * FS)

primary_noise = noise_path_1.copy()

if change_index < N:
    primary_noise[change_index:] = noise_path_2[change_index:]


# ============================================================
# SCALE NOISE TO TARGET SNR
# ============================================================

speech_power = np.mean(clean_speech ** 2)
noise_power = np.mean(primary_noise ** 2)

target_noise_power = speech_power / (
    10 ** (TARGET_SNR_DB / 10)
)

noise_scale = np.sqrt(
    target_noise_power / (noise_power + 1e-15)
)

primary_noise *= noise_scale


# ============================================================
# CREATE PRIMARY MICROPHONE SIGNAL
# ============================================================

primary = clean_speech + primary_noise


# ============================================================
# CREATE SPEECH ACTIVITY MASK
# ============================================================

# Frame-based energy detector.
#
# Frame length = 20 ms
# Hop = 10 ms

FRAME_LENGTH = int(0.020 * FS)
HOP_LENGTH = int(0.010 * FS)

speech_energy = np.zeros(N)

for start in range(0, N - FRAME_LENGTH, HOP_LENGTH):

    end = start + FRAME_LENGTH

    frame = clean_speech[start:end]

    rms = np.sqrt(
        np.mean(frame ** 2)
    )

    speech_energy[start:end] = rms


# Threshold based on maximum clean-speech energy

threshold = 0.10 * np.max(speech_energy)

speech_active = speech_energy > threshold


# ============================================================
# NLMS
# ============================================================

weights = np.zeros(NUM_TAPS)

reference_buffer = np.zeros(NUM_TAPS)

output = np.zeros(N)

mu_history = np.zeros(N)


for n in range(N):

    # Shift reference buffer

    reference_buffer[1:] = reference_buffer[:-1]

    reference_buffer[0] = reference_low[n]

    # Predicted noise

    predicted_noise = np.dot(
        weights,
        reference_buffer
    )

    # Error / cleaned signal

    error = primary[n] - predicted_noise

    output[n] = error

    # --------------------------------------------------------
    # DOUBLE-TALK PROTECTION
    # --------------------------------------------------------

    if speech_active[n]:

        # Speech present:
        # freeze adaptive coefficients

        mu = MU_ACTIVE

    else:

        # Speech absent:
        # allow NLMS adaptation

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

        normalization = (
            EPSILON + reference_power
        )

        weights += (
            mu
            * error
            * reference_buffer
            / normalization
        )


# ============================================================
# EVALUATION
# ============================================================

# Ignore initial transient

EVAL_START = int(0.5 * FS)

clean_eval = clean_speech[EVAL_START:]
primary_eval = primary[EVAL_START:]
output_eval = output[EVAL_START:]


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
# BEFORE / AFTER PATH CHANGE
# ============================================================

change_eval = max(
    EVAL_START,
    change_index
)

before_clean = clean_speech[EVAL_START:change_index]
before_primary = primary[EVAL_START:change_index]
before_output = output[EVAL_START:change_index]

after_clean = clean_speech[change_eval:]
after_primary = primary[change_eval:]
after_output = output[change_eval:]


before_input_snr = calculate_snr(
    before_clean,
    before_primary
)

before_output_snr = calculate_snr(
    before_clean,
    before_output
)

after_input_snr = calculate_snr(
    after_clean,
    after_primary
)

after_output_snr = calculate_snr(
    after_clean,
    after_output
)


# ============================================================
# RESULTS
# ============================================================

speech_percentage = (
    100 * np.mean(speech_active)
)

print()
print("=" * 60)
print("STAGE 4A DOUBLE-TALK PROTECTION")
print("=" * 60)

print()
print(f"Input SNR              : {input_snr:.2f} dB")
print(f"Output SNR             : {output_snr:.2f} dB")
print(f"SNR improvement        : {improvement:.2f} dB")

print()
print("BEFORE NOISE PATH CHANGE")
print(f"Input SNR              : {before_input_snr:.2f} dB")
print(f"Output SNR             : {before_output_snr:.2f} dB")
print(
    f"SNR improvement        : "
    f"{before_output_snr - before_input_snr:.2f} dB"
)

print()
print("AFTER NOISE PATH CHANGE")
print(f"Input SNR              : {after_input_snr:.2f} dB")
print(f"Output SNR             : {after_output_snr:.2f} dB")
print(
    f"SNR improvement        : "
    f"{after_output_snr - after_input_snr:.2f} dB"
)

print()
print("DOUBLE-TALK PROTECTION")
print(
    f"Speech-active samples  : "
    f"{speech_percentage:.2f}%"
)

print()
print("FINAL FILTER")
print(
    f"Coefficient norm       : "
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

wavfile.write(
    "results/nlms_stage4a_double_talk.wav",
    FS,
    np.int16(
        np.clip(
            output / (np.max(np.abs(output)) + 1e-12),
            -1,
            1
        ) * 32767
    )
)

np.save(
    "results/nlms_stage4a_double_talk.npy",
    output
)

np.save(
    "results/nlms_stage4a_coefficients.npy",
    weights
)

print()
print("Output:")
print("results/nlms_stage4a_double_talk.wav")
print("results/nlms_stage4a_double_talk.npy")
print("results/nlms_stage4a_coefficients.npy")

print()
print("=" * 60)