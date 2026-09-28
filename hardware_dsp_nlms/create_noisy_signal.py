import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import os

# ============================================================
# STAGE 1
# Create Primary Microphone + Reference Microphone Signals
# Target Sampling Rate = 24 kHz
# ============================================================


# ============================================================
# 1. FILE NAMES
# ============================================================

speech_file = "audio/clean_speech.wav"
noise_file = "audio/engine_noise.wav"

# Output files
primary_file = "results/primary_mic.wav"
reference_file = "results/reference_mic.wav"


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET_FS = 24000       # 24 kHz sampling rate
TARGET_SNR_DB = 5       # Desired SNR = +5 dB


# ============================================================
# 3. CREATE OUTPUT FOLDER IF IT DOES NOT EXIST
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# 4. LOAD CLEAN SPEECH
# ============================================================

print("Loading speech...")

fs_speech, speech = wavfile.read(speech_file)

speech = speech.astype(np.float64)

# Convert stereo to mono
if speech.ndim > 1:
    speech = np.mean(speech, axis=1)

print("Original speech sample rate:", fs_speech)
print("Speech samples:", len(speech))


# ============================================================
# 5. LOAD ENGINE NOISE
# ============================================================

print("\nLoading noise...")

fs_noise, noise = wavfile.read(noise_file)

noise = noise.astype(np.float64)

# Convert stereo to mono
if noise.ndim > 1:
    noise = np.mean(noise, axis=1)

print("Original noise sample rate:", fs_noise)
print("Noise samples:", len(noise))


# ============================================================
# 6. RESAMPLE SPEECH TO 24 kHz
# ============================================================

print("\nResampling to 24 kHz...")

if fs_speech != TARGET_FS:

    speech = resample_poly(
        speech,
        TARGET_FS,
        fs_speech
    )


# ============================================================
# 7. RESAMPLE NOISE TO 24 kHz
# ============================================================

if fs_noise != TARGET_FS:

    noise = resample_poly(
        noise,
        TARGET_FS,
        fs_noise
    )


print("Speech sample rate after resampling:", TARGET_FS)
print("Noise sample rate after resampling:", TARGET_FS)


# ============================================================
# 8. NORMALIZE SPEECH
# ============================================================

speech = speech / (np.max(np.abs(speech)) + 1e-12)


# ============================================================
# 9. NORMALIZE NOISE
# ============================================================

noise = noise / (np.max(np.abs(noise)) + 1e-12)


# ============================================================
# 10. MAKE BOTH SIGNALS SAME LENGTH
# ============================================================

length = min(len(speech), len(noise))

speech = speech[:length]
noise = noise[:length]

print("\nCommon signal length:", length)

duration = length / TARGET_FS

print("Signal duration:", round(duration, 2), "seconds")


# ============================================================
# 11. CALCULATE SPEECH AND NOISE POWER
# ============================================================

speech_power = np.mean(speech ** 2)

noise_power = np.mean(noise ** 2)


# ============================================================
# 12. CALCULATE REQUIRED NOISE LEVEL
# ============================================================

# SNR equation:
#
# SNR(dB) = 10 log10(Pspeech / Pnoise)
#
# Therefore:
#
# Pnoise = Pspeech / 10^(SNR/10)

target_snr_linear = 10 ** (TARGET_SNR_DB / 10)

desired_noise_power = speech_power / target_snr_linear


# ============================================================
# 13. SCALE NOISE
# ============================================================

noise_scale = np.sqrt(
    desired_noise_power / (noise_power + 1e-12)
)

noise_scaled = noise * noise_scale


# ============================================================
# 14. CREATE PRIMARY MICROPHONE SIGNAL
# ============================================================

# Primary microphone hears:
#
#       Speech + Environmental Noise
#
# d(n) = s(n) + v(n)

primary = speech + noise_scaled


# ============================================================
# 15. CREATE REFERENCE MICROPHONE SIGNAL
# ============================================================

# Reference microphone mainly captures:
#
#       Environmental Noise
#
# x(n) = v(n)

reference = noise_scaled.copy()


# ============================================================
# 16. CHECK FOR CLIPPING
# ============================================================

max_amplitude = max(
    np.max(np.abs(primary)),
    np.max(np.abs(reference))
)

if max_amplitude > 1:

    print("\nAmplitude above 1 detected.")
    print("Scaling signals to prevent clipping...")

    primary = primary / max_amplitude
    reference = reference / max_amplitude


# ============================================================
# 17. CONVERT TO 16-BIT PCM
# ============================================================

primary_int16 = np.int16(
    np.clip(primary, -1, 1) * 32767
)

reference_int16 = np.int16(
    np.clip(reference, -1, 1) * 32767
)


# ============================================================
# 18. SAVE PRIMARY MICROPHONE SIGNAL
# ============================================================

wavfile.write(
    primary_file,
    TARGET_FS,
    primary_int16
)


# ============================================================
# 19. SAVE REFERENCE MICROPHONE SIGNAL
# ============================================================

wavfile.write(
    reference_file,
    TARGET_FS,
    reference_int16
)


# ============================================================
# 20. VERIFY THE OUTPUT FILES
# ============================================================

check_fs_primary, primary_check = wavfile.read(
    primary_file
)

check_fs_reference, reference_check = wavfile.read(
    reference_file
)


# ============================================================
# 21. CALCULATE ACTUAL SNR
# ============================================================

# Convert saved primary back to normalized floating point
primary_check = primary_check.astype(np.float64) / 32767

reference_check = reference_check.astype(np.float64) / 32767


# ============================================================
# 22. FINAL INFORMATION
# ============================================================

print("\n")
print("============================================================")
print("                 STAGE 1 COMPLETE")
print("============================================================")

print("\nSystem settings:")
print("----------------------------")

print("Sampling rate       :", TARGET_FS, "Hz")
print("Nyquist frequency   :", TARGET_FS / 2, "Hz")
print("Target SNR          :", TARGET_SNR_DB, "dB")
print("Signal duration     :", round(duration, 2), "seconds")


print("\nOutput files:")
print("----------------------------")

print(primary_file)
print(reference_file)


print("\nVerification:")
print("----------------------------")

print("Primary microphone sampling rate   :", check_fs_primary, "Hz")
print("Reference microphone sampling rate :", check_fs_reference, "Hz")


print("\n============================================================")
print("Primary microphone:")
print("Speech + Environmental Noise")
print()
print("Reference microphone:")
print("Environmental Noise")
print("============================================================")