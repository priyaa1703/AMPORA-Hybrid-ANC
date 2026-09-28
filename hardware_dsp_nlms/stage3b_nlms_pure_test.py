import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STAGE 3B - PURE NOISE NLMS VALIDATION
# ============================================================

REFERENCE_FILE = "results/reference_low_80_1500.npy"

FS = 24000

NUM_TAPS = 128
MU = 0.1
EPSILON = 1e-8

DELAY = 20
GAIN = 0.8


# ============================================================
# LOAD REFERENCE
# ============================================================

print()
print("Loading reference noise...")

reference = np.load(
    REFERENCE_FILE
)

print("Samples:", len(reference))


# ============================================================
# CREATE KNOWN PRIMARY NOISE
# ============================================================

primary = np.zeros_like(reference)

primary[DELAY:] = (
    GAIN * reference[:-DELAY]
)


# ============================================================
# NLMS
# ============================================================

def nlms(d, x, taps, mu, epsilon):

    weights = np.zeros(taps)

    buffer = np.zeros(taps)

    error = np.zeros(len(d))

    estimated = np.zeros(len(d))

    for n in range(len(d)):

        buffer[1:] = buffer[:-1]

        buffer[0] = x[n]

        y = np.dot(
            weights,
            buffer
        )

        e = d[n] - y

        estimated[n] = y
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

    return error, estimated, weights


# ============================================================
# RUN
# ============================================================

print()
print("Running pure-noise NLMS test...")

error, estimated, weights = nlms(
    primary,
    reference,
    NUM_TAPS,
    MU,
    EPSILON
)


# ============================================================
# POWER
# ============================================================

input_power = np.mean(
    primary ** 2
)

output_power = np.mean(
    error ** 2
)


reduction_db = 10 * np.log10(
    input_power /
    (output_power + 1e-12)
)


# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print("       STAGE 3B PURE NOISE TEST")
print("==============================================")

print()
print("Delay       :", DELAY, "samples")
print("Gain        :", GAIN)
print("Taps        :", NUM_TAPS)
print("Step size   :", MU)

print()
print("Input power :", f"{input_power:.8e}")
print("Output power:", f"{output_power:.8e}")

print()
print("Noise reduction:")
print(f"{reduction_db:.2f} dB")

print()
print("Expected coefficient:")
print("Approximately 0.8 at tap", DELAY)

print()
print("Actual coefficient at tap", DELAY, ":")
print(f"{weights[DELAY]:.6f}")

print()
print("Final coefficient norm:")
print(
    f"{np.linalg.norm(weights):.6f}"
)


# ============================================================
# PLOTS
# ============================================================

time = np.arange(
    len(reference)
) / FS

samples = min(
    int(0.1 * FS),
    len(reference)
)


plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    primary[:samples]
)

plt.title(
    "Primary Noise"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


plt.figure(figsize=(12, 4))

plt.plot(
    time[:samples],
    estimated[:samples]
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
    error[:samples]
)

plt.title(
    "NLMS Error After Cancellation"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()


plt.show()