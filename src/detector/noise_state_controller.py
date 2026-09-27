from dataclasses import dataclass

@dataclass
class Thresholds:
    flux: float
    kurtosis: float
    snr_db: float

class NoiseStateController:
    """Rule-based controller. Thresholds must be tuned from development data."""
    def __init__(self, thresholds: Thresholds, smoothing_frames=5, hysteresis_frames=3):
        self.t=thresholds; self.smoothing=smoothing_frames; self.hysteresis=hysteresis_frames
        self.state="mixed_uncertain"; self.pending=None; self.count=0

    def classify(self, flux, kurtosis, snr_db=None):
        transient = flux >= self.t.flux or kurtosis >= self.t.kurtosis
        complex_cond = transient or (snr_db is not None and snr_db < self.t.snr_db)
        proposed = "transient" if transient else ("complex" if complex_cond else "stationary")
        if proposed != self.state:
            if proposed == self.pending: self.count += 1
            else: self.pending, self.count = proposed, 1
            if self.count >= self.hysteresis:
                self.state=proposed; self.pending=None; self.count=0
        else:
            self.pending=None; self.count=0
        return self.state

    def control(self):
        return {
            "stationary": {"nlms":"normal","ai":"normal"},
            "complex": {"nlms":"normal","ai":"stronger"},
            "transient": {"nlms":"slow_or_freeze","ai":"priority"},
            "mixed_uncertain": {"nlms":"conservative","ai":"conservative"},
        }[self.state]
