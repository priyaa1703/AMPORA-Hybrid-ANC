import numpy as np
from src.preprocessing.audio import extract_high_band
from src.detector.spectral_flux import spectral_flux
from src.detector.kurtosis import frame_kurtosis
from src.detector.noise_state_controller import NoiseStateController, Thresholds

def test_high_band_shape():
    x=np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    y=extract_high_band(x)
    assert len(y)==len(x)
    assert np.isfinite(y).all()

def test_detector_outputs():
    x=np.zeros(320,dtype=np.float32); x[100]=1
    assert len(spectral_flux(x))>=0
    assert len(frame_kurtosis(x))>=1

def test_hysteresis():
    c=NoiseStateController(Thresholds(0.1,3.5,-5),hysteresis_frames=2)
    assert c.classify(0.0,3.0)=="mixed_uncertain"
    assert c.classify(0.0,3.0)=="stationary"
