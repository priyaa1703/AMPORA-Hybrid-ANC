import numpy as np

def estimate_frame_snr_db(clean_like, residual, eps=1e-8):
    a=np.asarray(clean_like,dtype=np.float32); b=np.asarray(residual,dtype=np.float32)
    return float(10*np.log10((np.mean(a*a)+eps)/(np.mean(b*b)+eps)))
