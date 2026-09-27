import numpy as np

def spectral_flux(x, sr=16000, frame_ms=2.0, hop_ms=2.0):
    x=np.asarray(x,dtype=np.float32)
    n=max(16,int(sr*frame_ms/1000))
    hop=max(1,int(sr*hop_ms/1000))
    prev=None; vals=[]
    win=np.hanning(n).astype(np.float32)
    for i in range(0,max(1,len(x)-n+1),hop):
        mag=np.abs(np.fft.rfft(x[i:i+n]*win))
        mag/=mag.sum()+1e-8
        if prev is not None:
            vals.append(float(np.sqrt(np.sum((mag-prev)**2))))
        prev=mag
    return np.asarray(vals,dtype=np.float32)
