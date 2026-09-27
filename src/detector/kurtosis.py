import numpy as np
from scipy.stats import kurtosis

def frame_kurtosis(x, sr=16000, frame_ms=2.0, hop_ms=2.0):
    x=np.asarray(x,dtype=np.float32)
    n=max(16,int(sr*frame_ms/1000)); hop=max(1,int(sr*hop_ms/1000))
    vals=[]
    for i in range(0,max(1,len(x)-n+1),hop):
        frame=x[i:i+n]
        vals.append(float(kurtosis(frame,fisher=False,bias=False)) if len(frame)>3 else 3.0)
    return np.asarray(vals,dtype=np.float32)
