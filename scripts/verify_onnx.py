#!/usr/bin/env python3
"""Verify one or two ONNX artifacts.

Checks model validity, runtime loading, finite inference output, and—when both models
are supplied—input/output shape compatibility plus a numerical output-difference summary.
"""
import argparse, os, numpy as np, onnx, onnxruntime as ort

def inspect(path):
    model=onnx.load(path); onnx.checker.check_model(model)
    sess=ort.InferenceSession(path,providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0]; out=sess.get_outputs()[0]
    shape=[d if isinstance(d,int) and d>0 else 1 for d in inp.shape]
    x=np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    y=sess.run([out.name],{inp.name:x})[0]
    if not np.isfinite(y).all(): raise RuntimeError(f'NaN/Inf in {path}')
    return {'path':path,'input_shape':inp.shape,'output_shape':y.shape,'output':y,'size_bytes':os.path.getsize(path)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('models',nargs='+'); a=ap.parse_args()
    if len(a.models)>2: ap.error('Provide one or two ONNX models.')
    results=[inspect(p) for p in a.models]
    for r in results:
        print(r['path'],'OK','input',r['input_shape'],'output',r['output_shape'],'size_bytes',r['size_bytes'])
    if len(results)==2:
        a,b=results
        compatible=a['output_shape']==b['output_shape'] and a['input_shape']==b['input_shape']
        print('FP32/INT8 tensor-shape compatible:',compatible)
        if compatible:
            diff=np.asarray(a['output'],dtype=np.float32)-np.asarray(b['output'],dtype=np.float32)
            print('Random-input output MAE:',float(np.mean(np.abs(diff))))
            print('Random-input output max_abs:',float(np.max(np.abs(diff))))
if __name__=='__main__': main()
