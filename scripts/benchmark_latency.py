#!/usr/bin/env python3
"""Benchmark ONNX inference only. This is NOT end-to-end audio latency."""
import argparse,time,numpy as np, onnxruntime as ort, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--runs",type=int,default=100)
    ap.add_argument("--f",type=int,default=209); ap.add_argument("--t",type=int,default=161)
    a=ap.parse_args()
    s=ort.InferenceSession(a.model,providers=["CPUExecutionProvider"])
    i=s.get_inputs()[0]; x=np.random.default_rng(0).standard_normal((1,2,a.f,a.t)).astype(np.float32)
    for _ in range(10): s.run(None,{i.name:x})
    t=time.perf_counter()
    for _ in range(a.runs): s.run(None,{i.name:x})
    ms=(time.perf_counter()-t)*1000/a.runs
    print({"model":a.model,"runs":a.runs,"mean_inference_ms":ms,"size_bytes":os.path.getsize(a.model),
           "note":"Model inference only; STFT/iSTFT, buffering and scheduling are excluded."})
if __name__=="__main__": main()
