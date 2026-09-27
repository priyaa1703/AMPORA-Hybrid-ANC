#!/usr/bin/env python3
"""One-command software verification for AMPORA.

PyTorch/unit tests are always run. ONNX verification runs when onnx and onnxruntime
are installed (they are included in requirements.txt for Colab).
"""
import subprocess, sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    commands=[[sys.executable,'-m','pytest','-q']]
    for c in commands:
        print('\n$',' '.join(map(str,c))); r=subprocess.run(c,cwd=root)
        if r.returncode: raise SystemExit(r.returncode)
    try:
        import onnx, onnxruntime  # noqa: F401
    except ImportError:
        print('\nONNX verification skipped: install requirements.txt to enable it.')
        return
    c=[sys.executable,str(root/'scripts'/'verify_onnx.py'),str(root/'models'/'ampora_lite_dccrn_fp32.onnx'),str(root/'models'/'ampora_lite_dccrn_int8.onnx')]
    print('\n$',' '.join(map(str,c))); r=subprocess.run(c,cwd=root)
    if r.returncode: raise SystemExit(r.returncode)
    print('\nAMPORA software verification completed successfully.')
if __name__=='__main__': main()
