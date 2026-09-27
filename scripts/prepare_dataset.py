#!/usr/bin/env python3
"""Create leakage-resistant speech/noise manifests.

Usage:
  python scripts/prepare_dataset.py --speech data/raw/speech --noise data/raw/stationary data/raw/nonstationary data/raw/transient --out data/processed
"""
import argparse, csv, hashlib, re
from pathlib import Path
from collections import defaultdict
import random

def group_id(p):
    s=p.stem.lower()
    m=re.search(r'(?:speaker|spk|person|voice)[-_]?([a-z0-9]+)',s)
    if m: return "speaker_"+m.group(1)
    parts=s.split("_")
    if len(parts)>=2 and parts[0].isdigit(): return "fsdd_"+parts[1]
    # Safer fallback: each recording is isolated rather than guessing a speaker.
    return "recording_"+hashlib.sha1(str(p).encode()).hexdigest()[:12]

def split(groups, seed=42, val_frac=.2, test_frac=.2):
    keys=list(groups); random.Random(seed).shuffle(keys)
    n=len(keys); nt=max(1,round(n*test_frac)) if n>=3 else 0
    nv=max(1,round(n*val_frac)) if n>=3 else 0
    test=set(keys[:nt]); val=set(keys[nt:nt+nv])
    out={"train":[],"val":[],"test":[]}
    for k in keys:
        out["test" if k in test else "val" if k in val else "train"].extend(groups[k])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--speech",required=True)
    ap.add_argument("--noise",nargs="+",required=True)
    ap.add_argument("--out",default="data/processed")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    speech=sorted(Path(a.speech).rglob("*.wav"))
    groups=defaultdict(list)
    for p in speech: groups[group_id(p)].append(str(p))
    splits=split(groups,a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for name,items in splits.items():
        with open(out/f"{name}_speech.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["path","group_id"])
            for p in sorted(items): w.writerow([p,group_id(Path(p))])
    with open(out/"noise.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","noise_category"])
        for d in a.noise:
            cat=Path(d).name
            for p in sorted(Path(d).rglob("*.wav")): w.writerow([str(p),cat])
    print(f"Speech recordings: {len(speech)}; groups: {len(groups)}")
    print({k:len(v) for k,v in splits.items()})
    print("Noise manifest written. Do not split chunks from a recording across sets.")

if __name__=="__main__": main()
