#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',default='results/baselines.csv'); ap.add_argument('--hybrid',default='results/hybrid_fullband.csv'); ap.add_argument('--out',default='results/plots'); a=ap.parse_args()
    import matplotlib.pyplot as plt
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    frames=[]
    if Path(a.baseline).exists(): frames.append(pd.read_csv(a.baseline))
    if Path(a.hybrid).exists(): frames.append(pd.read_csv(a.hybrid))
    if not frames: raise FileNotFoundError('No result CSVs found.')
    df=pd.concat(frames,ignore_index=True)
    for metric in ['Delta_SI_SDR_dB','Delta_SNR_dB']:
        g=df.groupby('Method')[metric].mean().sort_values()
        fig,ax=plt.subplots(figsize=(9,4.5)); g.plot.bar(ax=ax); ax.set_ylabel(metric); ax.set_title(metric+' comparison'); fig.tight_layout(); fig.savefig(out/(metric+'_comparison.png'),dpi=180); plt.close(fig)
    if 'SNR_dB' in df:
        g=df.groupby(['Method','SNR_dB'])['Delta_SI_SDR_dB'].mean().unstack(0)
        fig,ax=plt.subplots(figsize=(9,5)); g.plot(ax=ax,marker='o'); ax.set_ylabel('Delta SI-SDR (dB)'); ax.set_xlabel('Input SNR (dB)'); ax.set_title('SI-SDR improvement vs SNR'); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(out/'si_sdr_vs_snr.png',dpi=180); plt.close(fig)
    if 'Noise_Type' in df:
        g=df.groupby(['Noise_Type','Method'])['Delta_SI_SDR_dB'].mean().unstack(1)
        fig,ax=plt.subplots(figsize=(10,5)); g.plot.bar(ax=ax); ax.set_ylabel('Delta SI-SDR (dB)'); ax.set_title('Performance by noise category'); fig.tight_layout(); fig.savefig(out/'by_noise_type.png',dpi=180); plt.close(fig)
    print('Saved plots to',out)
if __name__=='__main__': main()
