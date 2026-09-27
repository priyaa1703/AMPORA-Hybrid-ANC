# SIH demo assets
## SNR sweep (see sweep/metrics_by_snr.csv)
|   Input_SNR_dB |   avg_alpha |   Enhanced_SNR_dB |   Delta_SNR_dB |   Enhanced_SI_SDR_dB |   Enhanced_STOI |   Enhanced_PESQ |
|---------------:|------------:|------------------:|---------------:|---------------------:|----------------:|----------------:|
|            -10 |        1    |             -1.55 |           8.45 |                -5.5  |           0.734 |           1.162 |
|             -5 |        1    |             -1.09 |           3.91 |                -4.94 |           0.766 |           1.196 |
|              0 |        0.75 |              0.98 |           0.98 |                -1.44 |           0.78  |           1.158 |
|              5 |        0.5  |              3.66 |          -1.34 |                 2.37 |           0.799 |           1.195 |
|             10 |        0.25 |              7.99 |          -2.01 |                 7.58 |           0.867 |           1.367 |
|             15 |        0.05 |             14.38 |          -0.62 |                14.44 |           0.95  |           1.784 |

## Clip: hard_noisy (hard_noisy_-10dB)
- noise category: stationary
- Delta_SNR_dB: 12.57, Delta_SI_SDR_dB: 9.79, alpha: 1.00

## Clip: clean_input_no_harm (clean_input_no_harm_15dB)
- noise category: transient
- Delta_SNR_dB: 0.25 (near zero is the GOAL here -- system should barely touch already-clean input)
- alpha: 0.05

## Clip: transient_impulsive (transient_impulsive_3dB)
- noise file: gunfire_ruger_ar_556_dot223_caliber_029.wav
- Delta_SNR_dB: 2.66, Delta_SI_SDR_dB: 1.18, alpha: 0.62
