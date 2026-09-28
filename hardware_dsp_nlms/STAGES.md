# Hardware DSP implementation track: NLMS chain (author: @Vincili2005)

Staged Python reference implementation of the low-band, reference-assisted NLMS branch
(primary mic + reference mic), built step by step. Stage descriptions below come from the
script headers and file names.

| Script | Stage |
|---|---|
| `create_noisy_signal.py` | Stage 1: create primary-mic and reference-mic signals (24 kHz) |
| `stage2_filter_split.py` | Stage 2: 80 Hz to 8 kHz band-pass plus 1.5 kHz frequency split |
| `stage3_nlms.py`, `stage3a` to `stage3e` | Stage 3: NLMS validation series (controlled test, pure-noise test, matched two-mic simulation, frozen-coefficient learning) |
| `stage4_adaptive_nlms.py` | Stage 4: controlled adaptive NLMS with adaptation gating |
| `stage4a` to `stage4d` | Stage 4 variants: double-talk handling, practical double-talk, correlation detector |

## How it fits the full system

- Low band (80 Hz to 1.5 kHz): this NLMS chain.
- High band (1.5 kHz to 7.9 kHz): the DC-CRN in the AI/ML part (repo root).

## Integration notes (open items)

- This track runs at 24 kHz; the AI/ML pipeline runs at 16 kHz. Unify the sample rate before joining them end to end.
- These are Python scripts. Any port to a physical DSP board and any measured hardware results should be added here once done.
