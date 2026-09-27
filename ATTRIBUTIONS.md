# Attribution for bundled real audio (data/raw/speech_real, data/raw/noise_real)

This project bundles a small amount of real, properly-licensed audio so the pipeline
can be exercised end-to-end with genuine human speech and genuine recorded noise
(not only synthetic signals). Every file below is Creative-Commons licensed by its
original author; this section provides the attribution required by CC-BY and
documents the CC0 files for completeness. If you need a different license for your
use case (e.g. a fully public-domain-only corpus), remove this data and substitute
your own.

## Speech (`data/raw/speech_real/`)

Source repository: https://github.com/voxserv/audio_quality_testing_samples
(itself redistributing recordings originally posted to freesound.org under
Creative Commons licenses). Files here are re-sliced into ~3 second segments and
resampled to 16 kHz mono from the originals; segment filenames are prefixed
`speaker_<name>_...` so `scripts/prepare_dataset.py`'s speaker-grouping keeps all
segments from one original recording in the same train/val/test split (no leakage).

| Original recording | Author (freesound.org profile) |
|---|---|
| 127389__acclivity__thetimehascome.wav | acclivity |
| 156550__acclivity__a-dream-within-a-dream.wav | acclivity |
| 34210__acclivity__i-am-female.wav | acclivity |
| 165187__blaukreuz__global-village-hochdeutsch.wav | blaukreuz |
| 167554__speedenza__memory-eva-gore-booth.wav | Speedenza |
| 352762__kennysvoice__audiokingsz-illusion.wav | kennysvoice |
| 382326__scott-simpson__crossing-the-bar.wav | Scott Simpson |
| 72001__corsica-s__electric-masquerade.wav | Corsica_S |
| 75064__corsica-s__farah-faucet.wav | Corsica_S |

Original audio copyright (c) their respective authors, distributed under Creative
Commons licenses via freesound.org. Editing/concatenation of the originals into the
`audio_quality_testing_samples` repo by Stanislav Sinyagin. Further re-slicing to
short segments and resampling for this project by this pipeline
(`data/raw/speech_real/` generation, documented here).

## Noise (`data/raw/noise_real/{stationary,nonstationary,transient}/`)

Source repository: https://github.com/ALEX11BR/AltSFX (an OpenTTD sound-effect set,
itself compiled from freesound.org / soundbible.com / bigsoundbank.com sources).
Files here are resampled to 16 kHz mono and peak-normalized from the originals; only
the vehicle/engine/impact-relevant subset was selected and reorganized into
stationary / non-stationary / transient category folders matching this project's
noise taxonomy.

| File | Category | Original sound | Original author | License |
|---|---|---|---|---|
| industrial_fan.wav | stationary | "Industrial/Factory Fans" | IanStarGem (freesound.org) | CC0 |
| ac_rumble.wav | stationary | "AMBLM air conditioner rumble loop" | LudwigMueller (freesound.org) | CC0 |
| truck_engine_a.wav | stationary | "Truck engine" | Joseph SARDIN (bigsoundbank.com) | CC0 |
| truck_engine_b.wav | stationary | "Truck engine" (+ "school bus truck horn honk 50m away") | Joseph SARDIN; kyles (freesound.org) | CC0 |
| truck_engine_c.wav | stationary | "Truck engine" | Joseph SARDIN (bigsoundbank.com) | CC0 |
| helicopter.wav | nonstationary | "Helicopter Sound" | navaneetha kris (soundbible.com) | CC-BY 3.0 |
| aircraft_landing.wav | nonstationary | "LAX Airport Landing Sound One" | FreeToUseSounds (freesound.org) | CC-BY 3.0 |
| chainsaw.wav | nonstationary | "Chainsaw cutting firewood" | VlatkoBlazek (freesound.org) | CC-BY 3.0 |
| explosion.wav | transient | "Explosion_17" | tcpp (freesound.org) | CC-BY 3.0 |
| jackhammer.wav | transient | "Jackhammer" | MrAuralization (freesound.org) | CC-BY 3.0 |

## Why this is still not a substitute for a real defence-noise corpus

These are generic recordings (a civilian helicopter clip, a construction jackhammer,
a firework/demolition-style explosion, ordinary truck engines) -- useful stand-ins for
proving the pipeline handles *real* non-synthetic acoustic complexity in each of the
PS's required noise categories (stationary / non-stationary / transient), but they are
not gunfire, artillery, or military-vehicle recordings, and there are only 10 clips
covering a handful of sources. For a genuine defence-scenario result you still need
the licensed/restricted corpora listed in `DATASETS.md`. Say so explicitly in any
demo that uses this bundled data -- do not present it as validated defence-noise
performance.
