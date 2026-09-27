# Data directory

Raw audio is intentionally not committed.

Expected layout:

data/raw/
  speech/
  stationary/
  nonstationary/
  transient/

data/processed/
  train/
  val/
  test/

`prepare_dataset.py` creates speaker/recording-disjoint manifests. Noise categories are metadata labels; they are not claims that the recordings are defence-specific.
