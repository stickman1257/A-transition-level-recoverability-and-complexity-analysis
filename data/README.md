# Data Notes

This repository does not include raw participant EMG recordings or derived parquet files.

To run the analyses, place stage-level parquet files in `data/interim/`:

```text
data/interim/
  bandpass_df.parquet
  notch_df.parquet
  rectified_df.parquet
  lp_10hz_df.parquet
  lp_6hz_df.parquet
```

Required columns:

- `GROUP_ID`: participant/trial or segment identifier.
- `TIME`: sample time or sample index.
- EMG channel columns: identical channel names across all stage files.

The code infers EMG feature columns by excluding metadata columns. Each stage should contain the same groups, time samples, and channel set for paired transition analyses.

Raw and derived data availability should be handled according to the manuscript's ethics approval, consent language, and journal policy.
