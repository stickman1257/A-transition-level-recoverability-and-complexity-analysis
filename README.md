# A Transition-Level Recoverability and Complexity Analysis

Code and result summaries for the manuscript:

**Evaluating EMG preprocessing as input representation design for intelligent EMG systems: A transition-level recoverability and complexity analysis**

The repository evaluates conventional surface EMG preprocessing as a sequence of representation-changing transitions. The main analyses estimate:

- **Recoverability**: reverse reconstruction of a previous preprocessing stage from a later stage.
- **Complexity change**: permutation entropy and sample entropy changes across preprocessing transitions.
- **Sensitivity and controls**: model-architecture comparison, shuffled-pair negative control, and LP cutoff sensitivity.

## Repository Contents

```text
src/                         Analysis and model code
run_experiment.sh            Convenience runner for common experiments
requirements.txt             Python dependencies
results/
  recoverability/            Final recoverability summaries and figures
  complexity/                Final complexity summaries and sanity check output
  predictability_horizon/    Auxiliary predictability-horizon summaries
data/
  README.md                  Data placement and availability notes
```

Raw participant EMG files, derived stage parquet files, trained checkpoints, local logs, manuscript drafts, and Word/PDF files are intentionally excluded.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your system if GPU training is required.

## Data Layout

Place preprocessed stage files under `data/interim/`:

```text
data/interim/
  bandpass_df.parquet
  notch_df.parquet
  rectified_df.parquet
  lp_10hz_df.parquet
  lp_6hz_df.parquet       # optional, LP cutoff sensitivity
```

Each parquet file should contain `GROUP_ID`, `TIME`, and matching EMG channel columns across stages. See [data/README.md](data/README.md) for details.

## Reproduce Main Analyses

Recoverability suite:

```bash
./run_experiment.sh recoverability \
  --stage-pairs 'notch->bandpass' 'rectified->notch' 'lp_10hz->rectified' \
  --model-type conditional_unet \
  --epochs 50
```

Model comparison:

```bash
python src/run_recoverability_model_compare.py \
  --stage-dir data/interim \
  --output-dir results/recoverability_model_compare \
  --model-types conditional_unet cnn_1d lstm gru \
  --stage-pairs 'notch->bandpass' 'rectified->notch' 'lp_10hz->rectified' \
  --epochs 50 \
  --device cuda
```

Complexity transition analysis:

```bash
python src/run_complexity_stagepair_suite.py \
  --stage-dir data/interim \
  --stage-pairs 'notch->bandpass' 'rectified->notch' 'lp_10hz->rectified' \
  --window-size 1024 \
  --stride 512 \
  --output-dir results/complexity
```

Negative control:

```bash
python src/run_negative_control.py \
  --suite-dir results/recoverability_suite \
  --stage-dir data/interim \
  --stage-pairs 'notch->bandpass' 'rectified->notch' 'lp_10hz->rectified' \
  --output-dir results/negative_control \
  --folds 0 1 2 3 4 5 6 \
  --device cuda
```

## Included Result Summaries

The committed `results/` directory contains lightweight tables, JSON summaries, and manuscript-facing figures. Large fold-level checkpoints (`*.pt`) are not included.

Key files:

- [results/recoverability/comparison_summary.md](results/recoverability/comparison_summary.md)
- [results/recoverability/integrated_stagepair_summary.csv](results/recoverability/integrated_stagepair_summary.csv)
- [results/complexity/pair_summary.csv](results/complexity/pair_summary.csv)
- [results/predictability_horizon/comparison_summary.md](results/predictability_horizon/comparison_summary.md)

## Citation

If you use this code, please cite the associated manuscript. DOI and journal metadata can be added here after publication.
