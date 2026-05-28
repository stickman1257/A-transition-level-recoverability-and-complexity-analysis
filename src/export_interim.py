from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import (
    AVAILABLE_STAGE_NAMES,
    build_emg_stage_frames,
    infer_feature_columns,
    load_raw_emg_directory,
    load_table,
    save_table,
    validate_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stage-wise EMG interim dataframes.")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument("--input-table", help="Preprocessed table file such as data/df.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to save interim dataframe files")
    parser.add_argument(
        "--extra-lowpass-cutoffs",
        type=float,
        nargs="*",
        default=(),
        help="Additional lowpass envelope cutoffs to export, e.g. 20 30",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    using_raw_dir = bool(args.raw_dir)
    using_input_table = bool(args.input_table)
    if using_raw_dir == using_input_table:
        raise ValueError("Provide exactly one of --raw-dir or --input-table.")

    if using_raw_dir:
        raw_df, feature_columns, sampling_frequency = load_raw_emg_directory(args.raw_dir)
    else:
        raw_df = load_table(args.input_table)
        validate_dataframe(raw_df)
        feature_columns = infer_feature_columns(raw_df)
        if not feature_columns:
            raise ValueError("No EMG feature columns were inferred from the input table.")
        raw_df = raw_df[["GROUP_ID", "TIME", *feature_columns]].copy()
        dt = (
            raw_df.groupby("GROUP_ID")["TIME"]
            .apply(lambda s: np.median(np.diff(s.to_numpy())))
            .replace(0, np.nan)
            .dropna()
        )
        if dt.empty:
            raise ValueError("Could not estimate sampling frequency from TIME column.")
        sampling_frequency = 1.0 / float(dt.median())

    stage_frames = build_emg_stage_frames(raw_df, feature_columns, sampling_frequency)

    output_dir = Path(args.output_dir)
    for stage_name in AVAILABLE_STAGE_NAMES:
        save_table(output_dir / f"{stage_name}_df.parquet", stage_frames[stage_name])

    rectified_df = stage_frames["rectified"]
    processor = None
    if args.extra_lowpass_cutoffs:
        from utils import EMGSignalProcessor

        processor = EMGSignalProcessor(fs=sampling_frequency)

    for cutoff in args.extra_lowpass_cutoffs:
        if cutoff <= 0:
            raise ValueError(f"Lowpass cutoff must be positive, got {cutoff}.")
        cutoff_label = f"{int(cutoff)}" if float(cutoff).is_integer() else str(cutoff).replace(".", "_")
        extra_df = processor.lowpass_envelope(rectified_df, feature_columns, cutoff_hz=float(cutoff))
        save_table(output_dir / f"lp_{cutoff_label}hz_df.parquet", extra_df)

    print(f"Saved interim dataframes to: {output_dir.resolve()}")
    print(f"Rows: {len(raw_df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Sampling frequency: {sampling_frequency}")


if __name__ == "__main__":
    main()
