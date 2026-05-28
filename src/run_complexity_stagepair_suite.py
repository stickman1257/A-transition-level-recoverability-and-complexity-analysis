from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_metrics import compute_complexity_metrics
from utils import (
    build_emg_stage_frames,
    group_id_to_subject_id,
    infer_feature_columns,
    load_raw_emg_directory,
    load_stage_frame,
    normalize_stage_name,
    resolve_stage_frame,
    save_json,
    validate_stage_name,
)


DEFAULT_STAGE_PAIRS = (
    "notch->bandpass",
    "rectified->notch",
    "lp_10hz->rectified",
)
DEFAULT_MANIFEST_JSON = "data/recoverability_compare_models/gru/suite_manifest.json"
DEFAULT_OUTPUT_DIR = "data/complexity_stagepair_suite"
COMPLEXITY_DIRECTION = "pre_minus_post"
STAGE_ORDER = {
    "raw": 0,
    "dc_offset": 1,
    "bandpass": 2,
    "notch": 3,
    "rectified": 4,
    "lp_6hz": 5,
    "lp_10hz": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pair-wise EMG complexity deltas for recoverability stage pairs.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files.")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders.")
    parser.add_argument(
        "--manifest-json",
        default=DEFAULT_MANIFEST_JSON,
        help="Recoverability suite manifest used to inherit stage pairs and windowing defaults.",
    )
    parser.add_argument(
        "--stage-pairs",
        nargs="*",
        help="Pairs in the form condition->target. Defaults to manifest stage pairs or the project default triplet.",
    )
    parser.add_argument("--window-size", type=int, help="Sliding window size in samples.")
    parser.add_argument("--stride", type=int, help="Sliding window stride in samples.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save complexity CSV outputs.")
    parser.add_argument("--pe-order", type=int, default=3)
    parser.add_argument("--pe-delay", type=int, default=1)
    parser.add_argument("--pe-max-points", type=int, default=3000)
    parser.add_argument("--sampen-order", type=int, default=2)
    parser.add_argument("--sampen-r-ratio", type=float, default=0.2)
    parser.add_argument("--sampen-max-points", type=int, default=256)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_stage_pair(raw_pair: str) -> tuple[str, str]:
    separator = "->" if "->" in raw_pair else ":"
    if separator not in raw_pair:
        raise ValueError(f"Invalid stage pair: {raw_pair}. Use condition->target.")
    condition_stage, target_stage = [normalize_stage_name(part) for part in raw_pair.split(separator, 1)]
    validate_stage_name(condition_stage)
    validate_stage_name(target_stage)
    return condition_stage, target_stage


def resolve_stage_pairs(args: argparse.Namespace, manifest: dict[str, Any] | None) -> list[tuple[str, str]]:
    raw_pairs = args.stage_pairs
    if not raw_pairs and manifest is not None:
        raw_pairs = manifest.get("stage_pairs")
    if not raw_pairs:
        raw_pairs = list(DEFAULT_STAGE_PAIRS)
    return [parse_stage_pair(raw_pair) for raw_pair in raw_pairs]


def resolve_windowing(args: argparse.Namespace, manifest: dict[str, Any] | None) -> tuple[int, int]:
    training_config = manifest.get("training_config", {}) if manifest is not None else {}
    window_size = args.window_size if args.window_size is not None else training_config.get("window_size")
    stride = args.stride if args.stride is not None else training_config.get("stride")
    if window_size is None or stride is None:
        raise ValueError("window_size and stride must be provided via CLI or manifest.")
    return int(window_size), int(stride)


def resolve_data_mode(args: argparse.Namespace, manifest: dict[str, Any] | None) -> tuple[str | None, str | None]:
    manifest_stage_dir = manifest.get("stage_dir") if manifest is not None else None
    manifest_raw_dir = manifest.get("raw_dir") if manifest is not None else None
    stage_dir = args.stage_dir or manifest_stage_dir
    raw_dir = args.raw_dir or manifest_raw_dir
    mode_count = sum([bool(stage_dir), bool(raw_dir)])
    if mode_count != 1:
        raise ValueError("Provide exactly one data source via CLI or manifest: --stage-dir or --raw-dir.")
    return stage_dir, raw_dir


def resolve_complexity_pair(condition_stage: str, target_stage: str) -> tuple[str, str]:
    if condition_stage not in STAGE_ORDER or target_stage not in STAGE_ORDER:
        raise ValueError(
            f"Complexity direction requires known stage ordering, got {condition_stage!r} and {target_stage!r}."
        )
    if STAGE_ORDER[condition_stage] < STAGE_ORDER[target_stage]:
        pre_stage, post_stage = condition_stage, target_stage
    else:
        pre_stage, post_stage = target_stage, condition_stage
    if pre_stage == post_stage:
        raise ValueError(f"Complexity pre/post stages must differ, got {pre_stage}.")
    return pre_stage, post_stage


def direction_note(pre_stage: str, post_stage: str) -> str:
    return f"complexity delta computed as {pre_stage} - {post_stage}"


class StageFrameProvider:
    def __init__(self, *, stage_dir: str | None, raw_dir: str | None) -> None:
        self.stage_dir = stage_dir
        self.raw_dir = raw_dir
        self._stage_cache: dict[str, pd.DataFrame] = {}
        self._stage_frames: dict[str, pd.DataFrame] | None = None
        self._feature_columns: list[str] | None = None
        self._sampling_frequency: float | None = None

        if raw_dir:
            raw_df, feature_columns, sampling_frequency = load_raw_emg_directory(raw_dir)
            self._stage_frames = build_emg_stage_frames(raw_df, feature_columns, sampling_frequency)
            self._feature_columns = list(feature_columns)
            self._sampling_frequency = float(sampling_frequency)

    def get(self, stage_name: str) -> pd.DataFrame:
        stage_name = normalize_stage_name(stage_name)
        if stage_name in self._stage_cache:
            return self._stage_cache[stage_name]

        if self.raw_dir:
            if self._stage_frames is None or self._feature_columns is None or self._sampling_frequency is None:
                raise RuntimeError("Raw stage frames were not initialized correctly.")
            frame = resolve_stage_frame(stage_name, self._stage_frames, self._feature_columns, self._sampling_frequency)
        else:
            if self.stage_dir is None:
                raise RuntimeError("stage_dir is required when raw_dir is not used.")
            frame = load_stage_frame(self.stage_dir, stage_name)

        feature_columns = infer_feature_columns(frame)
        frame = frame[["GROUP_ID", "TIME", *feature_columns]].copy()
        frame = frame.sort_values(["GROUP_ID", "TIME"]).reset_index(drop=True)
        self._stage_cache[stage_name] = frame
        return frame


def resolve_pair_metadata(stage_pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    for condition_stage, target_stage in stage_pairs:
        pre_stage, post_stage = resolve_complexity_pair(condition_stage, target_stage)
        metadata.append(
            {
                "recoverability_pair": f"{condition_stage}->{target_stage}",
                "condition_stage": condition_stage,
                "target_stage": target_stage,
                "pre_stage": pre_stage,
                "post_stage": post_stage,
                "complexity_direction": COMPLEXITY_DIRECTION,
                "direction_note": direction_note(pre_stage, post_stage),
            }
        )
    return metadata


def resolve_feature_columns(stage_frames: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    stage_names = list(stage_frames)
    if not stage_names:
        raise ValueError("At least one stage frame is required.")

    ordered_candidates = infer_feature_columns(stage_frames[stage_names[0]])
    usable_columns: list[str] = []
    dropped_columns: list[str] = []
    for column in ordered_candidates:
        keep = True
        for stage_name in stage_names:
            frame = stage_frames[stage_name]
            if column not in frame.columns:
                keep = False
                break
            values = frame[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                keep = False
                break
        if keep:
            usable_columns.append(column)
        else:
            dropped_columns.append(column)

    if not usable_columns:
        raise ValueError("No shared finite EMG feature columns remain across the requested stages.")
    return usable_columns, dropped_columns


def compute_stage_window_rows(
    *,
    stage_name: str,
    stage_df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    stride: int,
    pe_order: int,
    pe_delay: int,
    pe_max_points: int,
    sampen_order: int,
    sampen_r_ratio: float,
    sampen_max_points: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group_df in stage_df.groupby("GROUP_ID", sort=False):
        if len(group_df) < window_size:
            continue
        subject_id = group_id_to_subject_id(str(group_id))
        for start_idx in range(0, len(group_df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = group_df.iloc[start_idx:end_idx][feature_columns].to_numpy(dtype=np.float32)
            payload = compute_complexity_metrics(
                window,
                channel_names=feature_columns,
                pe_order=pe_order,
                pe_delay=pe_delay,
                pe_max_points=pe_max_points,
                sampen_order=sampen_order,
                sampen_r_ratio=sampen_r_ratio,
                sampen_max_points=sampen_max_points,
            )
            for channel in feature_columns:
                rows.append(
                    {
                        "stage": stage_name,
                        "subject_id": subject_id,
                        "group_id": str(group_id),
                        "channel": channel,
                        "start_idx": int(start_idx),
                        "window_size": int(window_size),
                        "pe": float(payload["per_channel"][channel]["pe"]),
                        "sampen": float(payload["per_channel"][channel]["sampen"]),
                    }
                )
    return pd.DataFrame(rows)


def build_pair_window_df(
    *,
    pair_metadata: list[dict[str, str]],
    stage_window_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pair_frames: list[pd.DataFrame] = []
    join_keys = ["subject_id", "group_id", "channel", "start_idx", "window_size"]
    for meta in pair_metadata:
        pre_df = stage_window_frames[meta["pre_stage"]][join_keys + ["pe", "sampen"]].rename(
            columns={"pe": "pe_pre", "sampen": "sampen_pre"}
        )
        post_df = stage_window_frames[meta["post_stage"]][join_keys + ["pe", "sampen"]].rename(
            columns={"pe": "pe_post", "sampen": "sampen_post"}
        )
        merged = pre_df.merge(post_df, on=join_keys, how="inner", validate="one_to_one")
        if merged.empty:
            raise ValueError(f"No aligned windows were found for pair {meta['recoverability_pair']}.")
        for key, value in meta.items():
            merged[key] = value
        merged["delta_pe"] = merged["pe_pre"] - merged["pe_post"]
        merged["delta_sampen"] = merged["sampen_pre"] - merged["sampen_post"]
        pair_frames.append(merged)

    if not pair_frames:
        raise ValueError("No pair-wise complexity rows were created.")
    return pd.concat(pair_frames, ignore_index=True)


def finite_mean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_std(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr, ddof=0))


def build_pair_summary(window_df: pd.DataFrame, pair_order: list[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for recoverability_pair in pair_order:
        pair_df = window_df[window_df["recoverability_pair"] == recoverability_pair].copy()
        if pair_df.empty:
            continue
        summary_rows.append(
            {
                "recoverability_pair": recoverability_pair,
                "condition_stage": pair_df["condition_stage"].iloc[0],
                "target_stage": pair_df["target_stage"].iloc[0],
                "pre_stage": pair_df["pre_stage"].iloc[0],
                "post_stage": pair_df["post_stage"].iloc[0],
                "complexity_direction": pair_df["complexity_direction"].iloc[0],
                "direction_note": pair_df["direction_note"].iloc[0],
                "pe_pre_mean": finite_mean(pair_df["pe_pre"]),
                "pe_post_mean": finite_mean(pair_df["pe_post"]),
                "delta_pe_mean": finite_mean(pair_df["delta_pe"]),
                "delta_pe_std": finite_std(pair_df["delta_pe"]),
                "sampen_pre_mean": finite_mean(pair_df["sampen_pre"]),
                "sampen_post_mean": finite_mean(pair_df["sampen_post"]),
                "delta_sampen_mean": finite_mean(pair_df["delta_sampen"]),
                "delta_sampen_std": finite_std(pair_df["delta_sampen"]),
                "num_subjects": int(pair_df["subject_id"].nunique()),
                "num_groups": int(pair_df["group_id"].nunique()),
                "num_windows": int(pair_df[["group_id", "start_idx"]].drop_duplicates().shape[0]),
                "num_rows": int(len(pair_df)),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json) if args.manifest_json else None
    manifest = load_json(manifest_path) if manifest_path is not None and manifest_path.exists() else None

    stage_dir, raw_dir = resolve_data_mode(args, manifest)
    stage_pairs = resolve_stage_pairs(args, manifest)
    window_size, stride = resolve_windowing(args, manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_metadata = resolve_pair_metadata(stage_pairs)
    provider = StageFrameProvider(stage_dir=stage_dir, raw_dir=raw_dir)
    needed_stages = sorted({meta["pre_stage"] for meta in pair_metadata} | {meta["post_stage"] for meta in pair_metadata}, key=STAGE_ORDER.get)
    stage_frames = {stage_name: provider.get(stage_name) for stage_name in needed_stages}
    feature_columns, dropped_columns = resolve_feature_columns(stage_frames)
    if dropped_columns:
        print("Dropped non-finite/shared feature columns:", ", ".join(dropped_columns))

    stage_window_frames: dict[str, pd.DataFrame] = {}
    for stage_name in needed_stages:
        print(f"Computing stage-wise complexity for {stage_name}")
        stage_window_frames[stage_name] = compute_stage_window_rows(
            stage_name=stage_name,
            stage_df=stage_frames[stage_name],
            feature_columns=feature_columns,
            window_size=window_size,
            stride=stride,
            pe_order=args.pe_order,
            pe_delay=args.pe_delay,
            pe_max_points=args.pe_max_points,
            sampen_order=args.sampen_order,
            sampen_r_ratio=args.sampen_r_ratio,
            sampen_max_points=args.sampen_max_points,
        )

    pair_window_df = build_pair_window_df(pair_metadata=pair_metadata, stage_window_frames=stage_window_frames)
    pair_order = [meta["recoverability_pair"] for meta in pair_metadata]
    pair_window_df = pair_window_df[
        [
            "recoverability_pair",
            "condition_stage",
            "target_stage",
            "pre_stage",
            "post_stage",
            "complexity_direction",
            "direction_note",
            "subject_id",
            "group_id",
            "channel",
            "start_idx",
            "window_size",
            "pe_pre",
            "pe_post",
            "delta_pe",
            "sampen_pre",
            "sampen_post",
            "delta_sampen",
        ]
    ]
    pair_window_path = output_dir / "pair_window_delta_complexity.csv"
    pair_window_df.to_csv(pair_window_path, index=False)

    pair_summary_df = build_pair_summary(pair_window_df, pair_order)
    pair_summary_path = output_dir / "pair_summary.csv"
    pair_summary_df.to_csv(pair_summary_path, index=False)

    suite_manifest = {
        "mode": "complexity_stagepair",
        "stage_dir": str(Path(stage_dir).resolve()) if stage_dir else None,
        "raw_dir": str(Path(raw_dir).resolve()) if raw_dir else None,
        "source_manifest_json": str(manifest_path.resolve()) if manifest_path is not None and manifest_path.exists() else None,
        "stage_pairs": pair_order,
        "window_size": window_size,
        "stride": stride,
        "complexity_direction": COMPLEXITY_DIRECTION,
        "metrics": {
            "primary": "pe",
            "secondary": "sampen",
            "pe_order": args.pe_order,
            "pe_delay": args.pe_delay,
            "pe_max_points": args.pe_max_points,
            "sampen_order": args.sampen_order,
            "sampen_r_ratio": args.sampen_r_ratio,
            "sampen_max_points": args.sampen_max_points,
        },
        "outputs": {
            "pair_window_delta_complexity_csv": str(pair_window_path.resolve()),
            "pair_summary_csv": str(pair_summary_path.resolve()),
        },
    }
    save_json(output_dir / "suite_manifest.json", suite_manifest)

    print(f"Saved pair window complexity CSV to: {pair_window_path.resolve()}")
    print(f"Saved pair summary CSV to: {pair_summary_path.resolve()}")


if __name__ == "__main__":
    main()
