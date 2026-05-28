from __future__ import annotations

import argparse
import json
from pathlib import Path

from signal_metrics import aggregate_metric_payloads, compute_coherence_metrics
from utils import (
    align_frames,
    estimate_sampling_frequency,
    infer_feature_columns,
    load_stage_frame,
)


PAIR_DIR_TO_STAGE_PAIR = {
    "notch-bp": ("notch", "bandpass"),
    "rect-notch": ("rectified", "notch"),
    "lp-rect": ("lp_10hz", "rectified"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment unified recoverability JSON with model-free SCP coherence metric.")
    parser.add_argument(
        "--unified-json",
        default="data/recoverability_comparison_final/recoverability_unified_gru_summary.json",
        help="Unified recoverability JSON created from existing model-based outputs.",
    )
    parser.add_argument(
        "--stage-dir",
        default="data/interim",
        help="Directory containing stage parquet files.",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/recoverability_suite/suite_manifest.json",
        help="Recoverability suite manifest used to recover window and stride settings.",
    )
    parser.add_argument(
        "--output-path",
        default="data/recoverability_comparison_final/recoverability_unified_gru_with_scp_summary.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def aggregate_group_channels(group_payload: dict, channels: list[str], payload_key: str) -> dict[str, float]:
    channel_payloads = [
        group_payload[payload_key][channel]
        for channel in channels
        if channel in group_payload[payload_key]
    ]
    aggregate = aggregate_metric_payloads([{"summary": item, "per_channel": {}} for item in channel_payloads])
    return aggregate["summary_mean"]


def evaluate_group_scp(
    cond_df,
    target_df,
    group_id: str,
    feature_columns: list[str],
    window_size: int,
    stride: int,
) -> dict:
    cond_group = cond_df[cond_df["GROUP_ID"] == group_id]
    target_group = target_df[target_df["GROUP_ID"] == group_id]
    fs = estimate_sampling_frequency(target_group)

    window_payloads: list[dict] = []
    for start_idx in range(0, len(target_group) - window_size + 1, stride):
        end_idx = start_idx + window_size
        target_window = target_group.iloc[start_idx:end_idx][feature_columns].to_numpy(dtype=float)
        cond_window = cond_group.iloc[start_idx:end_idx][feature_columns].to_numpy(dtype=float)
        window_payloads.append(
            compute_coherence_metrics(
                target_window,
                cond_window,
                fs=fs,
                channel_names=feature_columns,
            )
        )

    aggregate = aggregate_metric_payloads(window_payloads)
    return {
        "group_id": group_id,
        "num_windows": len(window_payloads),
        "window_stride": stride,
        "summary_mean": aggregate["summary_mean"],
        "summary_std": aggregate["summary_std"],
        "per_channel_mean": aggregate["per_channel_mean"],
        "per_channel_std": aggregate["per_channel_std"],
    }


def build_session_analysis(group_results: list[dict]) -> dict:
    payloads = [
        {
            "summary": group_result["summary_mean"],
            "per_channel": group_result.get("per_channel_mean", {}),
        }
        for group_result in group_results
    ]
    aggregate = aggregate_metric_payloads(payloads)

    summary_std_payloads = [
        {
            "summary": group_result["summary_std"],
            "per_channel": group_result.get("per_channel_std", {}),
        }
        for group_result in group_results
    ]
    aggregate_std = aggregate_metric_payloads(summary_std_payloads)

    ranked = sorted(group_results, key=lambda item: float(item["summary_mean"]["scp_mean_coherence"]), reverse=True)
    def compact(item: dict) -> dict:
        payload = {
            "group_id": item["group_id"],
            "num_windows": item["num_windows"],
            "summary_mean": item["summary_mean"],
            "summary_std": item["summary_std"],
        }
        if "subject" in item:
            payload["subject"] = item["subject"]
        return payload
    return {
        "num_sessions": len(group_results),
        "between_session_mean": aggregate["summary_mean"],
        "between_session_std": aggregate["summary_std"],
        "within_session_mean_std": aggregate_std["summary_mean"],
        "within_session_std_std": aggregate_std["summary_std"],
        "top_sessions_by_scp": [compact(item) for item in ranked[:5]],
        "bottom_sessions_by_scp": [compact(item) for item in ranked[-5:]],
        "session_records": [compact(item) for item in ranked],
    }


def build_muscle_group_summary(group_results: list[dict], muscle_groups: dict) -> dict:
    output: dict[str, dict] = {}
    for group_key, group_config in muscle_groups.items():
        channels = group_config["channels"]

        suite_channel_payloads = [
            {"summary": group_result["per_channel_mean"][channel], "per_channel": {}}
            for group_result in group_results
            for channel in channels
            if channel in group_result["per_channel_mean"]
        ]
        suite_channel_stds = [
            {"summary": group_result["per_channel_std"][channel], "per_channel": {}}
            for group_result in group_results
            for channel in channels
            if channel in group_result["per_channel_std"]
        ]
        channel_aggregate = aggregate_metric_payloads(suite_channel_payloads)
        channel_std_aggregate = aggregate_metric_payloads(suite_channel_stds)

        session_level_records = []
        for group_result in group_results:
            session_level_records.append(
                {
                    "group_id": group_result["group_id"],
                    "subject": str(group_result["group_id"]).rsplit("_", 1)[0],
                    "num_windows": group_result["num_windows"],
                    "summary_mean": aggregate_group_channels(group_result, channels, "per_channel_mean"),
                    "summary_std": aggregate_group_channels(group_result, channels, "per_channel_std"),
                }
            )

        output[group_key] = {
            "label": group_config["label"],
            "description": group_config["description"],
            "muscles": group_config["muscles"],
            "channels": channels,
            "channel_count": len(channels),
            "suite_level_mean_across_channels": channel_aggregate["summary_mean"],
            "suite_level_std_across_channels": channel_aggregate["summary_std"],
            "suite_level_mean_of_channel_stds": channel_std_aggregate["summary_mean"],
            "session_level_analysis": build_session_analysis(session_level_records),
        }
    return output


def build_pair_scp_summary(
    *,
    stage_dir: Path,
    pair_key: str,
    window_size: int,
    stride: int,
    muscle_groups: dict,
) -> dict:
    condition_stage, target_stage = PAIR_DIR_TO_STAGE_PAIR[pair_key]
    target_df = load_stage_frame(stage_dir, target_stage)
    cond_df = load_stage_frame(stage_dir, condition_stage)
    target_df, cond_df, feature_columns = align_frames(target_df, cond_df, feature_columns=infer_feature_columns(target_df))

    group_ids = sorted(target_df["GROUP_ID"].unique().tolist())
    group_results = [
        evaluate_group_scp(
            cond_df=cond_df,
            target_df=target_df,
            group_id=group_id,
            feature_columns=feature_columns,
            window_size=window_size,
            stride=stride,
        )
        for group_id in group_ids
    ]

    pair_payloads = [
        {
            "summary": group_result["summary_mean"],
            "per_channel": group_result["per_channel_mean"],
        }
        for group_result in group_results
    ]
    pair_aggregate = aggregate_metric_payloads(pair_payloads)

    return {
        "metric_key": "scp_mean_coherence",
        "metric_label": "SCP",
        "metric_full_name": "mean magnitude-squared coherence",
        "interpretation": "higher_is_better",
        "rationale": "Model-free auxiliary metric intended to reduce model-capacity confounding in recoverability evaluation.",
        "windowing": {
            "window_size": window_size,
            "stride": stride,
            "nperseg": min(256, window_size),
        },
        "pair_summary": {
            "num_groups": len(group_results),
            "summary_mean": pair_aggregate["summary_mean"],
            "summary_std": pair_aggregate["summary_std"],
            "per_channel_mean": pair_aggregate["per_channel_mean"],
            "per_channel_std": pair_aggregate["per_channel_std"],
        },
        "session_analysis": build_session_analysis(group_results),
        "muscle_specific_summary": {
            channel_name: {
                "mean": metrics,
                "std": pair_aggregate["per_channel_std"][channel_name],
            }
            for channel_name, metrics in pair_aggregate["per_channel_mean"].items()
        },
        "muscle_group_summary": build_muscle_group_summary(group_results, muscle_groups),
    }


def main() -> None:
    args = parse_args()
    unified_path = Path(args.unified_json)
    stage_dir = Path(args.stage_dir)
    manifest = load_json(Path(args.manifest_json))
    unified_payload = load_json(unified_path)

    window_size = int(manifest["training_config"]["window_size"])
    stride = int(manifest["training_config"]["stride"])
    muscle_groups = unified_payload["muscle_group_definitions"]

    for pair_result in unified_payload["pair_results"]:
        pair_key = pair_result["pair_key"]
        pair_result["model_free_auxiliary_metric"] = build_pair_scp_summary(
            stage_dir=stage_dir,
            pair_key=pair_key,
            window_size=window_size,
            stride=stride,
            muscle_groups=muscle_groups,
        )

    unified_payload["model_free_auxiliary_metric_overview"] = {
        "metric_key": "scp_mean_coherence",
        "metric_label": "SCP",
        "metric_full_name": "mean magnitude-squared coherence",
        "interpretation": "higher_is_better",
        "rationale": "Added as a model-free auxiliary metric to mitigate criticism that recoverability scores may be inflated by reconstruction model capacity.",
    }

    output_path = Path(args.output_path)
    save_json(output_path, unified_payload)
    print(f"Saved SCP-augmented unified JSON to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
