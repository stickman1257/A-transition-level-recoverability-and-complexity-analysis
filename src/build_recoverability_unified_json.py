from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


PAIR_CONFIG = {
    "notch->bandpass": {
        "key": "notch-bp",
        "title": "Notch -> Bandpass",
        "dir_name": "notch_to_bandpass",
    },
    "rectified->notch": {
        "key": "rect-notch",
        "title": "Rectified -> Notch",
        "dir_name": "rectified_to_notch",
    },
    "lp_10hz->rectified": {
        "key": "lp-rect",
        "title": "Low-pass -> Rectified",
        "dir_name": "lp_10hz_to_rectified",
    },
}

MODEL_ALIASES = {
    "vanilla_unet": "u-net",
    "conditional_unet": "conditional-u-net",
    "cnn_1d": "cnn",
    "gru": "gru",
    "lstm": "lstm",
}

MUSCLE_GROUPS = {
    "thigh_anterior_knee_extensors": {
        "label": "Thigh anterior / Knee extensors",
        "description": "Knee extension and early-stance stabilization.",
        "muscles": ["Rectus femoris", "VMO"],
        "channels": [
            "LT_RECTUS_FEM",
            "LT_VMO",
            "RT_RECTUS_FEM",
            "RT_VMO",
        ],
    },
    "thigh_posterior_hamstrings": {
        "label": "Thigh posterior / Hamstrings",
        "description": "Knee flexion, late-swing braking, and hip extension.",
        "muscles": ["Biceps femoris", "Semitendinosus"],
        "channels": [
            "LT_BICEPS_FEM",
            "LT_SEMITEND",
            "RT_BICEPS_FEM",
            "RT_SEMITEND",
        ],
    },
    "ankle_dorsiflexors": {
        "label": "Ankle dorsiflexors",
        "description": "Toe clearance during swing.",
        "muscles": ["Tibialis anterior"],
        "channels": [
            "LT_TIBANT",
            "RT_TIBANT",
        ],
    },
    "ankle_plantarflexors": {
        "label": "Ankle plantarflexors",
        "description": "Push-off and late-stance propulsion.",
        "muscles": ["Medial gastrocnemius"],
        "channels": [
            "LT_MED_GASTRO",
            "RT_MED_GASTRO",
        ],
    },
    "hip_abductors_pelvic_stabilizers": {
        "label": "Hip abductors / Pelvic stabilizers",
        "description": "Pelvic stability during mid-stance.",
        "muscles": ["Gluteus medius"],
        "channels": [
            "LT_GLUT_MED",
            "RT_GLUT_MED",
        ],
    },
    "hip_adductors_medial_stabilizers": {
        "label": "Hip adductors / Medial stabilizers",
        "description": "Medial-chain alignment and stabilization.",
        "muscles": ["Adductors"],
        "channels": [
            "LT_ADDUCTORS",
            "RT_ADDUCTORS",
        ],
    },
}

PRIMARY_METRICS = ("ccc", "nrmse", "pearson_r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified recoverability summary JSON from existing outputs.")
    parser.add_argument(
        "--comparison-json",
        default="data/recoverability_comparison_final/comparison_summary.json",
        help="Combined model comparison summary JSON.",
    )
    parser.add_argument(
        "--selected-model",
        default="gru",
        choices=tuple(MODEL_ALIASES),
        help="Reference model for stability and muscle-group analyses.",
    )
    parser.add_argument(
        "--output-path",
        default="data/recoverability_comparison_final/recoverability_unified_gru_summary.json",
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


def model_label(model_type: str) -> str:
    return MODEL_ALIASES.get(model_type, model_type)


def aggregate_metric_dicts(metric_dicts: list[dict]) -> dict[str, dict[str, float]]:
    metric_names = sorted({key for item in metric_dicts for key in item})
    mean_dict: dict[str, float] = {}
    std_dict: dict[str, float] = {}
    for metric_name in metric_names:
        values = [
            float(item[metric_name])
            for item in metric_dicts
            if metric_name in item and math.isfinite(float(item[metric_name]))
        ]
        if not values:
            continue
        mean_dict[metric_name] = mean(values)
        std_dict[metric_name] = 0.0 if len(values) == 1 else pstdev(values)
    return {"mean": mean_dict, "std": std_dict}


def restrict_metrics(metric_dict: dict) -> dict:
    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in metric_dict.items()
        if metric_name in PRIMARY_METRICS
    }


def aggregate_group_channel_metrics(group_payload: dict, channels: list[str], payload_key: str) -> dict:
    channel_metrics = [
        group_payload[payload_key][channel]
        for channel in channels
        if channel in group_payload[payload_key]
    ]
    return aggregate_metric_dicts(channel_metrics)["mean"]


def build_session_records(fold_payload_path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    fold_payload = load_json(fold_payload_path)
    heldout_metrics = fold_payload["heldout_metrics"]
    session_records: list[dict] = []
    muscle_group_records: dict[str, list[dict]] = {group_key: [] for group_key in MUSCLE_GROUPS}

    for group in heldout_metrics["groups"]:
        session_record = {
            "fold": int(fold_payload["fold"]),
            "test_subject": str(fold_payload["test_subject"]),
            "group_id": str(group["group_id"]),
            "num_windows": int(group["num_windows"]),
            "window_stride": int(group["window_stride"]),
            "summary_mean": group["summary_mean"],
            "summary_std": group["summary_std"],
        }
        session_records.append(session_record)

        for group_key, group_config in MUSCLE_GROUPS.items():
            muscle_group_records[group_key].append(
                {
                    "fold": int(fold_payload["fold"]),
                    "test_subject": str(fold_payload["test_subject"]),
                    "group_id": str(group["group_id"]),
                    "num_windows": int(group["num_windows"]),
                    "summary_mean": aggregate_group_channel_metrics(group, group_config["channels"], "per_channel_mean"),
                    "summary_std": aggregate_group_channel_metrics(group, group_config["channels"], "per_channel_std"),
                }
            )

    return session_records, muscle_group_records


def build_session_analysis(session_records: list[dict]) -> dict:
    summary_means = [record["summary_mean"] for record in session_records]
    summary_stds = [record["summary_std"] for record in session_records]
    ranked_by_ccc = sorted(session_records, key=lambda item: float(item["summary_mean"]["ccc"]), reverse=True)

    def compact_session(record: dict) -> dict:
        return {
            "group_id": record["group_id"],
            "fold": record["fold"],
            "test_subject": record["test_subject"],
            "num_windows": record["num_windows"],
            "summary_mean": restrict_metrics(record["summary_mean"]),
            "summary_std": restrict_metrics(record["summary_std"]),
        }

    return {
        "num_sessions": len(session_records),
        "between_session_mean": aggregate_metric_dicts(summary_means)["mean"],
        "between_session_std": aggregate_metric_dicts(summary_means)["std"],
        "within_session_mean_std": aggregate_metric_dicts(summary_stds)["mean"],
        "within_session_std_std": aggregate_metric_dicts(summary_stds)["std"],
        "top_sessions_by_ccc": [compact_session(item) for item in ranked_by_ccc[:5]],
        "bottom_sessions_by_ccc": [compact_session(item) for item in ranked_by_ccc[-5:]],
        "session_records": [compact_session(item) for item in session_records],
    }


def build_muscle_group_summary(suite_summary: dict, muscle_group_records: dict[str, list[dict]]) -> dict:
    output: dict[str, dict] = {}
    for group_key, group_config in MUSCLE_GROUPS.items():
        channels = group_config["channels"]
        suite_per_channel_mean = [
            suite_summary["per_channel_mean"][channel]
            for channel in channels
            if channel in suite_summary["per_channel_mean"]
        ]
        suite_per_channel_std = [
            suite_summary["per_channel_std"][channel]
            for channel in channels
            if channel in suite_summary["per_channel_std"]
        ]
        session_records = muscle_group_records[group_key]
        output[group_key] = {
            "label": group_config["label"],
            "description": group_config["description"],
            "muscles": group_config["muscles"],
            "channels": channels,
            "channel_count": len(channels),
            "suite_level_mean_across_channels": aggregate_metric_dicts(suite_per_channel_mean)["mean"],
            "suite_level_std_across_channels": aggregate_metric_dicts(suite_per_channel_mean)["std"],
            "suite_level_mean_of_channel_stds": aggregate_metric_dicts(suite_per_channel_std)["mean"],
            "session_level_analysis": build_session_analysis(session_records),
        }
    return output


def selected_model_root(selected_model: str) -> Path:
    if selected_model in {"conditional_unet", "vanilla_unet"}:
        return Path("data/recoverability_suite")
    return Path("data/recoverability_compare_models") / selected_model


def build_pair_payload(pair_result: dict, selected_model: str) -> dict:
    pair_info = PAIR_CONFIG[pair_result["stage_pair"]]
    model_rows = []
    selected_rank = None
    for rank, row in enumerate(pair_result["rows"], start=1):
        model_name = model_label(row["model_type"])
        if row["model_type"] == selected_model:
            selected_rank = rank
        model_rows.append(
            {
                "rank_by_ccc": rank,
                "model": model_name,
                "metrics": {
                    "ccc": float(row["ccc"]),
                    "nrmse": float(row["nrmse"]),
                    "pearson_r": float(row["pearson_r"]),
                    "psd_distance": float(row["psd_distance"]),
                    "rms_relative_error": float(row["rms_relative_error"]),
                    "median_frequency_relative_error": float(row["median_frequency_relative_error"]),
                },
                "summary_path": row["summary_path"],
            }
        )

    selected_root = selected_model_root(selected_model)
    pair_dir = selected_root / pair_info["dir_name"]
    suite_summary = load_json(pair_dir / "suite_summary.json")

    session_records: list[dict] = []
    muscle_group_records: dict[str, list[dict]] = {group_key: [] for group_key in MUSCLE_GROUPS}
    for fold_idx in range(int(suite_summary["num_folds"])):
        fold_path = pair_dir / f"fold_{fold_idx}" / "heldout_subject_metrics.json"
        fold_session_records, fold_group_records = build_session_records(fold_path)
        session_records.extend(fold_session_records)
        for group_key in MUSCLE_GROUPS:
            muscle_group_records[group_key].extend(fold_group_records[group_key])

    return {
        "pair_key": pair_info["key"],
        "pair_title": pair_info["title"],
        "stage_pair": pair_result["stage_pair"],
        "actual_best_model_by_ccc": model_label(pair_result["best_model_by_ccc"]),
        "selected_reference_model": model_label(selected_model),
        "selected_reference_model_rank_by_ccc": selected_rank,
        "all_model_comparison": {
            "metric_priority": "ccc",
            "rows": model_rows,
        },
        "selected_model_analysis": {
            "model": model_label(selected_model),
            "suite_summary": {
                "num_folds": int(suite_summary["num_folds"]),
                "fold_summary_mean": suite_summary["fold_summary_mean"],
                "fold_summary_std": suite_summary["fold_summary_std"],
                "fold_records": suite_summary["folds"],
            },
            "session_analysis": build_session_analysis(session_records),
            "muscle_specific_summary": {
                channel_name: {
                    "mean": metrics,
                    "std": suite_summary["per_channel_std"][channel_name],
                }
                for channel_name, metrics in suite_summary["per_channel_mean"].items()
            },
            "muscle_group_summary": build_muscle_group_summary(suite_summary, muscle_group_records),
        },
    }


def main() -> None:
    args = parse_args()
    comparison_payload = load_json(Path(args.comparison_json))

    pair_payloads = [
        build_pair_payload(pair_result, selected_model=args.selected_model)
        for pair_result in comparison_payload["pair_comparisons"]
    ]

    unified_payload = {
        "analysis_axis": "recoverability-based evaluation of preprocessing",
        "generated_from_existing_results_only": True,
        "excluded_analysis_axes": ["lp_cutoff_sensitivity"],
        "selected_reference_model": model_label(args.selected_model),
        "selected_reference_model_source": "user_selected",
        "model_aliases": {raw_name: model_label(raw_name) for raw_name in comparison_payload["model_types"]},
        "muscle_group_definitions": MUSCLE_GROUPS,
        "pair_results": pair_payloads,
    }

    output_path = Path(args.output_path)
    save_json(output_path, unified_payload)
    print(f"Saved unified summary JSON to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
