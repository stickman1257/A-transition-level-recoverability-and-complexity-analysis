from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from run_next_window import PREDICTOR_MODEL_TYPES, normalize_predictor_model_type
from run_predictability_suite import DEFAULT_INPUT_STAGES, run_predictability_suite
from utils import normalize_stage_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictability comparison across multiple predictor backbones.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument(
        "--input-stages",
        nargs="*",
        default=list(DEFAULT_INPUT_STAGES),
        help="Input stages to compare for each model.",
    )
    parser.add_argument("--target-stage", default="lp_10hz")
    parser.add_argument("--output-dir", required=True, help="Base directory for model comparison outputs")
    parser.add_argument(
        "--model-types",
        nargs="*",
        default=["lstm", "gru", "cnn_1d", "conditional_unet"],
        help=f"Models to compare. Available: {', '.join(PREDICTOR_MODEL_TYPES)}",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--window-ms", type=float, default=250.0)
    parser.add_argument("--input-window-ms", type=float)
    parser.add_argument("--pred-window-ms", type=float)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--input-seq-len", type=int)
    parser.add_argument("--pred-seq-len", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip training when per-fold checkpoints already exist.",
    )
    parser.add_argument(
        "--save-window-summaries",
        action="store_true",
        help="Include per-window summaries in held-out metrics JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_markdown_summary(comparison_payload: dict) -> str:
    lines = ["# Predictability Model Comparison", ""]
    lines.append(f"Compared models: {', '.join(comparison_payload['model_types'])}")
    lines.append(f"Target stage: {comparison_payload['target_stage']}")
    lines.append("")

    for stage_result in comparison_payload["input_stage_comparisons"]:
        lines.append(f"## {stage_result['input_stage']}")
        lines.append("")
        lines.append(f"Best model by CCC: `{stage_result['best_model_by_ccc']}`")
        lines.append("")
        lines.append("| model | ccc | pearson_r | nrmse | mae |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in stage_result["rows"]:
            lines.append(
                f"| {row['model_type']} | {row['ccc']:.3f} | {row['pearson_r']:.3f} | "
                f"{row['nrmse']:.3f} | {row['mae']:.3f} |"
            )
        lines.append("")
        if stage_result.get("naive_baselines"):
            lines.append("Naive baselines")
            lines.append("")
            lines.append("| baseline | ccc | pearson_r | nrmse | mae |")
            lines.append("|---|---:|---:|---:|---:|")
            for row in stage_result["naive_baselines"]:
                lines.append(
                    f"| {row['baseline_name']} | {row['ccc']:.3f} | {row['pearson_r']:.3f} | "
                    f"{row['nrmse']:.3f} | {row['mae']:.3f} |"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.model_types = [normalize_predictor_model_type(model_type) for model_type in args.model_types]
    normalized_input_stages = [normalize_stage_name(stage_name) for stage_name in args.input_stages]

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths_by_model: dict[str, dict[str, str]] = {}
    for model_type in args.model_types:
        model_args = deepcopy(args)
        model_args.model_type = model_type
        model_args.output_dir = str(base_output_dir / model_type)
        print(f"\n===== Running comparison model: {model_type} =====")
        summary_paths_by_model[model_type] = run_predictability_suite(model_args)

    input_stage_comparisons: list[dict] = []
    for input_stage in normalized_input_stages:
        rows: list[dict] = []
        naive_baseline_rows: list[dict] = []
        for model_type in args.model_types:
            summary_path = Path(summary_paths_by_model[model_type][input_stage])
            payload = load_json(summary_path)
            summary = payload["fold_summary_mean"]
            rows.append(
                {
                    "model_type": model_type,
                    "ccc": float(summary["ccc"]),
                    "pearson_r": float(summary["pearson_r"]),
                    "nrmse": float(summary["nrmse"]),
                    "mae": float(summary["mae"]),
                    "summary_path": str(summary_path.resolve()),
                }
            )
            if not naive_baseline_rows and "naive_baselines" in payload:
                for baseline_name, baseline_payload in payload["naive_baselines"].items():
                    baseline_summary = baseline_payload["fold_summary_mean"]
                    naive_baseline_rows.append(
                        {
                            "baseline_name": baseline_name,
                            "ccc": float(baseline_summary["ccc"]),
                            "pearson_r": float(baseline_summary["pearson_r"]),
                            "nrmse": float(baseline_summary["nrmse"]),
                            "mae": float(baseline_summary["mae"]),
                        }
                    )

        rows.sort(key=lambda item: item["ccc"], reverse=True)
        naive_baseline_rows.sort(key=lambda item: item["ccc"], reverse=True)
        input_stage_comparisons.append(
            {
                "input_stage": input_stage,
                "best_model_by_ccc": rows[0]["model_type"],
                "rows": rows,
                "naive_baselines": naive_baseline_rows,
            }
        )

    comparison_payload = {
        "model_types": args.model_types,
        "input_stages": normalized_input_stages,
        "target_stage": args.target_stage,
        "input_stage_comparisons": input_stage_comparisons,
    }
    json_path = base_output_dir / "comparison_summary.json"
    md_path = base_output_dir / "comparison_summary.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(comparison_payload, file, ensure_ascii=False, indent=2)
    md_path.write_text(build_markdown_summary(comparison_payload), encoding="utf-8")

    print(f"\nSaved comparison JSON to: {json_path.resolve()}")
    print(f"Saved comparison markdown to: {md_path.resolve()}")


if __name__ == "__main__":
    main()
