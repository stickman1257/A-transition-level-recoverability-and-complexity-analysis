from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from model import AVAILABLE_RECONSTRUCTION_MODELS, normalize_reconstruction_model_type
from run_recoverability_suite import parse_stage_pair, run_recoverability_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recoverability comparison across multiple reconstruction backbones.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument(
        "--stage-pairs",
        nargs="*",
        default=["notch->bandpass", "rectified->notch", "lp_10hz->rectified"],
        help="Pairs in the form condition->target, e.g. notch->bandpass",
    )
    parser.add_argument("--output-dir", required=True, help="Base directory for model comparison outputs")
    parser.add_argument(
        "--model-types",
        nargs="*",
        default=["vanilla_unet", "cnn_1d", "lstm", "gru"],
        help=f"Models to compare. Available: {', '.join(AVAILABLE_RECONSTRUCTION_MODELS)}",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-warmup-epochs", type=int, default=15)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--corr-weight", type=float, default=0.1)
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
    lines = ["# Recoverability Model Comparison", ""]
    lines.append(f"Compared models: {', '.join(comparison_payload['model_types'])}")
    lines.append("")

    for pair_result in comparison_payload["pair_comparisons"]:
        lines.append(f"## {pair_result['stage_pair']}")
        lines.append("")
        lines.append(f"Best model by CCC: `{pair_result['best_model_by_ccc']}`")
        lines.append("")
        lines.append("| model | ccc | pearson_r | nrmse | psd_distance | rms_relative_error | median_frequency_relative_error |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in pair_result["rows"]:
            lines.append(
                f"| {row['model_type']} | {row['ccc']:.3f} | {row['pearson_r']:.3f} | {row['nrmse']:.3f} | "
                f"{row['psd_distance']:.3f} | {row['rms_relative_error']:.3f} | "
                f"{row['median_frequency_relative_error']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.model_types = [normalize_reconstruction_model_type(model_type) for model_type in args.model_types]
    args.stage_pairs = [f"{condition}->{target}" for condition, target in (parse_stage_pair(pair) for pair in args.stage_pairs)]

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths_by_model: dict[str, dict[str, str]] = {}
    for model_type in args.model_types:
        model_args = deepcopy(args)
        model_args.model_type = model_type
        model_args.output_dir = str(base_output_dir / model_type)
        print(f"\n===== Running comparison model: {model_type} =====")
        summary_paths_by_model[model_type] = run_recoverability_suite(model_args)

    pair_comparisons: list[dict] = []
    for stage_pair in args.stage_pairs:
        rows: list[dict] = []
        for model_type in args.model_types:
            summary_path = Path(summary_paths_by_model[model_type][stage_pair])
            payload = load_json(summary_path)
            summary = payload["fold_summary_mean"]
            rows.append(
                {
                    "model_type": model_type,
                    "ccc": float(summary["ccc"]),
                    "pearson_r": float(summary["pearson_r"]),
                    "nrmse": float(summary["nrmse"]),
                    "psd_distance": float(summary["psd_distance"]),
                    "rms_relative_error": float(summary["rms_relative_error"]),
                    "median_frequency_relative_error": float(summary["median_frequency_relative_error"]),
                    "summary_path": str(summary_path.resolve()),
                }
            )

        rows.sort(key=lambda item: item["ccc"], reverse=True)
        pair_comparisons.append(
            {
                "stage_pair": stage_pair,
                "best_model_by_ccc": rows[0]["model_type"],
                "rows": rows,
            }
        )

    comparison_payload = {
        "model_types": args.model_types,
        "stage_pairs": list(args.stage_pairs),
        "pair_comparisons": pair_comparisons,
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
