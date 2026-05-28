from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined recoverability comparison report from existing model outputs.")
    parser.add_argument(
        "--model-summary-dir",
        action="append",
        required=True,
        help="Mapping in the form model_type=summary_root. Example: conditional_unet=data/recoverability_suite",
    )
    parser.add_argument(
        "--stage-pairs",
        nargs="*",
        default=["notch->bandpass", "rectified->notch", "lp_10hz->rectified"],
        help="Stage pairs to include in the report.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save the combined comparison report")
    return parser.parse_args()


def pair_label(stage_pair: str) -> str:
    condition, target = stage_pair.split("->", 1)
    return f"{condition}_to_{target}".replace(".", "_")


def parse_model_summary_dirs(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --model-summary-dir value: {item!r}")
        model_type, path = item.split("=", 1)
        mapping[model_type.strip()] = Path(path).resolve()
    return mapping


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_markdown_summary(payload: dict) -> str:
    lines = ["# Recoverability Comparison Report", ""]
    lines.append(f"Models: {', '.join(payload['model_types'])}")
    lines.append("")

    for pair_result in payload["pair_comparisons"]:
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
    model_summary_dirs = parse_model_summary_dirs(args.model_summary_dir)

    pair_comparisons: list[dict] = []
    for stage_pair in args.stage_pairs:
        rows: list[dict] = []
        for model_type, summary_root in model_summary_dirs.items():
            summary_path = summary_root / pair_label(stage_pair) / "suite_summary.json"
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
                    "summary_path": str(summary_path),
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_types": list(model_summary_dirs),
        "stage_pairs": list(args.stage_pairs),
        "pair_comparisons": pair_comparisons,
    }

    json_path = output_dir / "comparison_summary.json"
    md_path = output_dir / "comparison_summary.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    md_path.write_text(build_markdown_summary(payload), encoding="utf-8")

    print(f"Saved comparison JSON to: {json_path.resolve()}")
    print(f"Saved comparison markdown to: {md_path.resolve()}")


if __name__ == "__main__":
    main()
