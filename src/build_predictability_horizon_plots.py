from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 20,
    }
)


STAGE_COLORS = {
    "raw": "#4d4d4d",
    "notch": "#1f77b4",
    "bandpass": "#2ca02c",
    "rectified": "#e67e22",
    "lp_10hz": "#c0392b",
}

STAGE_LABELS = {
    "raw": "Raw",
    "notch": "Notch",
    "bandpass": "Bandpass",
    "rectified": "Rectified",
    "lp_10hz": "LP 10Hz",
}

RECOVERABILITY_STAGE_MAP = {
    "notch": "notch->bandpass",
    "rectified": "rectified->notch",
    "lp_10hz": "lp_10hz->rectified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plots for the predictability horizon suite.")
    parser.add_argument("--comparison-json", required=True, help="comparison_summary.json from the horizon suite.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated plots.")
    parser.add_argument(
        "--recoverability-json",
        default="data/recoverability_comparison_final/recoverability_unified_gru_summary.json",
        help="Optional recoverability unified summary used for the rank-comparison plot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _rows_by_stage(comparison_payload: dict) -> dict[str, list[dict]]:
    rows_by_stage = {stage: [] for stage in comparison_payload["input_stages"]}
    for horizon_result in comparison_payload["horizon_rankings"]:
        for row in horizon_result["rows"]:
            rows_by_stage[row["input_stage"]].append(row)
    for rows in rows_by_stage.values():
        rows.sort(key=lambda item: float(item["horizon_ms"]))
    return rows_by_stage


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_metric_line_plot(comparison_payload: dict, output_dir: Path) -> Path:
    rows_by_stage = _rows_by_stage(comparison_payload)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10.0), sharex=True)
    fig.patch.set_facecolor("white")

    metric_specs = [
        ("mse", "Forecast MSE", False),
        ("ccc", "Forecast CCC", True),
    ]
    for ax, (metric_key, ylabel, higher_is_better) in zip(axes, metric_specs):
        for stage_name, rows in rows_by_stage.items():
            horizons = [float(row["horizon_ms"]) for row in rows]
            values = [float(row[metric_key]) for row in rows]
            ax.plot(
                horizons,
                values,
                marker="o",
                linewidth=2.2,
                markersize=7,
                color=STAGE_COLORS.get(stage_name, "#555555"),
                label=STAGE_LABELS.get(stage_name, stage_name),
            )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        direction = "higher is better" if higher_is_better else "lower is better"
        ax.text(
            0.98,
            1.02,
            direction,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#666666",
        )

    axes[-1].set_xlabel("Horizon (ms)")
    axes[0].legend(ncol=3, frameon=False, loc="upper left")
    fig.suptitle(
        f"Predictability Across Horizons ({comparison_payload['model_type']})",
        weight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    output_path = output_dir / "predictability_horizon_lines.png"
    _save(fig, output_path)
    return output_path


def build_cnp_plot(comparison_payload: dict, output_dir: Path) -> Path:
    rows_by_stage = _rows_by_stage(comparison_payload)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fig.patch.set_facecolor("white")

    for stage_name, rows in rows_by_stage.items():
        horizons = [float(row["horizon_ms"]) for row in rows]
        cnp_values = [float(row["cnp"]) for row in rows]
        ax.plot(
            horizons,
            cnp_values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=STAGE_COLORS.get(stage_name, "#555555"),
            label=STAGE_LABELS.get(stage_name, stage_name),
        )

    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Horizon (ms)")
    ax.set_ylabel("CNP")
    ax.set_title("Complexity-Normalized Predictability", loc="left", weight="bold")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    plt.tight_layout()

    output_path = output_dir / "predictability_cnp_scores.png"
    _save(fig, output_path)
    return output_path


def _recoverability_ranks(recoverability_payload: dict) -> dict[str, int]:
    rows = []
    for pair_result in recoverability_payload.get("pair_results", []):
        ccc = float(pair_result["selected_model_analysis"]["suite_summary"]["fold_summary_mean"]["ccc"])
        rows.append({"stage_pair": pair_result["stage_pair"], "ccc": ccc})
    rows.sort(key=lambda item: item["ccc"], reverse=True)
    return {row["stage_pair"]: rank for rank, row in enumerate(rows, start=1)}


def build_rank_comparison_plot(
    comparison_payload: dict,
    output_dir: Path,
    recoverability_json: Path,
) -> Path | None:
    if not recoverability_json.exists():
        return None

    recoverability_payload = load_json(recoverability_json)
    recoverability_ranks = _recoverability_ranks(recoverability_payload)
    if not recoverability_ranks:
        return None

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    fig.patch.set_facecolor("white")
    plotted = False

    for horizon_result in comparison_payload["horizon_rankings"]:
        horizon_ms = int(round(horizon_result["horizon_ms"]))
        for row in horizon_result["rows"]:
            input_stage = row["input_stage"]
            stage_pair = RECOVERABILITY_STAGE_MAP.get(input_stage)
            if stage_pair is None or stage_pair not in recoverability_ranks:
                continue

            plotted = True
            x = recoverability_ranks[stage_pair]
            y = int(row["rank_by_cnp"])
            ax.scatter(
                x,
                y,
                s=88,
                color=STAGE_COLORS.get(input_stage, "#555555"),
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                x + 0.05,
                y + 0.05,
                f"{STAGE_LABELS.get(input_stage, input_stage)} {horizon_ms}ms",
                fontsize=9,
                color="#333333",
            )

    if not plotted:
        plt.close(fig)
        return None

    max_rank = max(
        [len(comparison_payload["input_stages"])]
        + list(recoverability_ranks.values())
    )
    ticks = np.arange(1, max_rank + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.8, max_rank + 0.4)
    ax.set_ylim(max_rank + 0.4, 0.8)
    ax.set_xlabel("Recoverability rank (CCC)")
    ax.set_ylabel("Predictability rank (CNP)")
    ax.set_title("Recoverability vs Predictability Rank", loc="left", weight="bold")
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output_path = output_dir / "recoverability_vs_predictability_rank.png"
    _save(fig, output_path)
    return output_path


def build_plots(
    *,
    comparison_json: Path,
    output_dir: Path,
    recoverability_json: Path | None = None,
) -> list[Path]:
    comparison_payload = load_json(comparison_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = [
        build_metric_line_plot(comparison_payload, output_dir),
        build_cnp_plot(comparison_payload, output_dir),
    ]
    if recoverability_json is not None:
        rank_plot = build_rank_comparison_plot(comparison_payload, output_dir, recoverability_json)
        if rank_plot is not None:
            generated_paths.append(rank_plot)

    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps({"plots": [str(path.resolve()) for path in generated_paths]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return generated_paths


def main() -> None:
    args = parse_args()
    generated = build_plots(
        comparison_json=Path(args.comparison_json),
        output_dir=Path(args.output_dir),
        recoverability_json=Path(args.recoverability_json),
    )
    for path in generated:
        print(f"Saved plot: {path.resolve()}")


if __name__ == "__main__":
    main()
