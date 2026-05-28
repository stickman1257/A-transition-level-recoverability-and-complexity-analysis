from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 22,
    }
)


MODEL_LABELS = {
    "vanilla_unet": "U-Net",
    "conditional_unet": "Conditional U-Net",
    "cnn_1d": "CNN",
    "gru": "GRU",
    "lstm": "LSTM",
}

MODEL_COLORS = {
    "vanilla_unet": "#1f3c88",
    "conditional_unet": "#5f27cd",
    "cnn_1d": "#e67e22",
    "gru": "#0f9d58",
    "lstm": "#c0392b",
}

PAIR_LABELS = {
    "notch->bandpass": "notch-bp",
    "rectified->notch": "rect-notch",
    "lp_10hz->rectified": "lp-rect",
}

METRICS = (
    ("ccc", "CCC", True),
    ("nrmse", "NRMSE", False),
    ("pearson_r", "Pearson r", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model comparison plots for recoverability results.")
    parser.add_argument(
        "--comparison-json",
        default="data/recoverability_comparison_final/comparison_summary.json",
        help="Path to the combined recoverability comparison summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/recoverability_comparison_final/plots",
        help="Directory to save generated plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_pair_rows(pair_result: dict) -> list[dict]:
    rows: list[dict] = []
    for row in pair_result["rows"]:
        summary_path = Path(row["summary_path"])
        payload = load_json(summary_path)
        rows.append(
            {
                "model_type": row["model_type"],
                "display_name": MODEL_LABELS.get(row["model_type"], row["model_type"]),
                "color": MODEL_COLORS.get(row["model_type"], "#4c4c4c"),
                "mean": payload["fold_summary_mean"],
                "std": payload["fold_summary_std"],
            }
        )
    return rows


def ranked_indices(rows: list[dict], metric_key: str, higher_is_better: bool) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda idx: float(rows[idx]["mean"][metric_key]),
        reverse=higher_is_better,
    )


def add_bar_labels(ax: plt.Axes, bars, values: list[float]) -> None:
    # Deprecated for vertical layout; kept for compatibility.
    spread = max(values) - min(values) if values else 0.0
    offset = max(spread * 0.05, max(values) * 0.02 if values else 0.03, 0.012)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )


def add_horizontal_labels(
    ax: plt.Axes,
    means: list[float],
    stds: list[float],
    y_positions: np.ndarray,
    x_min: float,
    x_max: float,
) -> None:
    width = max(x_max - x_min, 1e-6)
    offset = width * 0.03
    right_padding = width * 0.04
    left_padding = width * 0.02
    for mean, std, y_pos in zip(means, stds, y_positions):
        x_pos = mean + std + offset
        if x_pos > x_max - right_padding:
            x_pos = x_max - right_padding
            ha = "right"
        elif x_pos < x_min + left_padding:
            x_pos = x_min + left_padding
            ha = "left"
        else:
            ha = "left"
        ax.text(
            x_pos,
            y_pos,
            f"{mean:.3f} ± {std:.3f}",
            ha=ha,
            va="center",
            fontsize=12,
            weight="bold",
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
        )


def should_zoom_axis(metric_key: str) -> bool:
    return metric_key in {"ccc", "pearson_r"}


def compute_zoom_limits(metric_key: str, means: list[float], stds: list[float]) -> tuple[float, float]:
    xmax = max(mean + std for mean, std in zip(means, stds)) if means else 1.0
    xmin = min(mean - std for mean, std in zip(means, stds)) if means else 0.0
    span = max(xmax - xmin, 1e-6)

    if metric_key in {"ccc", "pearson_r"}:
        pad = max(0.004, span * 0.35)
        x_lower = xmin - pad
        x_upper = xmax + pad
        x_upper = min(1.0, x_upper)
        return x_lower, x_upper

    x_lower = min(-0.01, xmin * 1.15) if xmin < 0 else 0.0
    x_upper = xmax * 1.35 if xmax > 0 else 1.0
    return x_lower, x_upper


def plot_pair(pair_result: dict, output_dir: Path) -> Path:
    stage_pair = pair_result["stage_pair"]
    pair_label = PAIR_LABELS.get(stage_pair, stage_pair)
    rows = build_pair_rows(pair_result)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(15.5, 10.8))
    fig.patch.set_facecolor("white")

    for ax, (metric_key, metric_label, higher_is_better) in zip(axes, METRICS):
        order = ranked_indices(rows, metric_key, higher_is_better)
        ordered_rows = [rows[idx] for idx in order]
        labels = [row["display_name"] for row in ordered_rows]
        means = [float(row["mean"][metric_key]) for row in ordered_rows]
        stds = [float(row["std"][metric_key]) for row in ordered_rows]
        colors = [row["color"] for row in ordered_rows]

        y = np.arange(len(ordered_rows), dtype=float) * 0.68
        x_lower, x_limit = compute_zoom_limits(metric_key, means, stds)
        bar_left = np.full(len(means), x_lower) if should_zoom_axis(metric_key) else None
        bar_width = np.asarray(means) - x_lower if should_zoom_axis(metric_key) else np.asarray(means)

        ax.barh(
            y,
            bar_width,
            left=bar_left,
            xerr=stds,
            color=colors,
            edgecolor="#1a1a1a",
            linewidth=1.0,
            height=0.4,
            error_kw={"elinewidth": 1.3, "capsize": 5, "capthick": 1.3},
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=13, weight="bold")
        ax.invert_yaxis()
        ax.set_title(metric_label, fontsize=17, weight="bold", loc="left", pad=10)
        ax.grid(axis="x", alpha=0.24, linewidth=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(x_lower, x_limit)
        add_horizontal_labels(ax, means, stds, y, x_lower, x_limit)

        direction = "higher is better" if higher_is_better else "lower is better"
        ax.text(
            0.98,
            1.07,
            direction,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color="#555555",
            bbox={"facecolor": "#f3f4f6", "edgecolor": "none", "boxstyle": "round,pad=0.34"},
        )

    fig.suptitle(
        f"Recoverability Model Comparison: {pair_label}",
        fontsize=22,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.02,
        "Bar length shows fold mean. Error bars and labels show mean ± std across 7 subject folds.",
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
    )
    plt.tight_layout(rect=(0.04, 0.055, 0.98, 0.94))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair_label}_model_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_contact_sheet(image_paths: list[Path], output_dir: Path) -> Path:
    images = [plt.imread(path) for path in image_paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(20, 6.6 * len(images)))
    if len(images) == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    for ax, image, path in zip(axes, images, image_paths):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(path.stem.replace("_", " "), fontsize=18, weight="bold", pad=14)

    plt.tight_layout()
    output_path = output_dir / "recoverability_model_comparison_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    comparison_payload = load_json(Path(args.comparison_json))
    output_dir = Path(args.output_dir)

    image_paths: list[Path] = []
    for pair_result in comparison_payload["pair_comparisons"]:
        image_paths.append(plot_pair(pair_result, output_dir))

    overview_path = build_contact_sheet(image_paths, output_dir)

    for image_path in image_paths:
        print(f"Saved pair plot: {image_path.resolve()}")
    print(f"Saved overview plot: {overview_path.resolve()}")


if __name__ == "__main__":
    main()
