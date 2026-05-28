from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from signal_metrics import permutation_entropy, sample_entropy


DEFAULT_OUTPUT_DIR = "data/complexity_stagepair_suite/sanity_check"
CONDITIONS = ("original", "mildly_smoothed", "strongly_smoothed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic sanity check for complexity reduction under smoothing.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save sanity-check outputs.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling frequency of the synthetic EMG signal.")
    parser.add_argument("--window-size", type=int, default=1024, help="Synthetic window length in samples.")
    parser.add_argument("--num-windows", type=int, default=32, help="Number of synthetic windows to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pe-order", type=int, default=3)
    parser.add_argument("--pe-delay", type=int, default=1)
    parser.add_argument("--pe-max-points", type=int, default=3000)
    parser.add_argument("--sampen-order", type=int, default=2)
    parser.add_argument("--sampen-r-ratio", type=float, default=0.2)
    parser.add_argument("--sampen-max-points", type=int, default=256)
    return parser.parse_args()


def moving_average(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    kernel = np.ones(int(kernel_size), dtype=np.float64) / float(kernel_size)
    return np.convolve(signal, kernel, mode="same")


def generate_emg_like_signal(length: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    nyquist = 0.5 * fs
    sos = butter(4, [20.0 / nyquist, 200.0 / nyquist], btype="bandpass", output="sos")
    noise = rng.standard_normal(length)
    band_limited = sosfiltfilt(sos, noise)

    envelope_noise = rng.standard_normal(length)
    envelope = np.abs(moving_average(envelope_noise, max(21, int(round(fs * 0.05)))))
    envelope = 0.35 + envelope

    burst = band_limited * envelope
    burst += 0.05 * np.sin(2.0 * np.pi * 8.0 * np.arange(length, dtype=np.float64) / fs)
    return burst.astype(np.float64)


def build_conditions(base_signal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "original": base_signal,
        "mildly_smoothed": moving_average(base_signal, 9),
        "strongly_smoothed": moving_average(base_signal, 41),
    }


def summarize_metric(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def build_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = (
        ("pe_mean", "pe_std", "Permutation Entropy"),
        ("sampen_mean", "sampen_std", "Sample Entropy"),
    )
    x = np.arange(len(summary_df))
    labels = summary_df["condition"].tolist()

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        ax.bar(x, summary_df[mean_col], yerr=summary_df[std_col], color=["#4C78A8", "#72B7B2", "#F58518"])
        ax.set_xticks(x, labels, rotation=15)
        ax.set_title(title)
        ax.set_ylabel("value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    for window_idx in range(args.num_windows):
        base_signal = generate_emg_like_signal(args.window_size, args.fs, rng)
        for condition, signal in build_conditions(base_signal).items():
            rows.append(
                {
                    "window_idx": int(window_idx),
                    "condition": condition,
                    "pe": permutation_entropy(
                        signal,
                        order=args.pe_order,
                        delay=args.pe_delay,
                        normalize=True,
                        max_points=args.pe_max_points,
                    ),
                    "sampen": sample_entropy(
                        signal,
                        order=args.sampen_order,
                        r_ratio=args.sampen_r_ratio,
                        max_points=args.sampen_max_points,
                    ),
                }
            )

    detail_df = pd.DataFrame(rows)
    summary_records: list[dict[str, float | int | str]] = []
    for condition in CONDITIONS:
        condition_df = detail_df[detail_df["condition"] == condition]
        pe_mean, pe_std = summarize_metric(condition_df["pe"])
        sampen_mean, sampen_std = summarize_metric(condition_df["sampen"])
        summary_records.append(
            {
                "condition": condition,
                "pe_mean": pe_mean,
                "pe_std": pe_std,
                "sampen_mean": sampen_mean,
                "sampen_std": sampen_std,
                "num_windows": int(len(condition_df)),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_path = output_dir / "sanity_condition_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pe_means = summary_df.set_index("condition")["pe_mean"]
    sampen_means = summary_df.set_index("condition")["sampen_mean"]
    pe_pass = bool(
        pe_means["original"] >= pe_means["mildly_smoothed"] >= pe_means["strongly_smoothed"]
    )
    sampen_pass = bool(
        sampen_means["original"] >= sampen_means["mildly_smoothed"] >= sampen_means["strongly_smoothed"]
    )

    results_df = pd.DataFrame(
        [
            {
                "check_name": "pe_monotonic_reduction",
                "expected_order": "original >= mildly_smoothed >= strongly_smoothed",
                "original_mean": float(pe_means["original"]),
                "mildly_smoothed_mean": float(pe_means["mildly_smoothed"]),
                "strongly_smoothed_mean": float(pe_means["strongly_smoothed"]),
                "passed": pe_pass,
            },
            {
                "check_name": "sampen_monotonic_reduction",
                "expected_order": "original >= mildly_smoothed >= strongly_smoothed",
                "original_mean": float(sampen_means["original"]),
                "mildly_smoothed_mean": float(sampen_means["mildly_smoothed"]),
                "strongly_smoothed_mean": float(sampen_means["strongly_smoothed"]),
                "passed": sampen_pass,
            },
            {
                "check_name": "overall",
                "expected_order": "all checks must pass",
                "original_mean": float("nan"),
                "mildly_smoothed_mean": float("nan"),
                "strongly_smoothed_mean": float("nan"),
                "passed": bool(pe_pass and sampen_pass),
            },
        ]
    )
    results_path = output_dir / "sanity_check_results.csv"
    results_df.to_csv(results_path, index=False)

    figure_path = output_dir / "sanity_pe_sampen_progression.png"
    build_plot(summary_df, figure_path)

    print(f"Saved sanity summary CSV to: {summary_path.resolve()}")
    print(f"Saved sanity results CSV to: {results_path.resolve()}")
    print(f"Saved sanity figure to: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
