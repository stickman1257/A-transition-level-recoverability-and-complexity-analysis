from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


class ReconstructionVisualizer:
    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)

    def plot_window(
        self,
        target_window: np.ndarray,
        recon_window: np.ndarray,
        channel_index: int = 0,
        title: str = "CVAE Reconstruction",
        save_path: str | Path | None = None,
    ) -> None:
        feature_name = self.feature_columns[channel_index]
        plt.figure(figsize=(14, 5))
        plt.plot(target_window[:, channel_index], label=f"Target ({feature_name})", alpha=0.5, color="gray")
        plt.plot(recon_window[:, channel_index], label=f"Reconstruction ({feature_name})", color="red", linewidth=1)
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_sampling_window(
        self,
        target_window: np.ndarray,
        sampled_reconstructions: np.ndarray,
        channel_index: int = 0,
        title: str = "CVAE Sampling Reconstruction",
        save_path: str | Path | None = None,
    ) -> None:
        feature_name = self.feature_columns[channel_index]
        sampled_channel = sampled_reconstructions[:, :, channel_index]
        mean_recon = np.mean(sampled_channel, axis=0)
        lower = np.percentile(sampled_channel, 5, axis=0)
        upper = np.percentile(sampled_channel, 95, axis=0)

        plt.figure(figsize=(14, 5))
        for sampled_trace in sampled_channel:
            plt.plot(sampled_trace, color="tab:blue", alpha=0.01, linewidth=0.6)
        plt.fill_between(
            np.arange(len(mean_recon)),
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label="5-95% interval",
        )
        plt.plot(target_window[:, channel_index], label=f"Target ({feature_name})", color="black", linewidth=1.2)
        plt.plot(mean_recon, label=f"Sample mean ({feature_name})", color="tab:red", linewidth=1.2)
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper center")
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.close()
