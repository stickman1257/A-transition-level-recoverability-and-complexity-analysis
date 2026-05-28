from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signal_metrics import aggregate_metric_payloads, compute_signal_metrics
from utils import estimate_sampling_frequency


@dataclass
class TestResult:
    metrics: dict
    target_window: np.ndarray
    recon_window: np.ndarray
    sampled_reconstructions: np.ndarray | None = None


class ReconstructionTester:
    def __init__(
        self,
        model: nn.Module,
        target_scaler,
        feature_columns: Sequence[str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.target_scaler = target_scaler
        self.feature_columns = list(feature_columns)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def reconstruct_batch(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        deterministic: bool = True,
        target_is_scaled: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            recon, _, _ = self.model(
                target.to(self.device),
                condition.to(self.device),
                deterministic=deterministic,
            )
        target_np = target.squeeze(0).cpu().numpy().T
        recon_np = recon.squeeze(0).cpu().numpy().T
        if target_is_scaled:
            target_final = self.target_scaler.inverse_transform_array(target_np)
        else:
            target_final = target_np
        recon_final = self.target_scaler.inverse_transform_array(recon_np)
        return target_final, recon_final

    def evaluate_loader(self, data_loader: DataLoader) -> float:
        scores = []
        with torch.no_grad():
            for target, condition in data_loader:
                recon, _, _ = self.model(
                    target.to(self.device),
                    condition.to(self.device),
                    deterministic=True,
                )
                scores.append(torch.mean((recon - target.to(self.device)) ** 2).item())
        return float(np.mean(scores))

    def converged_reconstruction(
        self,
        cond_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_id: str,
        window_size: int,
        feature_columns: Sequence[str],
        iterations: int = 100,
    ) -> TestResult:
        cond_group = cond_df_scaled[cond_df_scaled["GROUP_ID"] == group_id]
        target_group = target_df[target_df["GROUP_ID"] == group_id]
        if len(cond_group) == 0 or len(target_group) == 0:
            available_group_ids = sorted(set(target_df["GROUP_ID"]).intersection(cond_df_scaled["GROUP_ID"]))
            preview = ", ".join(map(str, available_group_ids[:10]))
            raise ValueError(
                f"GROUP_ID={group_id} was not found. "
                f"Available examples: {preview}"
                + (" ..." if len(available_group_ids) > 10 else "")
            )
        if len(cond_group) < window_size or len(target_group) < window_size:
            raise ValueError(f"GROUP_ID={group_id} is shorter than window_size={window_size}.")

        sample_target = target_group.iloc[:window_size][list(feature_columns)].to_numpy(dtype=np.float32)
        sample_cond = cond_group.iloc[:window_size][list(feature_columns)].to_numpy(dtype=np.float32)

        target_tensor = torch.from_numpy(sample_target.T).unsqueeze(0)
        cond_tensor = torch.from_numpy(sample_cond.T).unsqueeze(0)

        del iterations
        _, mean_recon = self.reconstruct_batch(target_tensor, cond_tensor, target_is_scaled=False)
        fs = estimate_sampling_frequency(target_df[target_df["GROUP_ID"] == group_id])
        metrics = compute_signal_metrics(sample_target, mean_recon, fs=fs, channel_names=feature_columns)
        return TestResult(
            metrics=metrics,
            target_window=sample_target,
            recon_window=mean_recon,
        )

    def sampled_reconstruction(
        self,
        cond_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_id: str,
        window_size: int,
        feature_columns: Sequence[str],
        iterations: int = 1000,
    ) -> TestResult:
        cond_group = cond_df_scaled[cond_df_scaled["GROUP_ID"] == group_id]
        target_group = target_df[target_df["GROUP_ID"] == group_id]
        if len(cond_group) == 0 or len(target_group) == 0:
            available_group_ids = sorted(set(target_df["GROUP_ID"]).intersection(cond_df_scaled["GROUP_ID"]))
            preview = ", ".join(map(str, available_group_ids[:10]))
            raise ValueError(
                f"GROUP_ID={group_id} was not found. "
                f"Available examples: {preview}"
                + (" ..." if len(available_group_ids) > 10 else "")
            )
        if len(cond_group) < window_size or len(target_group) < window_size:
            raise ValueError(f"GROUP_ID={group_id} is shorter than window_size={window_size}.")

        sample_target = target_group.iloc[:window_size][list(feature_columns)].to_numpy(dtype=np.float32)
        sample_cond = cond_group.iloc[:window_size][list(feature_columns)].to_numpy(dtype=np.float32)

        target_tensor = torch.from_numpy(sample_target.T).unsqueeze(0)
        cond_tensor = torch.from_numpy(sample_cond.T).unsqueeze(0)

        _, recon_final = self.reconstruct_batch(
            target_tensor,
            cond_tensor,
            deterministic=True,
            target_is_scaled=False,
        )
        sampled = np.repeat(recon_final[None, :, :], iterations, axis=0)
        mean_recon = np.mean(sampled, axis=0)
        fs = estimate_sampling_frequency(target_df[target_df["GROUP_ID"] == group_id])
        metrics = compute_signal_metrics(sample_target, mean_recon, fs=fs, channel_names=feature_columns)
        return TestResult(
            metrics=metrics,
            target_window=sample_target,
            recon_window=mean_recon,
            sampled_reconstructions=sampled,
        )

    def evaluate_group_windows(
        self,
        cond_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_id: str,
        window_size: int,
        feature_columns: Sequence[str],
        stride: int | None = None,
        include_window_summaries: bool = False,
    ) -> dict:
        if stride is None:
            stride = window_size

        cond_group = cond_df_scaled[cond_df_scaled["GROUP_ID"] == group_id]
        target_group = target_df[target_df["GROUP_ID"] == group_id]
        if len(cond_group) == 0 or len(target_group) == 0:
            available_group_ids = sorted(set(target_df["GROUP_ID"]).intersection(cond_df_scaled["GROUP_ID"]))
            preview = ", ".join(map(str, available_group_ids[:10]))
            raise ValueError(
                f"GROUP_ID={group_id} was not found. "
                f"Available examples: {preview}"
                + (" ..." if len(available_group_ids) > 10 else "")
            )
        if len(cond_group) < window_size or len(target_group) < window_size:
            raise ValueError(f"GROUP_ID={group_id} is shorter than window_size={window_size}.")

        fs = estimate_sampling_frequency(target_group)
        metrics_payloads: list[dict] = []
        window_summaries: list[dict[str, float | int]] = []

        for start_idx in range(0, len(target_group) - window_size + 1, stride):
            end_idx = start_idx + window_size
            target_window = target_group.iloc[start_idx:end_idx][list(feature_columns)].to_numpy(dtype=np.float32)
            cond_window = cond_group.iloc[start_idx:end_idx][list(feature_columns)].to_numpy(dtype=np.float32)

            target_tensor = torch.from_numpy(target_window.T).unsqueeze(0)
            cond_tensor = torch.from_numpy(cond_window.T).unsqueeze(0)
            _, recon_window = self.reconstruct_batch(target_tensor, cond_tensor, target_is_scaled=False)

            payload = compute_signal_metrics(target_window, recon_window, fs=fs, channel_names=feature_columns)
            metrics_payloads.append(payload)
            if include_window_summaries:
                window_summaries.append({"start_idx": start_idx, **payload["summary"]})

        aggregate = aggregate_metric_payloads(metrics_payloads)
        result = {
            "group_id": group_id,
            "num_windows": len(metrics_payloads),
            "window_stride": stride,
            "summary_mean": aggregate["summary_mean"],
            "summary_std": aggregate["summary_std"],
            "per_channel_mean": aggregate["per_channel_mean"],
            "per_channel_std": aggregate["per_channel_std"],
        }
        if include_window_summaries:
            result["window_summaries"] = window_summaries
        return result

    def evaluate_groups(
        self,
        cond_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_ids: Sequence[str],
        window_size: int,
        feature_columns: Sequence[str],
        stride: int | None = None,
        include_window_summaries: bool = False,
    ) -> dict:
        group_results: list[dict] = []
        for group_id in group_ids:
            group_results.append(
                self.evaluate_group_windows(
                    cond_df_scaled=cond_df_scaled,
                    target_df=target_df,
                    group_id=group_id,
                    window_size=window_size,
                    feature_columns=feature_columns,
                    stride=stride,
                    include_window_summaries=include_window_summaries,
                )
            )

        group_payloads = [
            {
                "summary": result["summary_mean"],
                "per_channel": result["per_channel_mean"],
            }
            for result in group_results
        ]
        aggregate = aggregate_metric_payloads(group_payloads)
        return {
            "num_groups": len(group_results),
            "group_ids": list(group_ids),
            "summary_mean": aggregate["summary_mean"],
            "summary_std": aggregate["summary_std"],
            "per_channel_mean": aggregate["per_channel_mean"],
            "per_channel_std": aggregate["per_channel_std"],
            "groups": group_results,
        }


CVAETester = ReconstructionTester
