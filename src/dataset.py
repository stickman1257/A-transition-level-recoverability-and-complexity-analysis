from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSample:
    group_id: str
    start_idx: int


class EMGWindowDataset(Dataset):
    def __init__(
        self,
        target_df: pd.DataFrame,
        cond_df: pd.DataFrame,
        feature_columns: Sequence[str],
        window_size: int = 1024,
        stride: int = 512,
        allowed_groups: Sequence[str] | None = None,
    ) -> None:
        self.target_df = target_df
        self.cond_df = cond_df
        self.feature_columns = list(feature_columns)
        self.window_size = window_size
        self.stride = stride
        self.allowed_groups = set(allowed_groups) if allowed_groups is not None else None
        self.samples = self._build_samples()

    def _build_samples(self) -> list[WindowSample]:
        samples: list[WindowSample] = []
        for group_id, group in self.target_df.groupby("GROUP_ID", sort=False):
            if self.allowed_groups is not None and group_id not in self.allowed_groups:
                continue
            length = len(group)
            if length < self.window_size:
                continue
            for start_idx in range(0, length - self.window_size + 1, self.stride):
                samples.append(WindowSample(group_id=group_id, start_idx=start_idx))
        if not samples:
            raise ValueError("No sliding-window samples were created. Check window size and groups.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        target_group = self.target_df[self.target_df["GROUP_ID"] == sample.group_id]
        cond_group = self.cond_df[self.cond_df["GROUP_ID"] == sample.group_id]

        start = sample.start_idx
        end = start + self.window_size

        target_window = target_group.iloc[start:end][self.feature_columns].to_numpy(dtype=np.float32).T
        cond_window = cond_group.iloc[start:end][self.feature_columns].to_numpy(dtype=np.float32).T

        return torch.from_numpy(target_window), torch.from_numpy(cond_window)

