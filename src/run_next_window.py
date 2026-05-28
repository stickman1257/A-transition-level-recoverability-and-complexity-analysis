from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import (
    AVAILABLE_RECONSTRUCTION_MODELS,
    ConvBlock1D,
    DownBlock1D,
    ResidualDilatedBlock1D,
    UpBlock1D,
    normalize_reconstruction_model_type,
)
from signal_metrics import aggregate_metric_payloads, compute_signal_metrics
from utils import (
    DataFrameScaler,
    SplitConfig,
    align_frames,
    build_emg_stage_frames,
    checkpoint_metadata,
    estimate_sampling_frequency,
    estimate_window_size,
    filter_finite_feature_columns,
    load_raw_emg_directory,
    load_stage_frame,
    load_table,
    make_generator,
    normalize_stage_name,
    resolve_stage_frame,
    save_json,
    seed_everything,
    split_group_ids,
    split_subject_cv,
    validate_stage_name,
)
from visualize import ReconstructionVisualizer


@dataclass
class WindowPair:
    group_id: str
    start_idx: int


DEFAULT_NAIVE_BASELINES = (
    "copy_current",
    "shift_last_value",
    "moving_average",
)

PREDICTOR_MODEL_TYPES = AVAILABLE_RECONSTRUCTION_MODELS + (
    "larocs_rnn_lstm_envelope",
    "larocs_rnn_lstm",
    "larocs_cnn_lstm",
)
PREDICTOR_MODEL_TYPE_ALIASES = {
    "rnn": "larocs_rnn_lstm_envelope",
    "rnn_lstm_envelope": "larocs_rnn_lstm_envelope",
    "larocs_rnn": "larocs_rnn_lstm_envelope",
    "rnn_lstm": "larocs_rnn_lstm",
    "larocs_lstm": "larocs_rnn_lstm",
    "cnn": "larocs_cnn_lstm",
    "cnn_lstm": "larocs_cnn_lstm",
    "larocs_cnn": "larocs_cnn_lstm",
}
DIRECT_PREDICTOR_MODEL_TYPES = {
    "larocs_rnn_lstm_envelope",
    "larocs_rnn_lstm",
    "larocs_cnn_lstm",
}


def normalize_predictor_model_type(model_type: str) -> str:
    canonical = str(model_type).strip().lower().replace(" ", "_").replace("-", "_")
    canonical = PREDICTOR_MODEL_TYPE_ALIASES.get(canonical, canonical)
    if canonical in PREDICTOR_MODEL_TYPES:
        return canonical
    return normalize_reconstruction_model_type(model_type)


class NextWindowDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        target_df: pd.DataFrame,
        feature_columns: Sequence[str],
        input_seq_len: int,
        pred_seq_len: int,
        horizon_gap_seq_len: int,
        stride: int,
        allowed_groups: Sequence[str] | None = None,
    ) -> None:
        self.input_df = input_df
        self.target_df = target_df
        self.feature_columns = list(feature_columns)
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.horizon_gap_seq_len = horizon_gap_seq_len
        self.stride = stride
        self.allowed_groups = set(allowed_groups) if allowed_groups is not None else None
        self.samples = self._build_samples()

    def _build_samples(self) -> list[WindowPair]:
        samples: list[WindowPair] = []
        for group_id, group in self.input_df.groupby("GROUP_ID", sort=False):
            if self.allowed_groups is not None and group_id not in self.allowed_groups:
                continue
            length = len(group)
            max_start = length - (self.input_seq_len + self.horizon_gap_seq_len + self.pred_seq_len)
            if max_start < 0:
                continue
            for start_idx in range(0, max_start + 1, self.stride):
                samples.append(WindowPair(group_id=group_id, start_idx=start_idx))
        if not samples:
            raise ValueError(
                "No next-window samples were created. Check input_seq_len, pred_seq_len, stride, and groups."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        input_group = self.input_df[self.input_df["GROUP_ID"] == sample.group_id]
        target_group = self.target_df[self.target_df["GROUP_ID"] == sample.group_id]

        start = sample.start_idx
        mid = start + self.input_seq_len
        target_start = mid + self.horizon_gap_seq_len
        end = target_start + self.pred_seq_len

        x_window = input_group.iloc[start:mid][self.feature_columns].to_numpy(dtype=np.float32)
        y_window = target_group.iloc[target_start:end][self.feature_columns].to_numpy(dtype=np.float32)
        return torch.from_numpy(x_window), torch.from_numpy(y_window)


@dataclass
class PredictorConfig:
    num_channels: int
    model_type: str = "lstm"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    architecture: str = "seq2seq"
    prediction_seq_len: int | None = None

    def __post_init__(self) -> None:
        self.model_type = normalize_predictor_model_type(self.model_type)
        if self.model_type not in PREDICTOR_MODEL_TYPES:
            raise ValueError(
                f"Unsupported predictor model_type: {self.model_type}. "
                f"Use one of {', '.join(PREDICTOR_MODEL_TYPES)}."
            )
        if self.prediction_seq_len is not None and self.prediction_seq_len <= 0:
            raise ValueError(f"prediction_seq_len must be positive, got {self.prediction_seq_len}")
        if self.model_type in DIRECT_PREDICTOR_MODEL_TYPES:
            self.architecture = "direct"
        if self.architecture not in {"many_to_many", "seq2seq", "direct"}:
            raise ValueError(f"Unsupported predictor architecture: {self.architecture}")


def load_predictor_config(
    config_payload: dict,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> PredictorConfig:
    payload = dict(config_payload)
    if "model_type" not in payload:
        payload["model_type"] = "lstm"
    payload["model_type"] = normalize_predictor_model_type(payload["model_type"])
    if "architecture" not in payload:
        if payload["model_type"] in DIRECT_PREDICTOR_MODEL_TYPES:
            payload["architecture"] = "direct"
        else:
            uses_seq2seq_keys = bool(model_state_dict) and any(
                key.startswith(("encoder.", "decoder.", "output_proj."))
                for key in model_state_dict
            )
            payload["architecture"] = "seq2seq" if uses_seq2seq_keys else "many_to_many"
    return PredictorConfig(**payload)


class LSTMNextWindowPredictor(nn.Module):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        if config.model_type in {"lstm", "gru"}:
            self._init_recurrent_predictor()
        elif config.model_type == "cnn_1d":
            self._init_cnn_predictor()
        elif config.model_type in {"conditional_unet", "vanilla_unet"}:
            self._init_unet_predictor()
        elif config.model_type == "larocs_rnn_lstm_envelope":
            self._init_larocs_rnn_lstm_envelope_predictor()
        elif config.model_type == "larocs_rnn_lstm":
            self._init_larocs_rnn_lstm_predictor()
        elif config.model_type == "larocs_cnn_lstm":
            self._init_larocs_cnn_lstm_predictor()
        else:
            raise ValueError(f"Unsupported predictor model_type: {config.model_type}")

    def _require_prediction_seq_len(self) -> int:
        if self.config.prediction_seq_len is None:
            raise ValueError(
                f"model_type={self.config.model_type} requires prediction_seq_len in PredictorConfig."
            )
        return int(self.config.prediction_seq_len)

    def _resolve_prediction_steps(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> int:
        if prediction_steps is not None:
            return int(prediction_steps)
        if target_future is not None:
            return int(target_future.size(1))
        if self.config.prediction_seq_len is not None:
            return int(self.config.prediction_seq_len)
        return int(x.size(1))

    def _reshape_direct_prediction(
        self,
        pred_flat: torch.Tensor,
        prediction_steps: int,
    ) -> torch.Tensor:
        base_prediction_steps = self._require_prediction_seq_len()
        pred = pred_flat.view(pred_flat.size(0), base_prediction_steps, self.config.num_channels)
        if pred.size(1) != prediction_steps:
            pred = F.interpolate(
                pred.transpose(1, 2),
                size=prediction_steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return pred

    def _init_recurrent_predictor(self) -> None:
        recurrent_kwargs = {
            "input_size": self.config.num_channels,
            "hidden_size": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "batch_first": True,
            "dropout": self.config.dropout if self.config.num_layers > 1 else 0.0,
        }
        rnn_cls = nn.LSTM if self.config.model_type == "lstm" else nn.GRU

        if self.config.architecture == "many_to_many":
            self.recurrent = rnn_cls(**recurrent_kwargs)
            self.output_proj = nn.Linear(self.config.hidden_dim, self.config.num_channels)
        else:
            self.encoder = rnn_cls(**recurrent_kwargs)
            self.decoder = rnn_cls(**recurrent_kwargs)
            self.output_proj = nn.Linear(self.config.hidden_dim, self.config.num_channels)

    def _init_cnn_predictor(self) -> None:
        channels = max(self.config.hidden_dim, self.config.num_channels)
        self.input_proj = nn.Conv1d(self.config.num_channels, channels, kernel_size=1)
        self.trunk = nn.Sequential(
            ResidualDilatedBlock1D(channels, dilation=1),
            ResidualDilatedBlock1D(channels, dilation=2),
            ResidualDilatedBlock1D(channels, dilation=4),
            ResidualDilatedBlock1D(channels, dilation=8),
        )
        self.output_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, self.config.num_channels, kernel_size=1),
        )

    def _init_unet_predictor(self) -> None:
        base_channels = max(self.config.hidden_dim, self.config.num_channels)
        self.input_proj = nn.Conv1d(self.config.num_channels, base_channels, kernel_size=1)
        self.down1 = DownBlock1D(base_channels, base_channels)
        self.down2 = DownBlock1D(base_channels, base_channels * 2)
        self.down3 = DownBlock1D(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock1D(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock1D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock1D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock1D(base_channels * 2, base_channels, base_channels)
        self.output_head = nn.Conv1d(base_channels, self.config.num_channels, kernel_size=1)

    def _init_larocs_rnn_lstm_envelope_predictor(self) -> None:
        prediction_seq_len = self._require_prediction_seq_len()
        self.larocs_envelope_lstm = nn.LSTM(
            input_size=self.config.num_channels,
            hidden_size=self.config.hidden_dim,
            num_layers=max(1, self.config.num_layers),
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.larocs_envelope_head = nn.Linear(
            self.config.hidden_dim,
            prediction_seq_len * self.config.num_channels,
        )

    def _init_larocs_rnn_lstm_predictor(self) -> None:
        prediction_seq_len = self._require_prediction_seq_len()
        self.larocs_lstm = nn.LSTM(
            input_size=self.config.num_channels,
            hidden_size=self.config.hidden_dim,
            num_layers=max(1, self.config.num_layers),
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        dense_hidden = max(self.config.hidden_dim, self.config.num_channels * 2)
        self.larocs_lstm_head = nn.Sequential(
            nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity(),
            nn.Linear(self.config.hidden_dim, dense_hidden),
            nn.Tanh(),
            nn.Linear(dense_hidden, prediction_seq_len * self.config.num_channels),
        )

    def _init_larocs_cnn_lstm_predictor(self) -> None:
        prediction_seq_len = self._require_prediction_seq_len()
        conv_channels = max(self.config.hidden_dim // 2, self.config.num_channels)
        self.larocs_cnn_encoder = nn.Sequential(
            nn.Conv1d(self.config.num_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv_channels, self.config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.larocs_cnn_lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=max(1, self.config.num_layers),
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        dense_hidden = max(self.config.hidden_dim, self.config.num_channels * 2)
        self.larocs_cnn_lstm_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, prediction_seq_len * self.config.num_channels),
        )

    def _forward_recurrent(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        if self.config.architecture == "many_to_many":
            recurrent_out, _ = self.recurrent(x)
            return self.output_proj(recurrent_out)

        if prediction_steps is None:
            if target_future is not None:
                prediction_steps = target_future.size(1)
            else:
                prediction_steps = x.size(1)

        _, state = self.encoder(x)

        # Train/val/test all use the same direct autoregressive decoding path.
        decoder_input = x.new_zeros((x.size(0), 1, self.config.num_channels))
        outputs: list[torch.Tensor] = []
        for _ in range(prediction_steps):
            decoder_out, state = self.decoder(decoder_input, state)
            step_pred = self.output_proj(decoder_out)
            outputs.append(step_pred)
            decoder_input = step_pred
        return torch.cat(outputs, dim=1)

    def _forward_cnn(
        self,
        x: torch.Tensor,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        if prediction_steps is None:
            prediction_steps = x.size(1)

        features = x.transpose(1, 2)
        features = self.input_proj(features)
        features = self.trunk(features)
        pred = self.output_head(features)
        if pred.size(-1) != prediction_steps:
            pred = F.interpolate(pred, size=prediction_steps, mode="linear", align_corners=False)
        return pred.transpose(1, 2)

    def _forward_unet(
        self,
        x: torch.Tensor,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        if prediction_steps is None:
            prediction_steps = x.size(1)

        features = self.input_proj(x.transpose(1, 2))
        skip1, features = self.down1(features)
        skip2, features = self.down2(features)
        skip3, features = self.down3(features)
        features = self.bottleneck(features)
        features = self.up3(features, skip3)
        features = self.up2(features, skip2)
        features = self.up1(features, skip1)
        pred = self.output_head(features)
        if pred.size(-1) != prediction_steps:
            pred = F.interpolate(pred, size=prediction_steps, mode="linear", align_corners=False)
        return pred.transpose(1, 2)

    def _forward_larocs_rnn_lstm_envelope(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        resolved_steps = self._resolve_prediction_steps(
            x,
            target_future=target_future,
            prediction_steps=prediction_steps,
        )
        recurrent_out, _ = self.larocs_envelope_lstm(x)
        pred_flat = self.larocs_envelope_head(recurrent_out[:, -1, :])
        return self._reshape_direct_prediction(pred_flat, resolved_steps)

    def _forward_larocs_rnn_lstm(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        resolved_steps = self._resolve_prediction_steps(
            x,
            target_future=target_future,
            prediction_steps=prediction_steps,
        )
        recurrent_out, _ = self.larocs_lstm(x)
        pred_flat = self.larocs_lstm_head(recurrent_out[:, -1, :])
        return self._reshape_direct_prediction(pred_flat, resolved_steps)

    def _forward_larocs_cnn_lstm(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        resolved_steps = self._resolve_prediction_steps(
            x,
            target_future=target_future,
            prediction_steps=prediction_steps,
        )
        features = self.larocs_cnn_encoder(x.transpose(1, 2)).transpose(1, 2)
        recurrent_out, _ = self.larocs_cnn_lstm(features)
        pred_flat = self.larocs_cnn_lstm_head(recurrent_out[:, -1, :])
        return self._reshape_direct_prediction(pred_flat, resolved_steps)

    def forward(
        self,
        x: torch.Tensor,
        target_future: torch.Tensor | None = None,
        prediction_steps: int | None = None,
    ) -> torch.Tensor:
        if self.config.model_type in {"lstm", "gru"}:
            return self._forward_recurrent(
                x,
                target_future=target_future,
                prediction_steps=prediction_steps,
            )
        if self.config.model_type == "cnn_1d":
            return self._forward_cnn(x, prediction_steps=prediction_steps or (target_future.size(1) if target_future is not None else None))
        if self.config.model_type in {"conditional_unet", "vanilla_unet"}:
            return self._forward_unet(x, prediction_steps=prediction_steps or (target_future.size(1) if target_future is not None else None))
        if self.config.model_type == "larocs_rnn_lstm_envelope":
            return self._forward_larocs_rnn_lstm_envelope(
                x,
                target_future=target_future,
                prediction_steps=prediction_steps,
            )
        if self.config.model_type == "larocs_rnn_lstm":
            return self._forward_larocs_rnn_lstm(
                x,
                target_future=target_future,
                prediction_steps=prediction_steps,
            )
        if self.config.model_type == "larocs_cnn_lstm":
            return self._forward_larocs_cnn_lstm(
                x,
                target_future=target_future,
                prediction_steps=prediction_steps,
            )
        raise ValueError(f"Unsupported predictor model_type: {self.config.model_type}")


@dataclass
class PredictionTrainerConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    patience: int = 10
    device: str = "cpu"


class NextWindowTrainer:
    def __init__(self, model: LSTMNextWindowPredictor, config: PredictionTrainerConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> tuple[LSTMNextWindowPredictor, list[dict]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        history: list[dict] = []
        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(x_batch, prediction_steps=y_batch.size(1))
                loss = self.criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x_batch.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            val_loss = self.evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch}/{self.config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(best_state)
        return self.model, history

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(x_batch, prediction_steps=y_batch.size(1))
                loss = self.criterion(pred, y_batch)
                total_loss += loss.item() * x_batch.size(0)
        return total_loss / len(data_loader.dataset)


@dataclass
class PredictionResult:
    metrics: dict
    target_window: np.ndarray
    pred_window: np.ndarray


def _resample_window(window: np.ndarray, output_len: int) -> np.ndarray:
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")
    if window.shape[0] == output_len:
        return window.astype(np.float32, copy=True)
    if window.shape[0] == 1:
        return np.repeat(window.astype(np.float32), output_len, axis=0)

    src = np.linspace(0.0, 1.0, num=window.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=output_len, dtype=np.float64)
    channels = [
        np.interp(dst, src, window[:, idx].astype(np.float64))
        for idx in range(window.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _causal_moving_average(window: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return window.astype(np.float32, copy=True)

    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.empty_like(window, dtype=np.float32)
    for idx in range(window.shape[1]):
        smoothed[:, idx] = np.convolve(window[:, idx], kernel, mode="full")[: window.shape[0]]
    return smoothed


def naive_predict_next_window(
    input_window: np.ndarray,
    pred_seq_len: int,
    baseline_name: str,
) -> np.ndarray:
    reference = _resample_window(input_window, pred_seq_len)

    if baseline_name == "copy_current":
        return reference

    if baseline_name == "shift_last_value":
        shifted = np.empty_like(reference)
        shifted[:-1] = reference[1:]
        shifted[-1] = reference[-1]
        return shifted

    if baseline_name == "moving_average":
        kernel_size = max(2, min(reference.shape[0], max(2, int(round(reference.shape[0] * 0.1)))))
        return _causal_moving_average(reference, kernel_size)

    raise ValueError(
        f"Unsupported naive baseline: {baseline_name}. "
        f"Use one of {', '.join(DEFAULT_NAIVE_BASELINES)}."
    )


def evaluate_naive_baseline_groups(
    input_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: Sequence[str],
    group_ids: Sequence[str],
    input_seq_len: int,
    pred_seq_len: int,
    stride: int,
    horizon_gap_seq_len: int = 0,
    baseline_names: Sequence[str] = DEFAULT_NAIVE_BASELINES,
    include_window_summaries: bool = False,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for baseline_name in baseline_names:
        group_results: list[dict] = []
        for group_id in group_ids:
            input_group = input_df[input_df["GROUP_ID"] == group_id]
            target_group = target_df[target_df["GROUP_ID"] == group_id]
            if len(input_group) == 0 or len(target_group) == 0:
                raise ValueError(f"GROUP_ID={group_id} was not found for baseline evaluation.")
            required_len = input_seq_len + horizon_gap_seq_len + pred_seq_len
            if len(input_group) < required_len or len(target_group) < required_len:
                raise ValueError(
                    f"GROUP_ID={group_id} is shorter than required length={required_len}."
                )

            metrics_payloads: list[dict] = []
            window_summaries: list[dict[str, float | int]] = []
            fs = estimate_sampling_frequency(target_group)
            max_start = len(input_group) - required_len

            for start_idx in range(0, max_start + 1, stride):
                mid_idx = start_idx + input_seq_len
                target_start = mid_idx + horizon_gap_seq_len
                end_idx = target_start + pred_seq_len
                x_window = input_group.iloc[start_idx:mid_idx][list(feature_columns)].to_numpy(dtype=np.float32)
                y_window = target_group.iloc[target_start:end_idx][list(feature_columns)].to_numpy(dtype=np.float32)
                pred_window = naive_predict_next_window(x_window, pred_seq_len=pred_seq_len, baseline_name=baseline_name)
                payload = compute_signal_metrics(y_window, pred_window, fs=fs, channel_names=feature_columns)
                metrics_payloads.append(payload)
                if include_window_summaries:
                    window_summaries.append({"start_idx": start_idx, **payload["summary"]})

            aggregate = aggregate_metric_payloads(metrics_payloads)
            group_result = {
                "group_id": group_id,
                "num_windows": len(metrics_payloads),
                "window_stride": stride,
                "summary_mean": aggregate["summary_mean"],
                "summary_std": aggregate["summary_std"],
                "per_channel_mean": aggregate["per_channel_mean"],
                "per_channel_std": aggregate["per_channel_std"],
            }
            if include_window_summaries:
                group_result["window_summaries"] = window_summaries
            group_results.append(group_result)

        aggregate = aggregate_metric_payloads(
            [
                {
                    "summary": result["summary_mean"],
                    "per_channel": result["per_channel_mean"],
                }
                for result in group_results
            ]
        )
        results[baseline_name] = {
            "baseline_name": baseline_name,
            "num_groups": len(group_results),
            "group_ids": list(group_ids),
            "summary_mean": aggregate["summary_mean"],
            "summary_std": aggregate["summary_std"],
            "per_channel_mean": aggregate["per_channel_mean"],
            "per_channel_std": aggregate["per_channel_std"],
            "groups": group_results,
        }

    return results


class NextWindowTester:
    def __init__(
        self,
        model: LSTMNextWindowPredictor,
        target_scaler: DataFrameScaler,
        feature_columns: Sequence[str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.target_scaler = target_scaler
        self.feature_columns = list(feature_columns)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_next_window(
        self,
        input_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_id: str,
        input_seq_len: int,
        pred_seq_len: int,
        horizon_gap_seq_len: int = 0,
    ) -> PredictionResult:
        input_group = input_df_scaled[input_df_scaled["GROUP_ID"] == group_id]
        target_group = target_df[target_df["GROUP_ID"] == group_id]
        if len(input_group) == 0 or len(target_group) == 0:
            available_group_ids = sorted(set(target_df["GROUP_ID"]).intersection(input_df_scaled["GROUP_ID"]))
            preview = ", ".join(map(str, available_group_ids[:10]))
            raise ValueError(
                f"GROUP_ID={group_id} was not found. "
                f"Available examples: {preview}" + (" ..." if len(available_group_ids) > 10 else "")
            )
        required_len = input_seq_len + horizon_gap_seq_len + pred_seq_len
        if len(input_group) < required_len or len(target_group) < required_len:
            raise ValueError(f"GROUP_ID={group_id} is shorter than required length={required_len}.")

        x_window = input_group.iloc[:input_seq_len][self.feature_columns].to_numpy(dtype=np.float32)
        target_start = input_seq_len + horizon_gap_seq_len
        y_window = target_group.iloc[target_start : target_start + pred_seq_len][self.feature_columns].to_numpy(
            dtype=np.float32
        )

        x_tensor = torch.from_numpy(x_window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(x_tensor, prediction_steps=pred_seq_len).squeeze(0).cpu().numpy()

        pred_window = self.target_scaler.inverse_transform_array(pred_scaled)
        fs = estimate_sampling_frequency(target_df[target_df["GROUP_ID"] == group_id])
        metrics = compute_signal_metrics(y_window, pred_window, fs=fs, channel_names=self.feature_columns)
        return PredictionResult(
            metrics=metrics,
            target_window=y_window,
            pred_window=pred_window,
        )

    def evaluate_group_windows(
        self,
        input_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_id: str,
        input_seq_len: int,
        pred_seq_len: int,
        horizon_gap_seq_len: int = 0,
        stride: int | None = None,
        include_window_summaries: bool = False,
    ) -> dict:
        if stride is None:
            stride = input_seq_len

        input_group = input_df_scaled[input_df_scaled["GROUP_ID"] == group_id]
        target_group = target_df[target_df["GROUP_ID"] == group_id]
        if len(input_group) == 0 or len(target_group) == 0:
            available_group_ids = sorted(set(target_df["GROUP_ID"]).intersection(input_df_scaled["GROUP_ID"]))
            preview = ", ".join(map(str, available_group_ids[:10]))
            raise ValueError(
                f"GROUP_ID={group_id} was not found. "
                f"Available examples: {preview}" + (" ..." if len(available_group_ids) > 10 else "")
            )
        required_len = input_seq_len + horizon_gap_seq_len + pred_seq_len
        if len(input_group) < required_len or len(target_group) < required_len:
            raise ValueError(f"GROUP_ID={group_id} is shorter than required length={required_len}.")

        metrics_payloads: list[dict] = []
        window_summaries: list[dict[str, float | int]] = []
        fs = estimate_sampling_frequency(target_group)

        for start_idx in range(0, len(input_group) - required_len + 1, stride):
            mid_idx = start_idx + input_seq_len
            target_start = mid_idx + horizon_gap_seq_len
            end_idx = target_start + pred_seq_len
            x_window = input_group.iloc[start_idx:mid_idx][self.feature_columns].to_numpy(dtype=np.float32)
            y_window = target_group.iloc[target_start:end_idx][self.feature_columns].to_numpy(dtype=np.float32)

            x_tensor = torch.from_numpy(x_window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_scaled = self.model(x_tensor, prediction_steps=pred_seq_len).squeeze(0).cpu().numpy()
            pred_window = self.target_scaler.inverse_transform_array(pred_scaled)

            payload = compute_signal_metrics(y_window, pred_window, fs=fs, channel_names=self.feature_columns)
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
        input_df_scaled: pd.DataFrame,
        target_df: pd.DataFrame,
        group_ids: Sequence[str],
        input_seq_len: int,
        pred_seq_len: int,
        horizon_gap_seq_len: int = 0,
        stride: int | None = None,
        include_window_summaries: bool = False,
    ) -> dict:
        group_results: list[dict] = []
        for group_id in group_ids:
            group_results.append(
                self.evaluate_group_windows(
                    input_df_scaled=input_df_scaled,
                    target_df=target_df,
                    group_id=group_id,
                    input_seq_len=input_seq_len,
                    pred_seq_len=pred_seq_len,
                    horizon_gap_seq_len=horizon_gap_seq_len,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-window raw EMG prediction runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train next-window predictor")
    train_parser.add_argument("--input", help="Path to input dataframe file")
    train_parser.add_argument("--target", help="Path to target dataframe file")
    train_parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    train_parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--input-stage", default="raw")
    train_parser.add_argument("--target-stage", default="lp_10hz")
    train_parser.add_argument("--window-ms", type=float, default=250.0)
    train_parser.add_argument("--input-window-ms", type=float)
    train_parser.add_argument("--pred-window-ms", type=float)
    train_parser.add_argument("--horizon-ms", type=float, default=0.0)
    train_parser.add_argument("--seq-len", type=int)
    train_parser.add_argument("--input-seq-len", type=int)
    train_parser.add_argument("--pred-seq-len", type=int)
    train_parser.add_argument("--horizon-seq-len", type=int)
    train_parser.add_argument("--stride", type=int)
    train_parser.add_argument(
        "--model-type",
        default="lstm",
        help=f"Predictor backbone. Available: {', '.join(PREDICTOR_MODEL_TYPES)}",
    )
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--split-mode", choices=["group_random", "subject_cv"], default="subject_cv")
    train_parser.add_argument("--cv-fold", type=int, default=0)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    test_parser = subparsers.add_parser("test", help="Evaluate next-window predictor")
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--input", help="Path to input dataframe file")
    test_parser.add_argument("--target", help="Path to target dataframe file")
    test_parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    test_parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    test_parser.add_argument("--input-stage", default="raw")
    test_parser.add_argument("--target-stage", default="lp_10hz")
    test_parser.add_argument("--group-id", required=True)
    test_parser.add_argument("--output-dir", required=True)
    test_parser.add_argument("--horizon-ms", type=float, default=0.0)
    test_parser.add_argument("--horizon-seq-len", type=int)
    test_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_input_target_frames(args: argparse.Namespace, input_stage_arg: str, target_stage_arg: str):
    validate_stage_name(getattr(args, input_stage_arg))
    validate_stage_name(getattr(args, target_stage_arg))

    using_raw_dir = bool(args.raw_dir)
    using_stage_dir = bool(args.stage_dir)
    using_tables = bool(args.input) and bool(args.target)
    mode_count = sum([using_raw_dir, using_stage_dir, using_tables])
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --raw-dir, --stage-dir, or both --input and --target.")

    if using_raw_dir:
        raw_df, feature_columns, sampling_frequency = load_raw_emg_directory(args.raw_dir)
        stage_frames = build_emg_stage_frames(raw_df, feature_columns, sampling_frequency)
        input_df = resolve_stage_frame(getattr(args, input_stage_arg), stage_frames, feature_columns, sampling_frequency)
        target_df = resolve_stage_frame(
            getattr(args, target_stage_arg),
            stage_frames,
            feature_columns,
            sampling_frequency,
        )
        input_df, target_df, feature_columns = align_frames(input_df, target_df, feature_columns=feature_columns)
    elif using_stage_dir:
        input_df = load_stage_frame(args.stage_dir, getattr(args, input_stage_arg))
        target_df = load_stage_frame(args.stage_dir, getattr(args, target_stage_arg))
        input_df, target_df, feature_columns = align_frames(input_df, target_df)
    else:
        input_df = load_table(args.input)
        target_df = load_table(args.target)
        input_df, target_df, feature_columns = align_frames(input_df, target_df)

    return filter_finite_feature_columns(input_df, target_df, feature_columns)


def _estimate_seq_len_from_window_ms(input_df: pd.DataFrame, window_ms: float) -> int:
    seq_len, _ = estimate_window_size(input_df, window_ms=window_ms)
    return seq_len


def _default_stride_for_seq_len(seq_len: int) -> int:
    return max(1, int(round(seq_len * 0.5)))


def resolve_window_lengths(args: argparse.Namespace, input_df: pd.DataFrame) -> tuple[int, int, int, int]:
    input_seq_len = args.input_seq_len if args.input_seq_len is not None else args.seq_len
    pred_seq_len = args.pred_seq_len if args.pred_seq_len is not None else args.seq_len
    horizon_gap_seq_len = args.horizon_seq_len
    stride = args.stride

    if input_seq_len is None:
        input_window_ms = args.input_window_ms if args.input_window_ms is not None else args.window_ms
        input_seq_len = _estimate_seq_len_from_window_ms(input_df, input_window_ms)

    if pred_seq_len is None:
        if args.pred_window_ms is not None:
            pred_seq_len = _estimate_seq_len_from_window_ms(input_df, args.pred_window_ms)
        elif args.input_seq_len is not None or args.input_window_ms is not None:
            pred_seq_len = input_seq_len
        else:
            pred_seq_len = _estimate_seq_len_from_window_ms(input_df, args.window_ms)

    if stride is None:
        if args.input_window_ms is not None:
            _, stride = estimate_window_size(input_df, window_ms=args.input_window_ms)
        elif args.seq_len is None and args.input_seq_len is None:
            _, stride = estimate_window_size(input_df, window_ms=args.window_ms)
        else:
            stride = _default_stride_for_seq_len(input_seq_len)

    if horizon_gap_seq_len is None:
        horizon_ms = float(getattr(args, "horizon_ms", 0.0) or 0.0)
        if horizon_ms > 0.0:
            horizon_gap_seq_len = _estimate_seq_len_from_window_ms(input_df, horizon_ms)
        else:
            horizon_gap_seq_len = 0

    return input_seq_len, pred_seq_len, horizon_gap_seq_len, stride


def should_share_scaler(args: argparse.Namespace) -> bool:
    input_stage = getattr(args, "input_stage", None)
    target_stage = getattr(args, "target_stage", None)
    if input_stage is not None and target_stage is not None:
        return normalize_stage_name(input_stage) == normalize_stage_name(target_stage)

    input_path = getattr(args, "input", None)
    target_path = getattr(args, "target", None)
    if input_path and target_path:
        return Path(input_path).resolve() == Path(target_path).resolve()
    return False


def train_command(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    input_df, target_df, feature_columns, dropped_columns = load_input_target_frames(args, "input_stage", "target_stage")
    if dropped_columns:
        print("Dropped non-finite feature columns:", ", ".join(dropped_columns))

    input_seq_len, pred_seq_len, horizon_gap_seq_len, stride = resolve_window_lengths(args, input_df)
    print(
        "Resolved window setup: "
        f"input_seq_len={input_seq_len}, pred_seq_len={pred_seq_len}, "
        f"horizon_gap_seq_len={horizon_gap_seq_len}, stride={stride}"
    )

    if args.split_mode == "subject_cv":
        train_groups, val_groups, test_groups, train_subjects, val_subject, test_subject = split_subject_cv(
            input_df["GROUP_ID"],
            fold_index=args.cv_fold,
        )
        print(f"Subject CV fold {args.cv_fold % 7}")
        print(f"Train subjects: {', '.join(train_subjects)}")
        print(f"Validation subject: {val_subject}")
        print(f"Test subject: {test_subject}")
    else:
        train_groups, val_groups, test_groups = split_group_ids(
            input_df["GROUP_ID"],
            SplitConfig(seed=args.seed),
        )

    input_scaler = DataFrameScaler(feature_columns).fit(input_df[input_df["GROUP_ID"].isin(train_groups)])
    if should_share_scaler(args):
        target_scaler = input_scaler
    else:
        target_scaler = DataFrameScaler(feature_columns).fit(target_df[target_df["GROUP_ID"].isin(train_groups)])

    input_scaled = input_scaler.transform(input_df)
    target_scaled = target_scaler.transform(target_df)

    train_ds = NextWindowDataset(
        input_scaled,
        target_scaled,
        feature_columns,
        input_seq_len,
        pred_seq_len,
        horizon_gap_seq_len,
        stride,
        train_groups,
    )
    val_ds = NextWindowDataset(
        input_scaled,
        target_scaled,
        feature_columns,
        input_seq_len,
        pred_seq_len,
        horizon_gap_seq_len,
        stride,
        val_groups,
    )
    test_ds = NextWindowDataset(
        input_scaled,
        target_scaled,
        feature_columns,
        input_seq_len,
        pred_seq_len,
        horizon_gap_seq_len,
        stride,
        test_groups,
    )

    generator = make_generator(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_config = PredictorConfig(
        num_channels=len(feature_columns),
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        prediction_seq_len=pred_seq_len,
    )
    model = LSTMNextWindowPredictor(model_config)
    trainer = NextWindowTrainer(
        model,
        PredictionTrainerConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            device=args.device,
        ),
    )
    best_model, history = trainer.fit(train_loader, val_loader)
    test_loss = trainer.evaluate(test_loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "next_window_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "model_config": model_config.__dict__,
            "metadata": checkpoint_metadata(
                feature_columns=feature_columns,
                window_size=input_seq_len,
                stride=stride,
                latent_dim=args.hidden_dim,
                split_config=SplitConfig(seed=args.seed),
            )
            | {
                "input_window_size": input_seq_len,
                "prediction_window_size": pred_seq_len,
                "horizon_gap_seq_len": horizon_gap_seq_len,
                "input_stage": args.input_stage,
                "target_stage": args.target_stage,
                "split_mode": args.split_mode,
                "cv_fold": args.cv_fold,
                "shared_scaler": should_share_scaler(args),
                "dropped_columns": dropped_columns,
            },
            "history": history,
            "test_loss": test_loss,
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
        },
        checkpoint_path,
    )
    save_json(
        output_dir / "train_summary.json",
        {
            "model_type": model_config.model_type,
            "architecture": model_config.architecture,
            "input_seq_len": input_seq_len,
            "pred_seq_len": pred_seq_len,
            "horizon_gap_seq_len": horizon_gap_seq_len,
            "stride": stride,
            "shared_scaler": should_share_scaler(args),
            "history": history,
            "test_loss": test_loss,
        },
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Test loss: {test_loss:.6f}")


def test_command(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_config = load_predictor_config(checkpoint["model_config"], checkpoint.get("model_state_dict"))
    feature_columns = checkpoint["metadata"]["feature_columns"]
    input_seq_len = int(checkpoint["metadata"].get("input_window_size", checkpoint["metadata"]["window_size"]))
    pred_seq_len = int(checkpoint["metadata"].get("prediction_window_size", checkpoint["metadata"]["window_size"]))
    horizon_gap_seq_len = int(checkpoint["metadata"].get("horizon_gap_seq_len", 0))

    input_df, target_df, _, _ = load_input_target_frames(args, "input_stage", "target_stage")
    input_df, target_df, _ = align_frames(input_df, target_df, feature_columns=feature_columns)
    input_scaled = checkpoint["input_scaler"].transform(input_df)

    model = LSTMNextWindowPredictor(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    tester = NextWindowTester(
        model=model,
        target_scaler=checkpoint["target_scaler"],
        feature_columns=feature_columns,
        device=args.device,
    )
    result = tester.predict_next_window(
        input_scaled,
        target_df,
        args.group_id,
        input_seq_len=input_seq_len,
        pred_seq_len=pred_seq_len,
        horizon_gap_seq_len=horizon_gap_seq_len,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = ReconstructionVisualizer(feature_columns)
    visualizer.plot_window(
        result.target_window,
        result.pred_window,
        channel_index=0,
        title=f"Next-window prediction for {args.group_id}",
        save_path=output_dir / f"{args.group_id}_next_window_prediction.png",
    )
    save_json(
        output_dir / f"{args.group_id}_metrics.json",
        result.metrics,
    )
    summary = result.metrics["summary"]
    print(f"NRMSE: {summary['nrmse']:.6f}")
    print(f"CCC: {summary['ccc']:.6f}")
    print(f"Pearson r: {summary['pearson_r']:.6f}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
