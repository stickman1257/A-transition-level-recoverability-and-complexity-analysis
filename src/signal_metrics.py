from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.signal import coherence, welch
from scipy.stats import pearsonr


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={arr.shape}")
    return arr


def nrmse(target: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    denom = np.max(target) - np.min(target)
    return float(rmse / (denom + eps))


def mse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean((target - pred) ** 2))


def concordance_correlation_coefficient(target: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mean_t = np.mean(target)
    mean_p = np.mean(pred)
    var_t = np.var(target)
    var_p = np.var(pred)
    cov = np.mean((target - mean_t) * (pred - mean_p))
    return float((2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + eps))


def pearson_correlation(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if np.std(target) == 0 or np.std(pred) == 0:
        return float("nan")
    corr, _ = pearsonr(target, pred)
    return float(corr)


def rms_relative_error(target: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    rms_target = np.sqrt(np.mean(target**2))
    rms_pred = np.sqrt(np.mean(pred**2))
    return float(abs(rms_pred - rms_target) / (abs(rms_target) + eps))


def mae(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(target - pred)))


def _prepare_complexity_signal(signal: np.ndarray, max_points: int | None = None) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if max_points is not None and max_points > 0 and arr.size > max_points:
        indices = np.linspace(0, arr.size - 1, num=max_points, dtype=np.int64)
        arr = arr[indices]
    return arr


def permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
    max_points: int | None = 3000,
) -> float:
    arr = _prepare_complexity_signal(signal, max_points=max_points)
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    pattern_count = arr.size - (order - 1) * delay
    if pattern_count <= 0:
        return float("nan")

    embedded = np.stack([arr[idx * delay : idx * delay + pattern_count] for idx in range(order)], axis=1)
    patterns = np.argsort(embedded, axis=1, kind="mergesort")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    if normalize:
        entropy /= np.log2(math.factorial(order))
    return float(entropy)


def sample_entropy(
    signal: np.ndarray,
    order: int = 2,
    r_ratio: float = 0.2,
    max_points: int | None = 1200,
) -> float:
    arr = _prepare_complexity_signal(signal, max_points=max_points)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if arr.size <= order + 1:
        return float("nan")

    std = float(np.std(arr))
    if not np.isfinite(std):
        return float("nan")
    if std == 0.0:
        return 0.0
    tolerance = r_ratio * std

    def _count_matches(template_order: int) -> int:
        n_templates = arr.size - template_order + 1
        if n_templates <= 1:
            return 0
        templates = np.asarray(
            [arr[idx : idx + template_order] for idx in range(n_templates)],
            dtype=np.float64,
        )
        match_count = 0
        for idx in range(n_templates - 1):
            distances = np.max(np.abs(templates[idx + 1 :] - templates[idx]), axis=1)
            match_count += int(np.sum(distances <= tolerance))
        return match_count

    matches_m = _count_matches(order)
    matches_m1 = _count_matches(order + 1)
    if matches_m == 0 or matches_m1 == 0:
        return float("nan")
    return float(-np.log(matches_m1 / matches_m))


def median_frequency(signal: np.ndarray, fs: float) -> float:
    freqs, psd = welch(np.asarray(signal, dtype=np.float64), fs=fs, nperseg=min(256, len(signal)))
    cumulative = np.cumsum(psd)
    if cumulative[-1] == 0:
        return 0.0
    idx = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def median_frequency_relative_error(target: np.ndarray, pred: np.ndarray, fs: float, eps: float = 1e-8) -> float:
    mf_target = median_frequency(target, fs)
    mf_pred = median_frequency(pred, fs)
    return float(abs(mf_pred - mf_target) / (abs(mf_target) + eps))


def psd_distance(target: np.ndarray, pred: np.ndarray, fs: float, eps: float = 1e-8) -> float:
    freqs_t, psd_t = welch(np.asarray(target, dtype=np.float64), fs=fs, nperseg=min(256, len(target)))
    freqs_p, psd_p = welch(np.asarray(pred, dtype=np.float64), fs=fs, nperseg=min(256, len(pred)))
    if len(freqs_t) != len(freqs_p):
        raise ValueError("PSD frequency grids do not match.")
    return float(np.linalg.norm(psd_t - psd_p) / (np.linalg.norm(psd_t) + eps))


def mean_magnitude_squared_coherence(
    target: np.ndarray,
    pred: np.ndarray,
    fs: float,
    min_freq_hz: float | None = None,
    max_freq_hz: float | None = None,
    eps: float = 1e-8,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if target.shape != pred.shape:
        raise ValueError(f"Target/pred shapes do not match: {target.shape} vs {pred.shape}")
    if len(target) < 4:
        return float("nan")
    if np.std(target) <= eps or np.std(pred) <= eps:
        return float("nan")

    freqs, coh = coherence(target, pred, fs=fs, nperseg=min(256, len(target)))
    valid = np.isfinite(coh)
    if min_freq_hz is not None:
        valid &= freqs >= float(min_freq_hz)
    if max_freq_hz is not None:
        valid &= freqs <= float(max_freq_hz)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(coh[valid]))


def compute_signal_metrics(target: np.ndarray, pred: np.ndarray, fs: float, channel_names: Sequence[str] | None = None) -> dict:
    target_2d = _as_2d(target)
    pred_2d = _as_2d(pred)
    if target_2d.shape != pred_2d.shape:
        raise ValueError(f"Target/pred shapes do not match: {target_2d.shape} vs {pred_2d.shape}")

    n_channels = target_2d.shape[1]
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]
    channel_names = list(channel_names)

    per_channel: dict[str, dict[str, float]] = {}
    keys = [
        "mse",
        "nrmse",
        "ccc",
        "pearson_r",
        "mae",
        "psd_distance",
        "rms_relative_error",
        "median_frequency_relative_error",
    ]
    aggregate = {key: [] for key in keys}

    for idx, channel_name in enumerate(channel_names):
        t = target_2d[:, idx]
        p = pred_2d[:, idx]
        metrics = {
            "mse": mse(t, p),
            "nrmse": nrmse(t, p),
            "ccc": concordance_correlation_coefficient(t, p),
            "pearson_r": pearson_correlation(t, p),
            "mae": mae(t, p),
            "psd_distance": psd_distance(t, p, fs),
            "rms_relative_error": rms_relative_error(t, p),
            "median_frequency_relative_error": median_frequency_relative_error(t, p, fs),
        }
        per_channel[channel_name] = metrics
        for key, value in metrics.items():
            if np.isfinite(value):
                aggregate[key].append(value)

    summary = {key: float(np.mean(values)) if values else float("nan") for key, values in aggregate.items()}
    return {"summary": summary, "per_channel": per_channel}


def compute_coherence_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    fs: float,
    channel_names: Sequence[str] | None = None,
    min_freq_hz: float | None = None,
    max_freq_hz: float | None = None,
) -> dict:
    target_2d = _as_2d(target)
    pred_2d = _as_2d(pred)
    if target_2d.shape != pred_2d.shape:
        raise ValueError(f"Target/pred shapes do not match: {target_2d.shape} vs {pred_2d.shape}")

    n_channels = target_2d.shape[1]
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]
    channel_names = list(channel_names)

    metric_key = "scp_mean_coherence"
    per_channel: dict[str, dict[str, float]] = {}
    aggregate: list[float] = []

    for idx, channel_name in enumerate(channel_names):
        value = mean_magnitude_squared_coherence(
            target_2d[:, idx],
            pred_2d[:, idx],
            fs=fs,
            min_freq_hz=min_freq_hz,
            max_freq_hz=max_freq_hz,
        )
        per_channel[channel_name] = {metric_key: value}
        if np.isfinite(value):
            aggregate.append(value)

    summary = {metric_key: float(np.mean(aggregate)) if aggregate else float("nan")}
    return {"summary": summary, "per_channel": per_channel}


def compute_complexity_metrics(
    signal: np.ndarray,
    channel_names: Sequence[str] | None = None,
    *,
    pe_order: int = 3,
    pe_delay: int = 1,
    pe_max_points: int | None = 3000,
    sampen_order: int = 2,
    sampen_r_ratio: float = 0.2,
    sampen_max_points: int | None = 1200,
) -> dict:
    signal_2d = _as_2d(signal)
    n_channels = signal_2d.shape[1]
    if channel_names is None:
        channel_names = [f"ch_{idx}" for idx in range(n_channels)]
    channel_names = list(channel_names)

    per_channel: dict[str, dict[str, float]] = {}
    aggregate = {"pe": [], "sampen": []}
    for idx, channel_name in enumerate(channel_names):
        seq = signal_2d[:, idx]
        metrics = {
            "pe": permutation_entropy(
                seq,
                order=pe_order,
                delay=pe_delay,
                normalize=True,
                max_points=pe_max_points,
            ),
            "sampen": sample_entropy(
                seq,
                order=sampen_order,
                r_ratio=sampen_r_ratio,
                max_points=sampen_max_points,
            ),
        }
        per_channel[channel_name] = metrics
        for key, value in metrics.items():
            if np.isfinite(value):
                aggregate[key].append(value)

    summary = {key: float(np.mean(values)) if values else float("nan") for key, values in aggregate.items()}
    return {"summary": summary, "per_channel": per_channel}


def aggregate_metric_payloads(metric_payloads: Sequence[dict]) -> dict:
    if not metric_payloads:
        raise ValueError("metric_payloads must not be empty.")

    summary_keys = sorted({key for payload in metric_payloads for key in payload.get("summary", {})})
    channel_names = sorted({channel for payload in metric_payloads for channel in payload.get("per_channel", {})})

    summary_mean: dict[str, float] = {}
    summary_std: dict[str, float] = {}
    for key in summary_keys:
        values = _finite_values(payload.get("summary", {}).get(key, float("nan")) for payload in metric_payloads)
        summary_mean[key], summary_std[key] = _mean_std(values)

    per_channel_mean: dict[str, dict[str, float]] = {}
    per_channel_std: dict[str, dict[str, float]] = {}
    for channel_name in channel_names:
        metric_keys = sorted(
            {
                key
                for payload in metric_payloads
                for key in payload.get("per_channel", {}).get(channel_name, {})
            }
        )
        per_channel_mean[channel_name] = {}
        per_channel_std[channel_name] = {}
        for key in metric_keys:
            values = _finite_values(
                payload.get("per_channel", {}).get(channel_name, {}).get(key, float("nan"))
                for payload in metric_payloads
            )
            per_channel_mean[channel_name][key], per_channel_std[channel_name][key] = _mean_std(values)

    return {
        "num_items": len(metric_payloads),
        "summary_mean": summary_mean,
        "summary_std": summary_std,
        "per_channel_mean": per_channel_mean,
        "per_channel_std": per_channel_std,
    }


def filter_metric_payload(payload: dict, allowed_metrics: Sequence[str]) -> dict:
    allowed = set(allowed_metrics)
    filtered: dict = {}
    for key, value in payload.items():
        if key in {"summary", "summary_mean", "summary_std"} and isinstance(value, dict):
            filtered[key] = {metric: metric_value for metric, metric_value in value.items() if metric in allowed}
            continue
        if key in {"per_channel", "per_channel_mean", "per_channel_std"} and isinstance(value, dict):
            filtered[key] = {
                channel_name: {
                    metric: metric_value
                    for metric, metric_value in channel_metrics.items()
                    if metric in allowed
                }
                for channel_name, channel_metrics in value.items()
            }
            continue
        if key == "window_summaries" and isinstance(value, list):
            filtered[key] = [
                {metric: metric_value for metric, metric_value in item.items() if metric == "start_idx" or metric in allowed}
                for item in value
            ]
            continue
        if key in {"groups", "folds"} and isinstance(value, list):
            filtered[key] = [
                filter_metric_payload(item, allowed)
                if isinstance(item, dict)
                else item
                for item in value
            ]
            continue
        filtered[key] = value
    return filtered


def _finite_values(values: Sequence[float] | np.ndarray | object) -> list[float]:
    out: list[float] = []
    for value in values:
        value = float(value)
        if np.isfinite(value):
            out.append(value)
    return out


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))
