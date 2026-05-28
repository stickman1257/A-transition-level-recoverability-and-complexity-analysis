from __future__ import annotations

import json
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
except ImportError:  # pragma: no cover - optional for preprocessing-only workflows
    torch = None


REQUIRED_COLUMNS = ("GROUP_ID", "TIME")
NON_EMG_COLUMNS = {"FS,uV", "Markers", "MarkerNames", "Activities", "ActivityNames"}
NON_FEATURE_COLUMNS = {
    "MARKERS",
    "MARKERNAMES",
    "ACTIVITIES",
    "ACTIVITYNAMES",
    "FS",
    "FS_std",
    "Prefix",
    "Task_No",
}
AVAILABLE_STAGE_NAMES = (
    "raw",
    "dc_offset",
    "bandpass",
    "notch",
    "rectified",
    "lp_6hz",
    "lp_10hz",
)
STAGE_ALIASES = {
    "bandpass+notch": "notch",
    "bandpass_notch": "notch",
    "bandpass-notch": "notch",
    "lp5": "lp_5hz",
    "lp6": "lp_6hz",
    "lp10": "lp_10hz",
}


@dataclass
class SplitConfig:
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total}.")


def normalize_stage_name(stage_name: str) -> str:
    canonical = str(stage_name).strip().lower().replace(" ", "")
    return STAGE_ALIASES.get(canonical, canonical)


class DataFrameScaler:
    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.feature_columns = list(feature_columns)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "DataFrameScaler":
        self.scaler.fit(df[self.feature_columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return out

    def inverse_transform_array(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(arr)


class EMGSignalProcessor:
    def __init__(self, fs: float) -> None:
        self.fs = fs

    def remove_dc_offset(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
        out = df.copy()
        for col in feature_columns:
            out[col] = (
                out.groupby("GROUP_ID")[col]
                .transform(lambda x: x - x.mean())
                .astype(np.float32)
            )
        return out

    def bandpass_filter(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        lowcut_hz: float = 20.0,
        highcut_hz: float = 450.0,
        order: int = 4,
    ) -> pd.DataFrame:
        nyquist = 0.5 * self.fs
        if not (0 < lowcut_hz < highcut_hz < nyquist):
            raise ValueError(
                f"Bandpass frequencies must satisfy 0 < lowcut < highcut < Nyquist ({nyquist})."
            )
        sos = butter(order, [lowcut_hz / nyquist, highcut_hz / nyquist], btype="band", output="sos")
        out = df.copy()
        for col in feature_columns:
            out[col] = (
                out.groupby("GROUP_ID")[col]
                .transform(lambda x: sosfiltfilt(sos, x.to_numpy()))
                .astype(np.float32)
            )
        return out

    def notch_filter(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        base_freq_hz: float = 60.0,
        quality_factor: float = 30.0,
        harmonics: int | None = None,
    ) -> pd.DataFrame:
        nyquist = 0.5 * self.fs
        if base_freq_hz <= 0 or base_freq_hz >= nyquist:
            raise ValueError(f"Notch base frequency must be between 0 and Nyquist ({nyquist}).")

        if harmonics is None:
            harmonics = int(nyquist // base_freq_hz)

        filters: list[tuple[np.ndarray, np.ndarray]] = []
        for multiplier in range(1, harmonics + 1):
            freq = base_freq_hz * multiplier
            if freq >= nyquist:
                break
            filters.append(iirnotch(w0=freq, Q=quality_factor, fs=self.fs))

        if not filters:
            raise ValueError("No valid notch frequencies were created.")

        out = df.copy()
        for col in feature_columns:
            def apply_notches(x: pd.Series) -> np.ndarray:
                values = x.to_numpy(dtype=np.float64)
                for b, a in filters:
                    values = filtfilt(b, a, values)
                return values

            out[col] = out.groupby("GROUP_ID")[col].transform(apply_notches).astype(np.float32)
        return out

    def rectify(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
        out = df.copy()
        out[list(feature_columns)] = out[list(feature_columns)].abs()
        return out

    def lowpass_envelope(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        cutoff_hz: float,
        order: int = 4,
    ) -> pd.DataFrame:
        b, a = butter(order, cutoff_hz / (0.5 * self.fs), btype="low", analog=False)
        out = df.copy()
        for col in feature_columns:
            out[col] = (
                out.groupby("GROUP_ID")[col]
                .transform(lambda x: filtfilt(b, a, x.to_numpy()))
                .astype(np.float32)
            )
        return out


def seed_everything(seed: int) -> None:
    if torch is None:
        raise ImportError("torch is required for seed_everything.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Keep deterministic settings where PyTorch supports them, but do not
        # crash on CUDA ops such as linear upsampling that lack deterministic
        # backward implementations.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def make_generator(seed: int) -> torch.Generator:
    if torch is None:
        raise ImportError("torch is required for make_generator.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_pickle(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    validate_dataframe(df)
    return [col for col in df.columns if col not in REQUIRED_COLUMNS and col not in NON_FEATURE_COLUMNS]


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def align_frames(
    target_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    validate_dataframe(target_df)
    validate_dataframe(cond_df)

    if feature_columns is None:
        target_features = set(infer_feature_columns(target_df))
        cond_features = set(infer_feature_columns(cond_df))
        feature_columns = sorted(target_features & cond_features)
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No shared EMG feature columns found.")

    base_columns = ["GROUP_ID", "TIME", *feature_columns]
    left = target_df[base_columns].copy()
    right = cond_df[base_columns].copy()

    left = left.sort_values(["GROUP_ID", "TIME"]).reset_index(drop=True)
    right = right.sort_values(["GROUP_ID", "TIME"]).reset_index(drop=True)

    if len(left) != len(right):
        raise ValueError("Target and condition data lengths do not match.")
    if not left[["GROUP_ID", "TIME"]].equals(right[["GROUP_ID", "TIME"]]):
        raise ValueError("Target and condition rows are not aligned on GROUP_ID/TIME.")

    return left, right, list(feature_columns)


def filter_finite_feature_columns(
    target_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    valid_columns: list[str] = []
    dropped_columns: list[str] = []

    for col in feature_columns:
        target_values = target_df[col].to_numpy(dtype=np.float64)
        cond_values = cond_df[col].to_numpy(dtype=np.float64)
        if np.isfinite(target_values).all() and np.isfinite(cond_values).all():
            valid_columns.append(col)
        else:
            dropped_columns.append(col)

    if not valid_columns:
        raise ValueError("No finite feature columns remain after filtering NaN/inf values.")

    base_columns = ["GROUP_ID", "TIME", *valid_columns]
    return (
        target_df[base_columns].copy(),
        cond_df[base_columns].copy(),
        valid_columns,
        dropped_columns,
    )


def split_group_ids(group_ids: Iterable[str], config: SplitConfig) -> tuple[list[str], list[str], list[str]]:
    config.validate()
    unique_groups = np.array(sorted(set(group_ids)))
    if len(unique_groups) < 3:
        raise ValueError("Need at least 3 unique GROUP_ID values for train/val/test split.")

    gss1 = GroupShuffleSplit(n_splits=1, train_size=config.train_size, random_state=config.seed)
    train_idx, temp_idx = next(gss1.split(unique_groups, groups=unique_groups))

    train_groups = unique_groups[train_idx]
    temp_groups = unique_groups[temp_idx]

    relative_val = config.val_size / (config.val_size + config.test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=config.seed)
    val_idx, test_idx = next(gss2.split(temp_groups, groups=temp_groups))

    val_groups = temp_groups[val_idx]
    test_groups = temp_groups[test_idx]
    return train_groups.tolist(), val_groups.tolist(), test_groups.tolist()


def group_id_to_subject_id(group_id: str) -> str:
    if "_" not in group_id:
        return group_id
    return group_id.split("_", 1)[0]


def split_subject_cv(
    group_ids: Iterable[str],
    fold_index: int = 0,
) -> tuple[list[str], list[str], list[str], list[str], str, str]:
    unique_groups = np.array(sorted(set(group_ids)))
    if len(unique_groups) == 0:
        raise ValueError("No GROUP_ID values were provided.")

    subject_to_groups: dict[str, list[str]] = {}
    for group_id in unique_groups:
        subject_id = group_id_to_subject_id(str(group_id))
        subject_to_groups.setdefault(subject_id, []).append(str(group_id))

    subjects = sorted(subject_to_groups)
    if len(subjects) != 7:
        raise ValueError(f"Subject CV expects 7 subjects, found {len(subjects)}: {subjects}")

    normalized_fold = fold_index % len(subjects)
    test_subject = subjects[normalized_fold]
    val_subject = subjects[(normalized_fold + 1) % len(subjects)]
    train_subjects = [subject for subject in subjects if subject not in {val_subject, test_subject}]

    train_groups = [group_id for subject in train_subjects for group_id in subject_to_groups[subject]]
    val_groups = list(subject_to_groups[val_subject])
    test_groups = list(subject_to_groups[test_subject])
    return train_groups, val_groups, test_groups, train_subjects, val_subject, test_subject


def estimate_window_size(
    df: pd.DataFrame,
    group_col: str = "GROUP_ID",
    time_col: str = "TIME",
    window_ms: float = 250.0,
    overlap: float = 0.5,
) -> tuple[int, int]:
    dt = (
        df.groupby(group_col)[time_col]
        .apply(lambda s: np.median(np.diff(s.to_numpy())))
        .replace(0, np.nan)
        .dropna()
    )
    if dt.empty:
        raise ValueError("Could not estimate sampling interval from TIME column.")
    fs = 1.0 / float(dt.median())
    window_samples = max(2, int(round(fs * (window_ms / 1000.0))))
    stride = max(1, int(round(window_samples * (1.0 - overlap))))
    return window_samples, stride


def estimate_sampling_frequency(
    df: pd.DataFrame,
    group_col: str = "GROUP_ID",
    time_col: str = "TIME",
) -> float:
    dt = (
        df.groupby(group_col)[time_col]
        .apply(lambda s: np.median(np.diff(s.to_numpy())))
        .replace(0, np.nan)
        .dropna()
    )
    if dt.empty:
        raise ValueError("Could not estimate sampling interval from TIME column.")
    return float(1.0 / dt.median())


def checkpoint_metadata(
    *,
    feature_columns: Sequence[str],
    window_size: int,
    stride: int,
    latent_dim: int,
    split_config: SplitConfig,
) -> dict:
    return {
        "feature_columns": list(feature_columns),
        "window_size": window_size,
        "stride": stride,
        "latent_dim": latent_dim,
        "split_config": asdict(split_config),
    }


def _extract_group_parts(csv_path: Path, raw_root: Path) -> tuple[str, str, str]:
    relative = csv_path.relative_to(raw_root)
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected raw CSV path layout: {csv_path}")
    trial = parts[-3]
    subject = parts[-4] if len(parts) >= 4 else raw_root.name
    cohort = parts[-5] if len(parts) >= 5 else raw_root.parent.name
    return cohort, subject, trial


def _load_raw_emg_csv(csv_path: Path, raw_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=None)
    if len(raw) < 4:
        raise ValueError(f"Raw CSV is too short: {csv_path}")

    frequency = float(raw.iloc[1, 1])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"Invalid sampling frequency in {csv_path}: {raw.iloc[0, 1]}")

    header = [str(value).replace("\ufeff", "").strip() for value in raw.iloc[3].fillna("").tolist()]
    data = raw.iloc[4:].reset_index(drop=True).copy()
    data.columns = header

    if "Time,s" not in data.columns:
        raise ValueError(f"Missing Time,s column in {csv_path}")

    emg_columns = [col for col in header if col.endswith(",uV") and col not in NON_EMG_COLUMNS]
    if not emg_columns:
        raise ValueError(f"No EMG columns found in {csv_path}")

    cohort, subject, trial = _extract_group_parts(csv_path, raw_root)
    group_id = f"{subject}_{trial}_{csv_path.stem}"

    out = data[["Time,s", *emg_columns]].copy()
    out = out.rename(columns={"Time,s": "TIME"})
    out["TIME"] = pd.to_numeric(out["TIME"], errors="coerce")
    for col in emg_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["TIME", *emg_columns]).reset_index(drop=True)
    out.insert(0, "GROUP_ID", group_id)
    out.attrs["sampling_frequency"] = frequency
    out.attrs["cohort"] = cohort
    out.attrs["subject"] = subject
    out.attrs["trial"] = trial
    return out


def load_raw_emg_directory(raw_dir: str | Path) -> tuple[pd.DataFrame, list[str], float]:
    raw_root = Path(raw_dir)
    csv_files = sorted(raw_root.rglob("csv_output/*.csv"))
    if not csv_files:
        raise ValueError(f"No raw CSV files found under {raw_root}")

    frames: list[pd.DataFrame] = []
    frequencies: list[float] = []
    for csv_path in csv_files:
        frame = _load_raw_emg_csv(csv_path, raw_root)
        frames.append(frame)
        frequencies.append(float(frame.attrs["sampling_frequency"]))

    if not np.allclose(frequencies, frequencies[0]):
        raise ValueError("Sampling frequencies are not consistent across raw CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    feature_columns = infer_feature_columns(combined)
    return combined, feature_columns, float(frequencies[0])


def build_emg_stage_frames(
    raw_df: pd.DataFrame,
    feature_columns: Sequence[str],
    sampling_frequency: float,
) -> dict[str, pd.DataFrame]:
    processor = EMGSignalProcessor(fs=sampling_frequency)

    stage_frames: dict[str, pd.DataFrame] = {}
    stage_frames["raw"] = raw_df.copy()
    stage_frames["dc_offset"] = processor.remove_dc_offset(stage_frames["raw"], feature_columns)
    stage_frames["bandpass"] = processor.bandpass_filter(stage_frames["dc_offset"], feature_columns)
    stage_frames["notch"] = processor.notch_filter(stage_frames["bandpass"], feature_columns)
    stage_frames["rectified"] = processor.rectify(stage_frames["notch"], feature_columns)
    stage_frames["lp_6hz"] = processor.lowpass_envelope(stage_frames["rectified"], feature_columns, cutoff_hz=6.0)
    stage_frames["lp_10hz"] = processor.lowpass_envelope(
        stage_frames["rectified"],
        feature_columns,
        cutoff_hz=10.0,
    )
    return stage_frames


def parse_lowpass_stage_name(stage_name: str) -> float | None:
    stage_name = normalize_stage_name(stage_name)
    match = re.fullmatch(r"lp_(\d+(?:\.\d+)?)hz", stage_name)
    if match is None:
        return None
    return float(match.group(1))


def validate_stage_name(stage_name: str) -> None:
    stage_name = normalize_stage_name(stage_name)
    if stage_name in AVAILABLE_STAGE_NAMES:
        return
    cutoff = parse_lowpass_stage_name(stage_name)
    if cutoff is not None and cutoff > 0:
        return
    raise ValueError(
        f"Unsupported stage name: {stage_name}. "
        f"Use one of {list(AVAILABLE_STAGE_NAMES)} or lp_<cutoff>hz such as lp_20hz."
    )


def resolve_stage_frame(
    stage_name: str,
    stage_frames: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    sampling_frequency: float,
) -> pd.DataFrame:
    stage_name = normalize_stage_name(stage_name)
    validate_stage_name(stage_name)
    if stage_name in stage_frames:
        return stage_frames[stage_name]

    cutoff = parse_lowpass_stage_name(stage_name)
    if cutoff is None:
        raise ValueError(f"Stage frame is not available: {stage_name}")

    if "rectified" not in stage_frames:
        raise ValueError("rectified stage is required to build lowpass envelope stages.")

    processor = EMGSignalProcessor(fs=sampling_frequency)
    return processor.lowpass_envelope(stage_frames["rectified"], feature_columns, cutoff_hz=cutoff)


def load_stage_frame(stage_dir: str | Path, stage_name: str) -> pd.DataFrame:
    stage_name = normalize_stage_name(stage_name)
    validate_stage_name(stage_name)
    stage_path = Path(stage_dir) / f"{stage_name}_df.parquet"
    if stage_path.exists():
        return load_table(stage_path)

    cutoff = parse_lowpass_stage_name(stage_name)
    if cutoff is not None:
        rectified_path = Path(stage_dir) / "rectified_df.parquet"
        if rectified_path.exists():
            rectified_df = load_table(rectified_path)
            feature_columns = infer_feature_columns(rectified_df)
            sampling_frequency = estimate_sampling_frequency(rectified_df)
            processor = EMGSignalProcessor(fs=sampling_frequency)
            return processor.lowpass_envelope(rectified_df, feature_columns, cutoff_hz=cutoff)

    raise FileNotFoundError(f"Stage dataframe not found: {stage_path}")


def save_table(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return
    raise ValueError(f"Unsupported output format: {path}")
