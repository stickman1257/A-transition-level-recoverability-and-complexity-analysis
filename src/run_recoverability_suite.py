from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import (
    AVAILABLE_RECONSTRUCTION_MODELS,
    ReconstructionModelConfig,
    build_reconstruction_model,
    normalize_reconstruction_model_type,
)
from run import train_command
from signal_metrics import aggregate_metric_payloads, filter_metric_payload
from test import ReconstructionTester
from utils import (
    align_frames,
    build_emg_stage_frames,
    load_raw_emg_directory,
    load_stage_frame,
    normalize_stage_name,
    resolve_stage_frame,
    save_json,
    split_subject_cv,
    validate_stage_name,
)


DEFAULT_STAGE_PAIRS = (
    "notch->bandpass",
    "rectified->notch",
    "lp_10hz->rectified",
)
RECOVERABILITY_METRICS = (
    "nrmse",
    "ccc",
    "pearson_r",
    "psd_distance",
    "rms_relative_error",
    "median_frequency_relative_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recoverability reconstruction experiments for multiple stage pairs.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument(
        "--stage-pairs",
        nargs="*",
        default=list(DEFAULT_STAGE_PAIRS),
        help="Pairs in the form condition->target, e.g. notch->bandpass",
    )
    parser.add_argument("--output-dir", required=True, help="Base directory for recoverability outputs")
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument(
        "--model-type",
        default="conditional_unet",
        help=f"Reconstruction backbone. Available: {', '.join(AVAILABLE_RECONSTRUCTION_MODELS)}",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-warmup-epochs", type=int, default=15)
    parser.add_argument("--l1-weight", type=float, default=0.5)
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip training when the fold checkpoint already exists and only run held-out evaluation.",
    )
    parser.add_argument(
        "--save-window-summaries",
        action="store_true",
        help="Include per-window summaries in the saved held-out metrics JSON.",
    )
    return parser.parse_args()


def validate_mode(args: argparse.Namespace) -> None:
    mode_count = sum([bool(args.stage_dir), bool(args.raw_dir)])
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --stage-dir or --raw-dir.")


def parse_stage_pair(raw_pair: str) -> tuple[str, str]:
    separator = "->" if "->" in raw_pair else ":"
    if separator not in raw_pair:
        raise ValueError(f"Invalid stage pair: {raw_pair}. Use condition->target.")
    condition_stage, target_stage = [normalize_stage_name(part) for part in raw_pair.split(separator, 1)]
    validate_stage_name(condition_stage)
    validate_stage_name(target_stage)
    return condition_stage, target_stage


def pair_label(condition_stage: str, target_stage: str) -> str:
    return f"{condition_stage}_to_{target_stage}".replace(".", "_")


def load_stage_pair_frames(
    *,
    stage_dir: str | None,
    raw_dir: str | None,
    target_stage: str,
    condition_stage: str,
    feature_columns: list[str] | None = None,
):
    if raw_dir:
        raw_df, inferred_features, sampling_frequency = load_raw_emg_directory(raw_dir)
        stage_frames = build_emg_stage_frames(raw_df, inferred_features, sampling_frequency)
        target_df = resolve_stage_frame(target_stage, stage_frames, inferred_features, sampling_frequency)
        cond_df = resolve_stage_frame(condition_stage, stage_frames, inferred_features, sampling_frequency)
        return align_frames(target_df, cond_df, feature_columns=feature_columns or inferred_features)

    if stage_dir is None:
        raise ValueError("stage_dir must be provided when raw_dir is not used.")

    target_df = load_stage_frame(stage_dir, target_stage)
    cond_df = load_stage_frame(stage_dir, condition_stage)
    return align_frames(target_df, cond_df, feature_columns=feature_columns)


def build_train_args(
    args: argparse.Namespace,
    *,
    target_stage: str,
    condition_stage: str,
    output_dir: Path,
    fold: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="train",
        target=None,
        condition=None,
        raw_dir=args.raw_dir,
        stage_dir=args.stage_dir,
        output_dir=str(output_dir),
        target_stage=target_stage,
        condition_stage=condition_stage,
        window_size=args.window_size,
        stride=args.stride,
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        recurrent_layers=args.recurrent_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        beta_start=args.beta_start,
        beta_warmup_epochs=args.beta_warmup_epochs,
        l1_weight=args.l1_weight,
        corr_weight=args.corr_weight,
        patience=args.patience,
        seed=args.seed,
        split_mode="subject_cv",
        cv_fold=fold,
        device=args.device,
    )


def evaluate_fold(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    target_stage: str,
    condition_stage: str,
    fold: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = ReconstructionModelConfig(**checkpoint["model_config"])
    model_type = normalize_reconstruction_model_type(model_config.model_type)
    feature_columns = list(checkpoint["metadata"]["feature_columns"])
    window_size = int(checkpoint["metadata"]["window_size"])
    eval_stride = args.eval_stride or int(checkpoint["metadata"].get("stride", window_size))

    target_df, cond_df, _ = load_stage_pair_frames(
        stage_dir=args.stage_dir,
        raw_dir=args.raw_dir,
        target_stage=target_stage,
        condition_stage=condition_stage,
        feature_columns=feature_columns,
    )

    invalid_columns = [
        col
        for col in feature_columns
        if not np.isfinite(target_df[col].to_numpy(dtype=float)).all()
        or not np.isfinite(cond_df[col].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise ValueError("Non-finite values were found in feature columns: " + ", ".join(invalid_columns))

    _, _, test_groups, train_subjects, val_subject, test_subject = split_subject_cv(
        target_df["GROUP_ID"],
        fold_index=fold,
    )

    model = build_reconstruction_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    cond_scaled = checkpoint["condition_scaler"].transform(cond_df)
    tester = ReconstructionTester(
        model=model,
        target_scaler=checkpoint["target_scaler"],
        feature_columns=feature_columns,
        device=args.device,
    )
    heldout_metrics = tester.evaluate_groups(
        cond_df_scaled=cond_scaled,
        target_df=target_df,
        group_ids=test_groups,
        window_size=window_size,
        feature_columns=feature_columns,
        stride=eval_stride,
        include_window_summaries=args.save_window_summaries,
    )
    heldout_metrics = filter_metric_payload(heldout_metrics, RECOVERABILITY_METRICS)
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path.resolve()),
        "model_type": model_type,
        "target_stage": target_stage,
        "condition_stage": condition_stage,
        "train_subjects": train_subjects,
        "validation_subject": val_subject,
        "test_subject": test_subject,
        "window_size": window_size,
        "eval_stride": eval_stride,
        "heldout_metrics": heldout_metrics,
    }


def run_recoverability_suite(args: argparse.Namespace) -> dict[str, str]:
    validate_mode(args)
    args.model_type = normalize_reconstruction_model_type(args.model_type)

    stage_pairs = [parse_stage_pair(raw_pair) for raw_pair in args.stage_pairs]
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    suite_manifest = {
        "mode": "recoverability",
        "stage_dir": str(Path(args.stage_dir).resolve()) if args.stage_dir else None,
        "raw_dir": str(Path(args.raw_dir).resolve()) if args.raw_dir else None,
        "folds": list(args.folds),
        "stage_pairs": [f"{condition}->{target}" for condition, target in stage_pairs],
        "training_config": {
            "model_type": args.model_type,
            "window_size": args.window_size,
            "stride": args.stride,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "recurrent_layers": args.recurrent_layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "beta_start": args.beta_start,
            "beta_warmup_epochs": args.beta_warmup_epochs,
            "l1_weight": args.l1_weight,
            "corr_weight": args.corr_weight,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
        },
    }
    save_json(base_output_dir / "suite_manifest.json", suite_manifest)
    summary_paths: dict[str, str] = {}

    for condition_stage, target_stage in stage_pairs:
        stage_pair_dir = base_output_dir / pair_label(condition_stage, target_stage)
        stage_pair_dir.mkdir(parents=True, exist_ok=True)
        fold_results: list[dict] = []

        for fold in args.folds:
            fold_dir = stage_pair_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = fold_dir / "cvae_checkpoint.pt"

            if not (args.reuse_existing and checkpoint_path.exists()):
                train_args = build_train_args(
                    args,
                    target_stage=target_stage,
                    condition_stage=condition_stage,
                    output_dir=fold_dir,
                    fold=fold,
                )
                print(
                    f"\n===== Training {args.model_type} recoverability pair "
                    f"{condition_stage}->{target_stage} | fold {fold} ====="
                )
                train_command(train_args)
            else:
                print(
                    f"\n===== Reusing checkpoint for {args.model_type} "
                    f"{condition_stage}->{target_stage} | fold {fold} ====="
                )

            fold_payload = evaluate_fold(
                args,
                checkpoint_path=checkpoint_path,
                target_stage=target_stage,
                condition_stage=condition_stage,
                fold=fold,
            )
            save_json(fold_dir / "heldout_subject_metrics.json", fold_payload)
            fold_results.append(fold_payload)

        fold_metric_payloads = [
            {
                "summary": result["heldout_metrics"]["summary_mean"],
                "per_channel": result["heldout_metrics"]["per_channel_mean"],
            }
            for result in fold_results
        ]
        fold_aggregate = aggregate_metric_payloads(fold_metric_payloads)
        stage_pair_summary = {
            "model_type": args.model_type,
            "condition_stage": condition_stage,
            "target_stage": target_stage,
            "num_folds": len(fold_results),
            "fold_summary_mean": fold_aggregate["summary_mean"],
            "fold_summary_std": fold_aggregate["summary_std"],
            "per_channel_mean": fold_aggregate["per_channel_mean"],
            "per_channel_std": fold_aggregate["per_channel_std"],
            "folds": [
                {
                    "fold": result["fold"],
                    "test_subject": result["test_subject"],
                    "summary_mean": result["heldout_metrics"]["summary_mean"],
                    "summary_std": result["heldout_metrics"]["summary_std"],
                    "num_groups": result["heldout_metrics"]["num_groups"],
                }
                for result in fold_results
            ],
        }
        stage_pair_summary = filter_metric_payload(stage_pair_summary, RECOVERABILITY_METRICS)
        summary_path = stage_pair_dir / "suite_summary.json"
        save_json(summary_path, stage_pair_summary)
        summary_paths[f"{condition_stage}->{target_stage}"] = str(summary_path.resolve())
        print(f"Saved recoverability summary to: {summary_path}")

    return summary_paths


def main() -> None:
    args = parse_args()
    run_recoverability_suite(args)


if __name__ == "__main__":
    main()
