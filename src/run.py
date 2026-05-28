from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import EMGWindowDataset
from model import (
    AVAILABLE_RECONSTRUCTION_MODELS,
    ReconstructionModelConfig,
    build_reconstruction_model,
    normalize_reconstruction_model_type,
)
from test import ReconstructionTester
from train import CVAETrainer, TrainerConfig
from utils import (
    AVAILABLE_STAGE_NAMES,
    DataFrameScaler,
    SplitConfig,
    align_frames,
    build_emg_stage_frames,
    checkpoint_metadata,
    filter_finite_feature_columns,
    load_raw_emg_directory,
    load_stage_frame,
    load_table,
    make_generator,
    resolve_stage_frame,
    save_json,
    seed_everything,
    split_subject_cv,
    split_group_ids,
    validate_stage_name,
)
from visualize import ReconstructionVisualizer


NONDETERMINISTIC_UPSAMPLE_WARNING = "upsample_linear1d_backward_out_cuda does not have a deterministic implementation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stroke EMG reconstruction runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a reconstruction model")
    train_parser.add_argument("--target", help="Path to target dataframe file")
    train_parser.add_argument("--condition", help="Path to condition dataframe file")
    train_parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    train_parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    train_parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints")
    train_parser.add_argument(
        "--target-stage",
        default="raw",
        help="Preprocessing stage to use as target. Use a built-in stage or lp_<cutoff>hz such as lp_20hz",
    )
    train_parser.add_argument(
        "--condition-stage",
        default="lp_10hz",
        help="Preprocessing stage to use as condition. Use a built-in stage or lp_<cutoff>hz such as lp_20hz",
    )
    train_parser.add_argument("--window-size", type=int, default=1024)
    train_parser.add_argument("--stride", type=int, default=512)
    train_parser.add_argument(
        "--model-type",
        default="conditional_unet",
        help=f"Reconstruction backbone. Available: {', '.join(AVAILABLE_RECONSTRUCTION_MODELS)}",
    )
    train_parser.add_argument("--latent-dim", type=int, default=64)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--recurrent-layers", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--beta", type=float, default=1.0)
    train_parser.add_argument("--beta-start", type=float, default=1.0)
    train_parser.add_argument("--beta-warmup-epochs", type=int, default=0)
    train_parser.add_argument("--l1-weight", type=float, default=0.0)
    train_parser.add_argument("--corr-weight", type=float, default=0.0)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--split-mode",
        choices=["group_random", "subject_cv"],
        default="group_random",
        help="group_random: GROUP_ID random split, subject_cv: 5 train / 1 val / 1 test by subject",
    )
    train_parser.add_argument(
        "--cv-fold",
        type=int,
        default=0,
        help="Fold index for subject_cv mode. 0-6 rotates test subject and uses the next subject as validation.",
    )
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    test_parser = subparsers.add_parser("test", help="Run deterministic reconstruction")
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--target", help="Path to target dataframe file")
    test_parser.add_argument("--condition", help="Path to condition dataframe file")
    test_parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    test_parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    test_parser.add_argument(
        "--target-stage",
        default="raw",
        help="Preprocessing stage to use as target. Use a built-in stage or lp_<cutoff>hz such as lp_20hz",
    )
    test_parser.add_argument(
        "--condition-stage",
        default="lp_10hz",
        help="Preprocessing stage to use as condition. Use a built-in stage or lp_<cutoff>hz such as lp_20hz",
    )
    test_parser.add_argument("--group-id", required=True)
    test_parser.add_argument("--iterations", type=int, default=100)
    test_parser.add_argument("--output-dir", required=True)
    test_parser.add_argument("--sampling-iterations", type=int, default=0)
    test_parser.add_argument("--channel-index", type=int, default=0)
    test_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train_command(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    validate_stage_name(args.target_stage)
    validate_stage_name(args.condition_stage)
    model_type = normalize_reconstruction_model_type(args.model_type)

    using_raw_dir = bool(args.raw_dir)
    using_stage_dir = bool(args.stage_dir)
    using_tables = bool(args.target) and bool(args.condition)
    mode_count = sum([using_raw_dir, using_stage_dir, using_tables])
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --raw-dir, --stage-dir, or both --target and --condition.")

    if using_raw_dir:
        raw_df, feature_columns, sampling_frequency = load_raw_emg_directory(args.raw_dir)
        stage_frames = build_emg_stage_frames(raw_df, feature_columns, sampling_frequency)
        target_df = resolve_stage_frame(args.target_stage, stage_frames, feature_columns, sampling_frequency)
        cond_df = resolve_stage_frame(args.condition_stage, stage_frames, feature_columns, sampling_frequency)
        target_df, cond_df, feature_columns = align_frames(target_df, cond_df, feature_columns=feature_columns)
    elif using_stage_dir:
        target_df = load_stage_frame(args.stage_dir, args.target_stage)
        cond_df = load_stage_frame(args.stage_dir, args.condition_stage)
        target_df, cond_df, feature_columns = align_frames(target_df, cond_df)
    else:
        target_df = load_table(args.target)
        cond_df = load_table(args.condition)
        target_df, cond_df, feature_columns = align_frames(target_df, cond_df)

    target_df, cond_df, feature_columns, dropped_columns = filter_finite_feature_columns(
        target_df,
        cond_df,
        feature_columns,
    )
    if dropped_columns:
        print("Dropped non-finite feature columns:", ", ".join(dropped_columns))

    if args.split_mode == "subject_cv":
        train_groups, val_groups, test_groups, train_subjects, val_subject, test_subject = split_subject_cv(
            target_df["GROUP_ID"],
            fold_index=args.cv_fold,
        )
        print(f"Subject CV fold {args.cv_fold % 7}")
        print(f"Train subjects: {', '.join(train_subjects)}")
        print(f"Validation subject: {val_subject}")
        print(f"Test subject: {test_subject}")
    else:
        train_groups, val_groups, test_groups = split_group_ids(
            target_df["GROUP_ID"],
            SplitConfig(seed=args.seed),
        )

    target_scaler = DataFrameScaler(feature_columns).fit(target_df[target_df["GROUP_ID"].isin(train_groups)])
    cond_scaler = DataFrameScaler(feature_columns).fit(cond_df[cond_df["GROUP_ID"].isin(train_groups)])

    target_scaled = target_scaler.transform(target_df)
    cond_scaled = cond_scaler.transform(cond_df)

    train_ds = EMGWindowDataset(
        target_df=target_scaled,
        cond_df=cond_scaled,
        feature_columns=feature_columns,
        window_size=args.window_size,
        stride=args.stride,
        allowed_groups=train_groups,
    )
    val_ds = EMGWindowDataset(
        target_df=target_scaled,
        cond_df=cond_scaled,
        feature_columns=feature_columns,
        window_size=args.window_size,
        stride=args.stride,
        allowed_groups=val_groups,
    )
    test_ds = EMGWindowDataset(
        target_df=target_scaled,
        cond_df=cond_scaled,
        feature_columns=feature_columns,
        window_size=args.window_size,
        stride=args.stride,
        allowed_groups=test_groups,
    )

    generator = make_generator(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_config = ReconstructionModelConfig(
        num_channels=len(feature_columns),
        model_type=model_type,
        latent_dim=args.latent_dim,
        window_size=args.window_size,
        base_channels=args.base_channels,
        recurrent_layers=args.recurrent_layers,
    )
    model = build_reconstruction_model(model_config)
    trainer = CVAETrainer(
        model=model,
        config=TrainerConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            beta=args.beta,
            beta_start=args.beta_start,
            beta_warmup_epochs=args.beta_warmup_epochs,
            l1_weight=args.l1_weight,
            corr_weight=args.corr_weight,
            patience=args.patience,
            device=args.device,
        ),
    )
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        best_model, history = trainer.fit(train_loader, val_loader)

    nondeterministic_upsample_warning_seen = False
    for warning in captured_warnings:
        if NONDETERMINISTIC_UPSAMPLE_WARNING in str(warning.message):
            nondeterministic_upsample_warning_seen = True
            continue
        warnings.showwarning(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
            warning.file,
            warning.line,
        )

    test_loss = trainer.evaluate(test_loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "cvae_checkpoint.pt"

    trainer.save_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": best_model.state_dict(),
            "model_config": model_config.__dict__,
            "metadata": checkpoint_metadata(
                feature_columns=feature_columns,
                window_size=args.window_size,
                stride=args.stride,
                latent_dim=args.latent_dim,
                split_config=SplitConfig(seed=args.seed),
            )
            | (
                {
                    "raw_dir": str(Path(args.raw_dir).resolve()),
                    "target_stage": args.target_stage,
                    "condition_stage": args.condition_stage,
                }
                if using_raw_dir
                else {
                    "stage_dir": str(Path(args.stage_dir).resolve()),
                    "target_stage": args.target_stage,
                    "condition_stage": args.condition_stage,
                }
                if using_stage_dir
                else {}
            )
            | {
                "model_type": model_type,
                "split_mode": args.split_mode,
                "cv_fold": args.cv_fold,
                "base_channels": args.base_channels,
                "recurrent_layers": args.recurrent_layers,
                "beta": args.beta,
                "beta_start": args.beta_start,
                "beta_warmup_epochs": args.beta_warmup_epochs,
                "l1_weight": args.l1_weight,
                "corr_weight": args.corr_weight,
            },
            "history": history,
            "test_loss": test_loss,
            "target_scaler": target_scaler,
            "condition_scaler": cond_scaler,
        },
    )
    save_json(
        output_dir / "train_summary.json",
        {
            "history": history,
            "test_loss": test_loss,
            "training_config": {
                "model_type": model_type,
                "base_channels": args.base_channels,
                "latent_dim": args.latent_dim,
                "recurrent_layers": args.recurrent_layers,
                "beta": args.beta,
                "beta_start": args.beta_start,
                "beta_warmup_epochs": args.beta_warmup_epochs,
                "l1_weight": args.l1_weight,
                "corr_weight": args.corr_weight,
            },
        },
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Test loss: {test_loss:.6f}")
    if nondeterministic_upsample_warning_seen:
        print(
            "Warning: CUDA linear upsampling backward was non-deterministic during this training run; "
            "results may vary slightly across runs even with the same seed."
        )


def test_command(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_config = ReconstructionModelConfig(**checkpoint["model_config"])
    feature_columns = checkpoint["metadata"]["feature_columns"]
    window_size = checkpoint["metadata"]["window_size"]
    validate_stage_name(args.target_stage)
    validate_stage_name(args.condition_stage)

    model = build_reconstruction_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    using_raw_dir = bool(args.raw_dir)
    using_stage_dir = bool(args.stage_dir)
    using_tables = bool(args.target) and bool(args.condition)
    mode_count = sum([using_raw_dir, using_stage_dir, using_tables])
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --raw-dir, --stage-dir, or both --target and --condition.")

    if using_raw_dir:
        raw_df, _, sampling_frequency = load_raw_emg_directory(args.raw_dir)
        stage_frames = build_emg_stage_frames(raw_df, feature_columns, sampling_frequency)
        target_df = resolve_stage_frame(args.target_stage, stage_frames, feature_columns, sampling_frequency)
        cond_df = resolve_stage_frame(args.condition_stage, stage_frames, feature_columns, sampling_frequency)
        target_df, cond_df, _ = align_frames(target_df, cond_df, feature_columns=feature_columns)
    elif using_stage_dir:
        target_df = load_stage_frame(args.stage_dir, args.target_stage)
        cond_df = load_stage_frame(args.stage_dir, args.condition_stage)
        target_df, cond_df, _ = align_frames(target_df, cond_df, feature_columns=feature_columns)
    else:
        target_df = load_table(args.target)
        cond_df = load_table(args.condition)
        target_df, cond_df, _ = align_frames(target_df, cond_df, feature_columns=feature_columns)

    invalid_columns = [
        col
        for col in feature_columns
        if not np.isfinite(target_df[col].to_numpy(dtype=float)).all()
        or not np.isfinite(cond_df[col].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise ValueError(
            "Non-finite values were found in checkpoint feature columns: " + ", ".join(invalid_columns)
        )

    cond_scaled = checkpoint["condition_scaler"].transform(cond_df)

    tester = ReconstructionTester(
        model=model,
        target_scaler=checkpoint["target_scaler"],
        feature_columns=feature_columns,
        device=args.device,
    )
    result = tester.converged_reconstruction(
        cond_df_scaled=cond_scaled,
        target_df=target_df,
        group_id=args.group_id,
        window_size=window_size,
        feature_columns=feature_columns,
        iterations=args.iterations,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = ReconstructionVisualizer(feature_columns)
    visualizer.plot_window(
        result.target_window,
        result.recon_window,
        channel_index=args.channel_index,
        title=f"Reconstruction for {args.group_id}",
        save_path=output_dir / f"{args.group_id}_reconstruction.png",
    )
    if args.sampling_iterations > 0:
        sampled_result = tester.sampled_reconstruction(
            cond_df_scaled=cond_scaled,
            target_df=target_df,
            group_id=args.group_id,
            window_size=window_size,
            feature_columns=feature_columns,
            iterations=args.sampling_iterations,
        )
        visualizer.plot_sampling_window(
            sampled_result.target_window,
            sampled_result.sampled_reconstructions,
            channel_index=args.channel_index,
            title=f"Sampling reconstruction for {args.group_id} ({args.sampling_iterations} draws)",
            save_path=output_dir / f"{args.group_id}_sampling_{args.sampling_iterations}.png",
        )
        np.savez_compressed(
            output_dir / f"{args.group_id}_sampling_{args.sampling_iterations}.npz",
            target_window=sampled_result.target_window,
            recon_window_mean=sampled_result.recon_window,
            sampled_reconstructions=sampled_result.sampled_reconstructions,
            feature_columns=np.array(feature_columns, dtype=object),
        )
        save_json(
            output_dir / f"{args.group_id}_sampling_{args.sampling_iterations}_metrics.json",
            sampled_result.metrics,
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
