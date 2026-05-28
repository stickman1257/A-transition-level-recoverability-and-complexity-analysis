from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from run_next_window import (
    DEFAULT_NAIVE_BASELINES,
    LSTMNextWindowPredictor,
    NextWindowTester,
    PREDICTOR_MODEL_TYPES,
    evaluate_naive_baseline_groups,
    load_input_target_frames,
    load_predictor_config,
    normalize_predictor_model_type,
    train_command as train_prediction_command,
)
from signal_metrics import aggregate_metric_payloads, filter_metric_payload
from utils import align_frames, normalize_stage_name, save_json, split_subject_cv, validate_stage_name


DEFAULT_INPUT_STAGES = (
    "raw",
    "bandpass+notch",
    "rectified",
    "lp_5hz",
    "lp_10hz",
)
PREDICTABILITY_METRICS = (
    "nrmse",
    "pearson_r",
    "ccc",
    "mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictability experiments for multiple input stages.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument(
        "--input-stages",
        nargs="*",
        default=list(DEFAULT_INPUT_STAGES),
        help="Input stages to compare. 'bandpass+notch' is treated as the 'notch' stage.",
    )
    parser.add_argument("--target-stage", default="lp_10hz")
    parser.add_argument("--output-dir", required=True, help="Base directory for predictability outputs")
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--window-ms", type=float, default=250.0)
    parser.add_argument("--input-window-ms", type=float)
    parser.add_argument("--pred-window-ms", type=float)
    parser.add_argument("--horizon-ms", type=float, default=0.0)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--input-seq-len", type=int)
    parser.add_argument("--pred-seq-len", type=int)
    parser.add_argument("--horizon-seq-len", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument(
        "--model-type",
        default="lstm",
        help=f"Predictor backbone. Available: {', '.join(PREDICTOR_MODEL_TYPES)}",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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


def normalize_input_stages(stage_names: list[str]) -> list[str]:
    normalized = [normalize_stage_name(stage_name) for stage_name in stage_names]
    for stage_name in normalized:
        validate_stage_name(stage_name)
    return normalized


def stage_label(stage_name: str) -> str:
    return stage_name.replace(".", "_")


def build_train_args(
    args: argparse.Namespace,
    *,
    input_stage: str,
    output_dir: Path,
    fold: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="train",
        input=None,
        target=None,
        raw_dir=args.raw_dir,
        stage_dir=args.stage_dir,
        output_dir=str(output_dir),
        input_stage=input_stage,
        target_stage=args.target_stage,
        window_ms=args.window_ms,
        input_window_ms=args.input_window_ms,
        pred_window_ms=args.pred_window_ms,
        horizon_ms=getattr(args, "horizon_ms", 0.0),
        seq_len=args.seq_len,
        input_seq_len=args.input_seq_len,
        pred_seq_len=args.pred_seq_len,
        horizon_seq_len=getattr(args, "horizon_seq_len", None),
        stride=args.stride,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
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
    input_stage: str,
    fold: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = load_predictor_config(checkpoint["model_config"], checkpoint.get("model_state_dict"))
    feature_columns = list(checkpoint["metadata"]["feature_columns"])
    input_seq_len = int(checkpoint["metadata"].get("input_window_size", checkpoint["metadata"]["window_size"]))
    pred_seq_len = int(checkpoint["metadata"].get("prediction_window_size", checkpoint["metadata"]["window_size"]))
    horizon_gap_seq_len = int(checkpoint["metadata"].get("horizon_gap_seq_len", 0))
    eval_stride = args.eval_stride or int(checkpoint["metadata"].get("stride", input_seq_len))

    eval_args = argparse.Namespace(
        input=None,
        target=None,
        raw_dir=args.raw_dir,
        stage_dir=args.stage_dir,
        input_stage=input_stage,
        target_stage=args.target_stage,
    )
    input_df, target_df, _, _ = load_input_target_frames(eval_args, "input_stage", "target_stage")
    input_df, target_df, _ = align_frames(input_df, target_df, feature_columns=feature_columns)

    invalid_columns = [
        col
        for col in feature_columns
        if not np.isfinite(input_df[col].to_numpy(dtype=float)).all()
        or not np.isfinite(target_df[col].to_numpy(dtype=float)).all()
    ]
    if invalid_columns:
        raise ValueError("Non-finite values were found in feature columns: " + ", ".join(invalid_columns))

    _, _, test_groups, train_subjects, val_subject, test_subject = split_subject_cv(
        input_df["GROUP_ID"],
        fold_index=fold,
    )

    model = LSTMNextWindowPredictor(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    input_scaled = checkpoint["input_scaler"].transform(input_df)
    tester = NextWindowTester(
        model=model,
        target_scaler=checkpoint["target_scaler"],
        feature_columns=feature_columns,
        device=args.device,
    )
    heldout_metrics = tester.evaluate_groups(
        input_df_scaled=input_scaled,
        target_df=target_df,
        group_ids=test_groups,
        input_seq_len=input_seq_len,
        pred_seq_len=pred_seq_len,
        horizon_gap_seq_len=horizon_gap_seq_len,
        stride=eval_stride,
        include_window_summaries=args.save_window_summaries,
    )
    heldout_metrics = filter_metric_payload(heldout_metrics, PREDICTABILITY_METRICS)
    naive_baselines = {
        baseline_name: filter_metric_payload(baseline_payload, PREDICTABILITY_METRICS)
        for baseline_name, baseline_payload in evaluate_naive_baseline_groups(
            input_df=input_df,
            target_df=target_df,
            feature_columns=feature_columns,
            group_ids=test_groups,
            input_seq_len=input_seq_len,
            pred_seq_len=pred_seq_len,
            horizon_gap_seq_len=horizon_gap_seq_len,
            stride=eval_stride,
            baseline_names=DEFAULT_NAIVE_BASELINES,
            include_window_summaries=args.save_window_summaries,
        ).items()
    }
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path.resolve()),
        "model_type": model_config.model_type,
        "input_stage": input_stage,
        "target_stage": args.target_stage,
        "train_subjects": train_subjects,
        "validation_subject": val_subject,
        "test_subject": test_subject,
        "input_seq_len": input_seq_len,
        "pred_seq_len": pred_seq_len,
        "horizon_gap_seq_len": horizon_gap_seq_len,
        "eval_stride": eval_stride,
        "heldout_metrics": heldout_metrics,
        "naive_baselines": naive_baselines,
    }


def run_predictability_suite(args: argparse.Namespace) -> dict[str, str]:
    validate_mode(args)

    args.model_type = normalize_predictor_model_type(args.model_type)
    args.target_stage = normalize_stage_name(args.target_stage)
    validate_stage_name(args.target_stage)
    input_stages = normalize_input_stages(list(args.input_stages))

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "mode": "predictability",
        "stage_dir": str(Path(args.stage_dir).resolve()) if args.stage_dir else None,
        "raw_dir": str(Path(args.raw_dir).resolve()) if args.raw_dir else None,
        "folds": list(args.folds),
        "input_stages": input_stages,
        "target_stage": args.target_stage,
        "training_config": {
            "model_type": args.model_type,
            "window_ms": args.window_ms,
            "input_window_ms": args.input_window_ms,
            "pred_window_ms": args.pred_window_ms,
            "horizon_ms": getattr(args, "horizon_ms", 0.0),
            "seq_len": args.seq_len,
            "input_seq_len": args.input_seq_len,
            "pred_seq_len": args.pred_seq_len,
            "horizon_seq_len": getattr(args, "horizon_seq_len", None),
            "stride": args.stride,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
        },
    }
    save_json(base_output_dir / "suite_manifest.json", suite_manifest)
    summary_paths: dict[str, str] = {}

    for input_stage in input_stages:
        input_stage_dir = base_output_dir / stage_label(input_stage)
        input_stage_dir.mkdir(parents=True, exist_ok=True)
        fold_results: list[dict] = []

        for fold in args.folds:
            fold_dir = input_stage_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = fold_dir / "next_window_checkpoint.pt"

            if not (args.reuse_existing and checkpoint_path.exists()):
                train_args = build_train_args(
                    args,
                    input_stage=input_stage,
                    output_dir=fold_dir,
                    fold=fold,
                )
                print(
                    f"\n===== Training {args.model_type} predictability input "
                    f"{input_stage} | fold {fold} ====="
                )
                train_prediction_command(train_args)
            else:
                print(
                    f"\n===== Reusing checkpoint for {args.model_type} "
                    f"input {input_stage} | fold {fold} ====="
                )

            fold_payload = evaluate_fold(
                args,
                checkpoint_path=checkpoint_path,
                input_stage=input_stage,
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
        naive_baseline_summary: dict[str, dict] = {}
        for baseline_name in DEFAULT_NAIVE_BASELINES:
            baseline_metric_payloads = [
                {
                    "summary": result["naive_baselines"][baseline_name]["summary_mean"],
                    "per_channel": result["naive_baselines"][baseline_name]["per_channel_mean"],
                }
                for result in fold_results
            ]
            baseline_aggregate = aggregate_metric_payloads(baseline_metric_payloads)
            naive_baseline_summary[baseline_name] = filter_metric_payload(
                {
                    "baseline_name": baseline_name,
                    "num_folds": len(fold_results),
                    "fold_summary_mean": baseline_aggregate["summary_mean"],
                    "fold_summary_std": baseline_aggregate["summary_std"],
                    "per_channel_mean": baseline_aggregate["per_channel_mean"],
                    "per_channel_std": baseline_aggregate["per_channel_std"],
                    "folds": [
                        {
                            "fold": result["fold"],
                            "test_subject": result["test_subject"],
                            "summary_mean": result["naive_baselines"][baseline_name]["summary_mean"],
                            "summary_std": result["naive_baselines"][baseline_name]["summary_std"],
                            "num_groups": result["naive_baselines"][baseline_name]["num_groups"],
                        }
                        for result in fold_results
                    ],
                },
                PREDICTABILITY_METRICS,
            )
        input_stage_summary = {
            "model_type": args.model_type,
            "input_stage": input_stage,
            "target_stage": args.target_stage,
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
                    "naive_baselines": {
                        baseline_name: result["naive_baselines"][baseline_name]["summary_mean"]
                        for baseline_name in DEFAULT_NAIVE_BASELINES
                    },
                }
                for result in fold_results
            ],
            "naive_baselines": naive_baseline_summary,
        }
        input_stage_summary = filter_metric_payload(input_stage_summary, PREDICTABILITY_METRICS)
        summary_path = input_stage_dir / "suite_summary.json"
        save_json(summary_path, input_stage_summary)
        summary_paths[input_stage] = str(summary_path.resolve())
        print(f"Saved predictability summary to: {summary_path}")

    return summary_paths


def main() -> None:
    args = parse_args()
    run_predictability_suite(args)


if __name__ == "__main__":
    main()
