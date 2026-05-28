from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_next_window import (
    DEFAULT_NAIVE_BASELINES,
    LSTMNextWindowPredictor,
    NextWindowTester,
    evaluate_naive_baseline_groups,
    load_input_target_frames,
    load_predictor_config,
    normalize_predictor_model_type,
    train_command as train_prediction_command,
)
from signal_metrics import (
    aggregate_metric_payloads,
    compute_complexity_metrics,
    filter_metric_payload,
)
from utils import align_frames, normalize_stage_name, save_json, split_subject_cv, validate_stage_name


DEFAULT_INPUT_STAGES = (
    "raw",
    "notch",
    "bandpass",
    "rectified",
    "lp_10hz",
)
DEFAULT_HORIZONS_MS = (50, 100, 150)
PREDICTABILITY_METRICS = (
    "mse",
    "mae",
    "nrmse",
    "pearson_r",
    "ccc",
)
DEFAULT_RECOVERABILITY_JSON = "data/recoverability_comparison_final/recoverability_unified_gru_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run subject-held-out predictability experiments with a fixed 250 ms input, "
            "fixed 50 ms target, and direct multi-horizon forecasting."
        )
    )
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument(
        "--input-stages",
        nargs="*",
        default=list(DEFAULT_INPUT_STAGES),
        help="Input stages to compare against an lp_10hz target.",
    )
    parser.add_argument("--target-stage", default="lp_10hz")
    parser.add_argument("--output-dir", required=True, help="Base directory for predictability outputs")
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--input-window-ms", type=float, default=250.0)
    parser.add_argument("--pred-window-ms", type=float, default=50.0)
    parser.add_argument(
        "--horizons-ms",
        type=float,
        nargs="*",
        default=list(DEFAULT_HORIZONS_MS),
        help="Prediction horizons in milliseconds.",
    )
    parser.add_argument("--input-seq-len", type=int)
    parser.add_argument("--pred-seq-len", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument(
        "--model-type",
        default="larocs_rnn_lstm_envelope",
        help="Predictor backbone. Defaults to the best Larocs-style model observed so far.",
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
        "--recoverability-json",
        default=DEFAULT_RECOVERABILITY_JSON,
        help="Optional unified recoverability summary used for the rank-comparison plot.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation after the suite finishes.",
    )
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


def pair_label(input_stage: str, target_stage: str) -> str:
    return f"{stage_label(input_stage)}_to_{stage_label(target_stage)}"


def horizon_label(horizon_ms: float) -> str:
    return f"horizon_{int(round(horizon_ms)):03d}ms"


def build_train_args(
    args: argparse.Namespace,
    *,
    input_stage: str,
    output_dir: Path,
    fold: int,
    horizon_ms: float,
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
        window_ms=args.input_window_ms,
        input_window_ms=args.input_window_ms,
        pred_window_ms=args.pred_window_ms,
        horizon_ms=horizon_ms,
        seq_len=None,
        input_seq_len=args.input_seq_len,
        pred_seq_len=args.pred_seq_len,
        horizon_seq_len=None,
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


def _load_aligned_stage_frames(
    args: argparse.Namespace,
    *,
    input_stage: str,
    target_stage: str,
    feature_columns: list[str] | None = None,
) -> tuple:
    eval_args = argparse.Namespace(
        input=None,
        target=None,
        raw_dir=args.raw_dir,
        stage_dir=args.stage_dir,
        input_stage=input_stage,
        target_stage=target_stage,
    )
    input_df, target_df, _, _ = load_input_target_frames(eval_args, "input_stage", "target_stage")
    return align_frames(input_df, target_df, feature_columns=feature_columns)


def _complexity_payload_for_groups(
    input_df: pd.DataFrame,
    feature_columns: list[str],
    group_ids: list[str],
) -> dict:
    group_results: list[dict] = []
    for group_id in group_ids:
        group_df = input_df[input_df["GROUP_ID"] == group_id]
        payload = compute_complexity_metrics(
            group_df[feature_columns].to_numpy(dtype=np.float32),
            channel_names=feature_columns,
        )
        group_results.append(
            {
                "group_id": group_id,
                "num_samples": int(len(group_df)),
                "summary_mean": payload["summary"],
                "per_channel_mean": payload["per_channel"],
            }
        )

    aggregate = aggregate_metric_payloads(
        [
            {
                "summary": result["summary_mean"],
                "per_channel": result["per_channel_mean"],
            }
            for result in group_results
        ]
    )
    return {
        "num_groups": len(group_results),
        "group_ids": list(group_ids),
        "summary_mean": aggregate["summary_mean"],
        "summary_std": aggregate["summary_std"],
        "per_channel_mean": aggregate["per_channel_mean"],
        "per_channel_std": aggregate["per_channel_std"],
        "groups": group_results,
    }


def evaluate_fold(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    input_stage: str,
    fold: int,
    horizon_ms: float,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = load_predictor_config(checkpoint["model_config"], checkpoint.get("model_state_dict"))
    feature_columns = list(checkpoint["metadata"]["feature_columns"])
    input_seq_len = int(checkpoint["metadata"].get("input_window_size", checkpoint["metadata"]["window_size"]))
    pred_seq_len = int(checkpoint["metadata"].get("prediction_window_size", checkpoint["metadata"]["window_size"]))
    horizon_gap_seq_len = int(checkpoint["metadata"].get("horizon_gap_seq_len", 0))
    eval_stride = args.eval_stride or int(checkpoint["metadata"].get("stride", input_seq_len))

    input_df, target_df, _ = _load_aligned_stage_frames(
        args,
        input_stage=input_stage,
        target_stage=args.target_stage,
        feature_columns=feature_columns,
    )

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
    test_group_ids = list(test_groups)

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
        group_ids=test_group_ids,
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
            group_ids=test_group_ids,
            input_seq_len=input_seq_len,
            pred_seq_len=pred_seq_len,
            horizon_gap_seq_len=horizon_gap_seq_len,
            stride=eval_stride,
            baseline_names=DEFAULT_NAIVE_BASELINES,
            include_window_summaries=args.save_window_summaries,
        ).items()
    }
    complexity = _complexity_payload_for_groups(input_df, feature_columns, test_group_ids)
    return {
        "fold": fold,
        "horizon_ms": horizon_ms,
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
        "complexity": complexity,
    }


def _aggregate_fold_summary_records(fold_results: list[dict]) -> tuple[dict, dict]:
    fold_metric_payloads = [
        {
            "summary": result["heldout_metrics"]["summary_mean"],
            "per_channel": result["heldout_metrics"]["per_channel_mean"],
        }
        for result in fold_results
    ]
    fold_complexity_payloads = [
        {
            "summary": result["complexity"]["summary_mean"],
            "per_channel": result["complexity"]["per_channel_mean"],
        }
        for result in fold_results
    ]
    return (
        aggregate_metric_payloads(fold_metric_payloads),
        aggregate_metric_payloads(fold_complexity_payloads),
    )


def _attach_relative_to_raw(stage_horizon_payloads: dict[float, dict[str, dict]], input_stages: list[str]) -> None:
    eps = 1e-8
    for horizon_ms, stage_payloads in stage_horizon_payloads.items():
        if "raw" not in stage_payloads:
            continue
        raw_payload = stage_payloads["raw"]
        raw_folds = {
            int(record["fold"]): record
            for record in raw_payload["folds"]
        }

        for stage_name in input_stages:
            payload = stage_payloads[stage_name]
            relative_fold_records: list[dict] = []
            for fold_record in payload["folds"]:
                fold = int(fold_record["fold"])
                raw_fold = raw_folds[fold]
                raw_mse = float(raw_fold["heldout_metrics"]["summary_mean"]["mse"])
                raw_pe = float(raw_fold["complexity"]["summary_mean"]["pe"])
                stage_mse = float(fold_record["heldout_metrics"]["summary_mean"]["mse"])
                stage_pe = float(fold_record["complexity"]["summary_mean"]["pe"])
                delta_mse_vs_raw = stage_mse - raw_mse
                delta_pe_vs_raw = stage_pe - raw_pe
                cnp = -delta_mse_vs_raw / (abs(delta_pe_vs_raw) + eps)
                relative_fold_records.append(
                    {
                        "fold": fold,
                        "raw_mse": raw_mse,
                        "stage_mse": stage_mse,
                        "delta_mse_vs_raw": delta_mse_vs_raw,
                        "raw_pe": raw_pe,
                        "stage_pe": stage_pe,
                        "delta_pe_vs_raw": delta_pe_vs_raw,
                        "cnp": cnp,
                    }
                )

            relative_aggregate = aggregate_metric_payloads(
                [
                    {
                        "summary": {key: value for key, value in record.items() if key != "fold"},
                        "per_channel": {},
                    }
                    for record in relative_fold_records
                ]
            )
            payload["relative_to_raw"] = {
                "fold_records": relative_fold_records,
                "summary_mean": relative_aggregate["summary_mean"],
                "summary_std": relative_aggregate["summary_std"],
            }


def _build_naive_baseline_summary(fold_results: list[dict]) -> dict[str, dict]:
    naive_baseline_summary: dict[str, dict] = {}
    for baseline_name in DEFAULT_NAIVE_BASELINES:
        baseline_metric_payloads = [
            {
                "summary": result["naive_baselines"][baseline_name]["summary_mean"],
                "per_channel": result["naive_baselines"][baseline_name]["per_channel_mean"],
            }
            for result in fold_results
        ]
        naive_baseline_summary[baseline_name] = aggregate_metric_payloads(baseline_metric_payloads)
    return naive_baseline_summary


def build_markdown_summary(comparison_payload: dict) -> str:
    lines = ["# Predictability Horizon Summary", ""]
    lines.append(f"Model: `{comparison_payload['model_type']}`")
    lines.append(f"Target stage: `{comparison_payload['target_stage']}`")
    lines.append("")

    for horizon_result in comparison_payload["horizon_rankings"]:
        lines.append(f"## Horizon {int(round(horizon_result['horizon_ms']))} ms")
        lines.append("")
        lines.append("| rank | input stage | cnp | mse | delta_mse_vs_raw | delta_pe_vs_raw | ccc | pearson_r | pe | sampen |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in horizon_result["rows"]:
            lines.append(
                f"| {row['rank_by_cnp']} | {row['input_stage']} | {row['cnp']:.4f} | "
                f"{row['mse']:.4f} | {row['delta_mse_vs_raw']:.4f} | {row['delta_pe_vs_raw']:.4f} | "
                f"{row['ccc']:.4f} | {row['pearson_r']:.4f} | {row['pe']:.4f} | {row['sampen']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_predictability_horizon_suite(args: argparse.Namespace) -> Path:
    validate_mode(args)

    args.model_type = normalize_predictor_model_type(args.model_type)
    args.target_stage = normalize_stage_name(args.target_stage)
    validate_stage_name(args.target_stage)
    input_stages = normalize_input_stages(list(args.input_stages))
    horizons_ms = [float(horizon_ms) for horizon_ms in args.horizons_ms]

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "mode": "predictability_horizon",
        "stage_dir": str(Path(args.stage_dir).resolve()) if args.stage_dir else None,
        "raw_dir": str(Path(args.raw_dir).resolve()) if args.raw_dir else None,
        "folds": list(args.folds),
        "input_stages": input_stages,
        "target_stage": args.target_stage,
        "horizons_ms": horizons_ms,
        "training_config": {
            "model_type": args.model_type,
            "input_window_ms": args.input_window_ms,
            "pred_window_ms": args.pred_window_ms,
            "input_seq_len": args.input_seq_len,
            "pred_seq_len": args.pred_seq_len,
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

    stage_horizon_payloads: dict[float, dict[str, dict]] = {horizon_ms: {} for horizon_ms in horizons_ms}

    for input_stage in input_stages:
        pair_dir = base_output_dir / pair_label(input_stage, args.target_stage)
        pair_dir.mkdir(parents=True, exist_ok=True)

        for horizon_ms in horizons_ms:
            horizon_dir = pair_dir / horizon_label(horizon_ms)
            horizon_dir.mkdir(parents=True, exist_ok=True)
            fold_results: list[dict] = []

            for fold in args.folds:
                fold_dir = horizon_dir / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = fold_dir / "next_window_checkpoint.pt"

                if not (args.reuse_existing and checkpoint_path.exists()):
                    train_args = build_train_args(
                        args,
                        input_stage=input_stage,
                        output_dir=fold_dir,
                        fold=fold,
                        horizon_ms=horizon_ms,
                    )
                    print(
                        f"\n===== Training {args.model_type} predictability "
                        f"{input_stage}->{args.target_stage} | horizon {int(round(horizon_ms))} ms | fold {fold} ====="
                    )
                    train_prediction_command(train_args)
                else:
                    print(
                        f"\n===== Reusing checkpoint for {args.model_type} "
                        f"{input_stage}->{args.target_stage} | horizon {int(round(horizon_ms))} ms | fold {fold} ====="
                    )

                fold_payload = evaluate_fold(
                    args,
                    checkpoint_path=checkpoint_path,
                    input_stage=input_stage,
                    fold=fold,
                    horizon_ms=horizon_ms,
                )
                save_json(fold_dir / "heldout_subject_metrics.json", fold_payload)
                fold_results.append(fold_payload)

            fold_aggregate, complexity_aggregate = _aggregate_fold_summary_records(fold_results)
            suite_summary = {
                "input_stage": input_stage,
                "target_stage": args.target_stage,
                "model_type": args.model_type,
                "horizon_ms": horizon_ms,
                "input_window_ms": args.input_window_ms,
                "pred_window_ms": args.pred_window_ms,
                "num_folds": len(fold_results),
                "fold_summary_mean": fold_aggregate["summary_mean"],
                "fold_summary_std": fold_aggregate["summary_std"],
                "per_channel_mean": fold_aggregate["per_channel_mean"],
                "per_channel_std": fold_aggregate["per_channel_std"],
                "complexity_summary_mean": complexity_aggregate["summary_mean"],
                "complexity_summary_std": complexity_aggregate["summary_std"],
                "complexity_per_channel_mean": complexity_aggregate["per_channel_mean"],
                "complexity_per_channel_std": complexity_aggregate["per_channel_std"],
                "naive_baselines": _build_naive_baseline_summary(fold_results),
                "folds": fold_results,
            }
            summary_path = horizon_dir / "suite_summary.json"
            save_json(summary_path, suite_summary)
            stage_horizon_payloads[horizon_ms][input_stage] = {
                "summary_path": str(summary_path.resolve()),
                **suite_summary,
            }

    _attach_relative_to_raw(stage_horizon_payloads, input_stages)

    for horizon_ms, payloads_by_stage in stage_horizon_payloads.items():
        for input_stage, payload in payloads_by_stage.items():
            summary_path = Path(payload["summary_path"])
            save_json(summary_path, payload)

    horizon_rankings: list[dict] = []
    for horizon_ms in horizons_ms:
        rows: list[dict] = []
        for input_stage in input_stages:
            payload = stage_horizon_payloads[horizon_ms][input_stage]
            summary = payload["fold_summary_mean"]
            complexity = payload["complexity_summary_mean"]
            relative = payload["relative_to_raw"]["summary_mean"]
            rows.append(
                {
                    "input_stage": input_stage,
                    "horizon_ms": horizon_ms,
                    "summary_path": payload["summary_path"],
                    "mse": float(summary["mse"]),
                    "mae": float(summary["mae"]),
                    "nrmse": float(summary["nrmse"]),
                    "pearson_r": float(summary["pearson_r"]),
                    "ccc": float(summary["ccc"]),
                    "pe": float(complexity["pe"]),
                    "sampen": float(complexity["sampen"]),
                    "raw_mse": float(relative["raw_mse"]),
                    "delta_mse_vs_raw": float(relative["delta_mse_vs_raw"]),
                    "delta_pe_vs_raw": float(relative["delta_pe_vs_raw"]),
                    "cnp": float(relative["cnp"]),
                }
            )

        rows.sort(key=lambda item: item["cnp"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank_by_cnp"] = rank
        horizon_rankings.append({"horizon_ms": horizon_ms, "ranking_metric": "cnp", "rows": rows})

    comparison_payload = {
        "mode": "predictability_horizon",
        "model_type": args.model_type,
        "input_stages": input_stages,
        "target_stage": args.target_stage,
        "horizons_ms": horizons_ms,
        "suite_manifest_path": str((base_output_dir / "suite_manifest.json").resolve()),
        "horizon_rankings": horizon_rankings,
    }
    comparison_json = base_output_dir / "comparison_summary.json"
    comparison_md = base_output_dir / "comparison_summary.md"
    with comparison_json.open("w", encoding="utf-8") as file:
        json.dump(comparison_payload, file, ensure_ascii=False, indent=2)
    comparison_md.write_text(build_markdown_summary(comparison_payload), encoding="utf-8")

    if not args.skip_plots:
        from build_predictability_horizon_plots import build_plots

        build_plots(
            comparison_json=comparison_json,
            output_dir=base_output_dir / "plots",
            recoverability_json=Path(args.recoverability_json),
        )

    print(f"\nSaved comparison JSON to: {comparison_json.resolve()}")
    print(f"Saved comparison markdown to: {comparison_md.resolve()}")
    return comparison_json


def main() -> None:
    run_predictability_horizon_suite(parse_args())


if __name__ == "__main__":
    main()
