"""
Negative control experiment for recoverability.

Uses the SAME trained models from the recoverability suite but evaluates
with SHUFFLED condition-target pairs: condition signal comes from a
DIFFERENT trial than the target (derangement within the test subject).

If the model truly relies on the temporally aligned structure between
paired preprocessing stages, performance should drop significantly under
this control.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import ReconstructionModelConfig, build_reconstruction_model
from run_recoverability_suite import (
    RECOVERABILITY_METRICS,
    load_stage_pair_frames,
    pair_label,
    parse_stage_pair,
)
from signal_metrics import (
    aggregate_metric_payloads,
    compute_signal_metrics,
    filter_metric_payload,
)
from utils import (
    estimate_sampling_frequency,
    save_json,
    split_subject_cv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Negative control: evaluate recoverability models with shuffled condition-target pairs."
    )
    parser.add_argument("--suite-dir", required=True, help="Base directory of an existing recoverability suite")
    parser.add_argument("--stage-dir", required=True, help="Directory containing <stage>_df.parquet files")
    parser.add_argument(
        "--stage-pairs",
        nargs="*",
        default=["notch->bandpass", "rectified->notch", "lp_10hz->rectified"],
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save negative control results")
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(7)))
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_derangement(group_ids: list[str], seed: int = 42) -> dict[str, str]:
    """
    Build a derangement mapping: {target_gid -> cond_gid_to_use},
    where every group maps to a DIFFERENT group.
    """
    rng = np.random.default_rng(seed)
    ids = list(group_ids)
    shuffled = ids.copy()
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(ids, shuffled)):
            break
    return {ids[i]: shuffled[i] for i in range(len(ids))}


def evaluate_group_windows_shuffled(
    *,
    model: torch.nn.Module,
    target_scaler,
    cond_df_scaled,
    target_df,
    target_gid: str,
    cond_gid: str,
    window_size: int,
    feature_columns: list[str],
    stride: int,
    device: torch.device,
) -> dict:
    """
    Evaluate windows pairing target from target_gid with condition from
    cond_gid (a different group — the mismatched control).
    """
    import pandas as pd

    cond_group = cond_df_scaled[cond_df_scaled["GROUP_ID"] == cond_gid]
    target_group = target_df[target_df["GROUP_ID"] == target_gid]

    if len(cond_group) == 0:
        raise ValueError(f"Cond GROUP_ID={cond_gid} not found.")
    if len(target_group) == 0:
        raise ValueError(f"Target GROUP_ID={target_gid} not found.")

    fs = estimate_sampling_frequency(target_group)
    valid_len = min(len(cond_group), len(target_group))
    metrics_payloads: list[dict] = []

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, valid_len - window_size + 1, stride):
            end_idx = start_idx + window_size
            target_window = target_group.iloc[start_idx:end_idx][feature_columns].to_numpy(dtype=np.float32)
            cond_window = cond_group.iloc[start_idx:end_idx][feature_columns].to_numpy(dtype=np.float32)

            if target_window.shape[0] < window_size or cond_window.shape[0] < window_size:
                continue

            target_tensor = torch.from_numpy(target_window.T).unsqueeze(0).to(device)
            cond_tensor = torch.from_numpy(cond_window.T).unsqueeze(0).to(device)

            recon, _, _ = model(target_tensor, cond_tensor, deterministic=True)
            recon_np = recon.squeeze(0).cpu().numpy().T
            recon_final = target_scaler.inverse_transform_array(recon_np)

            payload = compute_signal_metrics(target_window, recon_final, fs=fs, channel_names=feature_columns)
            metrics_payloads.append(payload)

    if not metrics_payloads:
        raise ValueError(
            f"No valid windows for target_gid={target_gid}, cond_gid={cond_gid}, "
            f"valid_len={valid_len}, window_size={window_size}"
        )

    aggregate = aggregate_metric_payloads(metrics_payloads)
    return {
        "target_gid": target_gid,
        "cond_gid": cond_gid,
        "num_windows": len(metrics_payloads),
        "summary_mean": aggregate["summary_mean"],
        "summary_std": aggregate["summary_std"],
        "per_channel_mean": aggregate["per_channel_mean"],
        "per_channel_std": aggregate["per_channel_std"],
    }


def evaluate_fold_shuffled(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    target_stage: str,
    condition_stage: str,
    fold: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = ReconstructionModelConfig(**checkpoint["model_config"])
    feature_columns = list(checkpoint["metadata"]["feature_columns"])
    window_size = int(checkpoint["metadata"]["window_size"])
    eval_stride = args.eval_stride or int(checkpoint["metadata"].get("stride", window_size))

    target_df, cond_df, _ = load_stage_pair_frames(
        stage_dir=args.stage_dir,
        raw_dir=None,
        target_stage=target_stage,
        condition_stage=condition_stage,
        feature_columns=feature_columns,
    )

    _, _, test_groups, _, _, test_subject = split_subject_cv(
        target_df["GROUP_ID"], fold_index=fold
    )

    cond_scaled = checkpoint["condition_scaler"].transform(cond_df)

    shuffle_map = make_derangement(test_groups, seed=args.seed)

    model = build_reconstruction_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    dev = torch.device(args.device)
    target_scaler = checkpoint["target_scaler"]

    group_results: list[dict] = []
    for target_gid in test_groups:
        cond_gid = shuffle_map[target_gid]
        print(f"  Evaluating (shuffled) target={target_gid} <- cond={cond_gid}")
        result = evaluate_group_windows_shuffled(
            model=model,
            target_scaler=target_scaler,
            cond_df_scaled=cond_scaled,
            target_df=target_df,
            target_gid=target_gid,
            cond_gid=cond_gid,
            window_size=window_size,
            feature_columns=feature_columns,
            stride=eval_stride,
            device=dev,
        )
        group_results.append(result)

    group_payloads = [
        {"summary": r["summary_mean"], "per_channel": r["per_channel_mean"]}
        for r in group_results
    ]
    aggregate = aggregate_metric_payloads(group_payloads)
    heldout_metrics = {
        "num_groups": len(group_results),
        "group_ids": test_groups,
        "summary_mean": aggregate["summary_mean"],
        "summary_std": aggregate["summary_std"],
        "per_channel_mean": aggregate["per_channel_mean"],
        "per_channel_std": aggregate["per_channel_std"],
        "groups": group_results,
    }
    heldout_metrics = filter_metric_payload(heldout_metrics, RECOVERABILITY_METRICS)
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path.resolve()),
        "target_stage": target_stage,
        "condition_stage": condition_stage,
        "test_subject": test_subject,
        "control_type": "shuffled_condition",
        "shuffle_mapping": shuffle_map,
        "window_size": window_size,
        "eval_stride": eval_stride,
        "heldout_metrics": heldout_metrics,
    }


def run_negative_control(args: argparse.Namespace) -> None:
    stage_pairs = [parse_stage_pair(p) for p in args.stage_pairs]
    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pair_summaries: dict[str, dict] = {}

    for condition_stage, target_stage in stage_pairs:
        label = pair_label(condition_stage, target_stage)
        pair_output_dir = output_dir / label
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        fold_results: list[dict] = []

        for fold in args.folds:
            checkpoint_path = suite_dir / label / f"fold_{fold}" / "cvae_checkpoint.pt"
            if not checkpoint_path.exists():
                print(f"[SKIP] Checkpoint not found: {checkpoint_path}")
                continue

            print(
                f"\n===== Negative control: {condition_stage}->{target_stage} | fold {fold} ====="
            )
            fold_payload = evaluate_fold_shuffled(
                args,
                checkpoint_path=checkpoint_path,
                target_stage=target_stage,
                condition_stage=condition_stage,
                fold=fold,
            )
            fold_dir = pair_output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_json(fold_dir / "shuffled_heldout_metrics.json", fold_payload)
            fold_results.append(fold_payload)

        if not fold_results:
            print(f"[WARN] No fold results for {label}, skipping summary.")
            continue

        fold_metric_payloads = [
            {
                "summary": result["heldout_metrics"]["summary_mean"],
                "per_channel": result["heldout_metrics"]["per_channel_mean"],
            }
            for result in fold_results
        ]
        fold_aggregate = aggregate_metric_payloads(fold_metric_payloads)
        pair_summary = {
            "control_type": "shuffled_condition",
            "condition_stage": condition_stage,
            "target_stage": target_stage,
            "num_folds": len(fold_results),
            "fold_summary_mean": fold_aggregate["summary_mean"],
            "fold_summary_std": fold_aggregate["summary_std"],
            "per_channel_mean": fold_aggregate["per_channel_mean"],
            "per_channel_std": fold_aggregate["per_channel_std"],
            "folds": [
                {
                    "fold": r["fold"],
                    "test_subject": r["test_subject"],
                    "shuffle_mapping": r["shuffle_mapping"],
                    "summary_mean": r["heldout_metrics"]["summary_mean"],
                    "summary_std": r["heldout_metrics"]["summary_std"],
                    "num_groups": r["heldout_metrics"]["num_groups"],
                }
                for r in fold_results
            ],
        }
        pair_summary = filter_metric_payload(pair_summary, RECOVERABILITY_METRICS)
        summary_path = pair_output_dir / "control_summary.json"
        save_json(summary_path, pair_summary)
        all_pair_summaries[f"{condition_stage}->{target_stage}"] = pair_summary
        print(f"Saved negative control summary: {summary_path}")

    save_json(output_dir / "control_comparison_summary.json", all_pair_summaries)
    print(f"\nAll negative control summaries saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_negative_control(args)


if __name__ == "__main__":
    main()
