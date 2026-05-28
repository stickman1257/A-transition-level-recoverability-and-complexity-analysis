from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 7-fold subject CV for the reconstruction pipeline.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument("--target", help="Path to target dataframe file")
    parser.add_argument("--condition", help="Path to condition dataframe file")
    parser.add_argument("--target-stage", default="raw")
    parser.add_argument("--condition-stage", default="lp_10hz")
    parser.add_argument("--output-dir", required=True, help="Base directory for CV outputs")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-warmup-epochs", type=int, default=0)
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--corr-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    mode_count = sum(
        [
            bool(args.stage_dir),
            bool(args.raw_dir),
            bool(args.target) and bool(args.condition),
        ]
    )
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --stage-dir, --raw-dir, or both --target and --condition.")

    command = [sys.executable, str(Path(__file__).with_name("run.py")), "train"]
    if args.stage_dir:
        command += ["--stage-dir", args.stage_dir]
    elif args.raw_dir:
        command += ["--raw-dir", args.raw_dir]
    else:
        command += ["--target", args.target, "--condition", args.condition]

    command += [
        "--target-stage",
        args.target_stage,
        "--condition-stage",
        args.condition_stage,
        "--window-size",
        str(args.window_size),
        "--stride",
        str(args.stride),
        "--latent-dim",
        str(args.latent_dim),
        "--base-channels",
        str(args.base_channels),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--beta",
        str(args.beta),
        "--beta-start",
        str(args.beta_start),
        "--beta-warmup-epochs",
        str(args.beta_warmup_epochs),
        "--l1-weight",
        str(args.l1_weight),
        "--corr-weight",
        str(args.corr_weight),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--split-mode",
        "subject_cv",
    ]
    if args.device:
        command += ["--device", args.device]
    return command


def read_test_loss(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    test_loss = payload.get("test_loss")
    return None if test_loss is None else float(test_loss)


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    base_command = build_base_command(args)
    fold_results: list[dict[str, object]] = []

    for fold in range(7):
        fold_output_dir = base_output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        command = base_command + ["--cv-fold", str(fold), "--output-dir", str(fold_output_dir)]
        print(f"\n===== Running fold {fold} =====")
        print(" ".join(command))
        completed = subprocess.run(command, check=False)

        summary_path = fold_output_dir / "train_summary.json"
        fold_results.append(
            {
                "fold": fold,
                "returncode": int(completed.returncode),
                "output_dir": str(fold_output_dir.resolve()),
                "test_loss": read_test_loss(summary_path),
            }
        )

        if completed.returncode != 0:
            print(f"Fold {fold} failed with return code {completed.returncode}.")

    summary_path = base_output_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"folds": fold_results}, file, ensure_ascii=False, indent=2)

    print(f"\nSaved CV summary to: {summary_path.resolve()}")
    for result in fold_results:
        print(
            f"fold={result['fold']} returncode={result['returncode']} "
            f"test_loss={result['test_loss']} output_dir={result['output_dir']}"
        )


if __name__ == "__main__":
    main()
