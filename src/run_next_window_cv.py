from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 7-fold subject CV for next-window EMG prediction.")
    parser.add_argument("--stage-dir", help="Directory containing <stage>_df.parquet files")
    parser.add_argument("--raw-dir", help="Directory containing raw csv_output folders")
    parser.add_argument("--input", help="Path to input dataframe file")
    parser.add_argument("--target", help="Path to target dataframe file")
    parser.add_argument("--input-stage", default="raw")
    parser.add_argument("--target-stage", default="lp_10hz")
    parser.add_argument("--output-dir", required=True, help="Base directory for CV outputs")
    parser.add_argument("--window-ms", type=float, default=250.0)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--model-type", default="lstm")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    mode_count = sum(
        [
            bool(args.stage_dir),
            bool(args.raw_dir),
            bool(args.input) and bool(args.target),
        ]
    )
    if mode_count != 1:
        raise ValueError("Provide exactly one of: --stage-dir, --raw-dir, or both --input and --target.")

    command = [sys.executable, str(Path(__file__).with_name("run_next_window.py")), "train"]
    if args.stage_dir:
        command += ["--stage-dir", args.stage_dir]
    elif args.raw_dir:
        command += ["--raw-dir", args.raw_dir]
    else:
        command += ["--input", args.input, "--target", args.target]

    command += ["--input-stage", args.input_stage]
    command += ["--target-stage", args.target_stage]
    command += ["--window-ms", str(args.window_ms)]
    if args.seq_len is not None:
        command += ["--seq-len", str(args.seq_len)]
    if args.stride is not None:
        command += ["--stride", str(args.stride)]
    command += ["--model-type", args.model_type]
    command += ["--hidden-dim", str(args.hidden_dim)]
    command += ["--num-layers", str(args.num_layers)]
    command += ["--dropout", str(args.dropout)]
    command += ["--batch-size", str(args.batch_size)]
    command += ["--epochs", str(args.epochs)]
    command += ["--learning-rate", str(args.learning_rate)]
    command += ["--patience", str(args.patience)]
    command += ["--seed", str(args.seed)]
    command += ["--split-mode", "subject_cv"]
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
