#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
STAGE_DIR="${STAGE_DIR:-data/interim}"

usage() {
  cat <<'EOF'
Usage:
  ./run_experiment.sh recoverability [extra args...]
  ./run_experiment.sh predictability [extra args...]
  ./run_experiment.sh predictability-compare [extra args...]
  ./run_experiment.sh predictability-larocs [extra args...]
  ./run_experiment.sh predictability-horizon [extra args...]
  ./run_experiment.sh predictability-horizon-plots [extra args...]
  ./run_experiment.sh reconstruction-cv [extra args...]
  ./run_experiment.sh prediction-cv [extra args...]

Environment variables:
  GPU_ID      CUDA device index to expose. Default: 0
  DEVICE      torch device passed to scripts. Default: cuda
  STAGE_DIR   Input stage directory. Default: data/interim
  PYTHON_BIN  Python executable to use. Default: .venv/bin/python if present

Examples:
  ./run_experiment.sh predictability
  GPU_ID=0 ./run_experiment.sh reconstruction-cv --epochs 100
  ./run_experiment.sh prediction-cv --input-stage raw --target-stage lp_10hz
EOF
}

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi
shift

case "$MODE" in
  recoverability)
    OUTPUT_DIR="${OUTPUT_DIR:-data/recoverability_suite}"
    CMD=(
      "$PYTHON_BIN" src/run_recoverability_suite.py
      --stage-dir "$STAGE_DIR"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  predictability)
    OUTPUT_DIR="${OUTPUT_DIR:-data/predictability_suite}"
    CMD=(
      "$PYTHON_BIN" src/run_predictability_suite.py
      --stage-dir "$STAGE_DIR"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  predictability-compare)
    OUTPUT_DIR="${OUTPUT_DIR:-data/predictability_model_compare}"
    CMD=(
      "$PYTHON_BIN" src/run_predictability_model_compare.py
      --stage-dir "$STAGE_DIR"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  predictability-larocs)
    OUTPUT_DIR="${OUTPUT_DIR:-data/predictability_model_compare_larocs}"
    CMD=(
      "$PYTHON_BIN" src/run_larocs_predictability_compare.py
      --stage-dir "$STAGE_DIR"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  predictability-horizon)
    OUTPUT_DIR="${OUTPUT_DIR:-data/predictability_horizon_suite}"
    CMD=(
      "$PYTHON_BIN" src/run_predictability_horizon_suite.py
      --stage-dir "$STAGE_DIR"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  predictability-horizon-plots)
    OUTPUT_DIR="${OUTPUT_DIR:-data/predictability_horizon_suite}"
    CMD=(
      "$PYTHON_BIN" src/build_predictability_horizon_plots.py
      --comparison-json "$OUTPUT_DIR/comparison_summary.json"
      --output-dir "$OUTPUT_DIR/plots"
    )
    ;;
  reconstruction-cv)
    OUTPUT_DIR="${OUTPUT_DIR:-data/cv_outputs}"
    TARGET_STAGE="${TARGET_STAGE:-rectified}"
    CONDITION_STAGE="${CONDITION_STAGE:-lp_10hz}"
    CMD=(
      "$PYTHON_BIN" src/run_subject_cv.py
      --stage-dir "$STAGE_DIR"
      --target-stage "$TARGET_STAGE"
      --condition-stage "$CONDITION_STAGE"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  prediction-cv)
    OUTPUT_DIR="${OUTPUT_DIR:-data/next_window_cv}"
    INPUT_STAGE="${INPUT_STAGE:-rectified}"
    TARGET_STAGE="${TARGET_STAGE:-lp_10hz}"
    CMD=(
      "$PYTHON_BIN" src/run_next_window_cv.py
      --stage-dir "$STAGE_DIR"
      --input-stage "$INPUT_STAGE"
      --target-stage "$TARGET_STAGE"
      --output-dir "$OUTPUT_DIR"
      --device "$DEVICE"
    )
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running: ${CMD[*]} $*"
"${CMD[@]}" "$@"
