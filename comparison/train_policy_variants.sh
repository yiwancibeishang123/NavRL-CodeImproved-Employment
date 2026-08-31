#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/isaac-training/training/scripts/train.py"

variant="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

COMMON_OVERRIDES=(
  "headless=true"
  "wandb.mode=online"
)

run_variant() {
  local name="$1"
  shift
  echo "[Comparison] training ${name}"
  python "${TRAIN_SCRIPT}" \
    "wandb.name=comparison/${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

train_baseline_ppo() {
  run_variant "baseline_ppo" \
    "algo.method.step1_enabled=false" \
    "algo.method.step2_enabled=false" \
    "algo.method.step3_enabled=false" \
    "$@"
}

train_geom_encoder_ppo() {
  run_variant "geom_encoder_ppo" \
    "algo.method.step1_enabled=true" \
    "algo.method.step2_enabled=false" \
    "algo.method.step3_enabled=false" \
    "$@"
}

train_risk_aux_ppo() {
  run_variant "risk_aux_ppo" \
    "algo.method.step1_enabled=true" \
    "algo.method.step2_enabled=true" \
    "algo.method.step3_enabled=false" \
    "$@"
}

train_ours_constrained() {
  run_variant "ours_constrained" \
    "algo.method.step1_enabled=true" \
    "algo.method.step2_enabled=true" \
    "algo.method.step3_enabled=true" \
    "$@"
}

case "${variant}" in
  baseline_ppo)
    train_baseline_ppo "$@"
    ;;
  geom_encoder_ppo)
    train_geom_encoder_ppo "$@"
    ;;
  risk_aux_ppo)
    train_risk_aux_ppo "$@"
    ;;
  ours_constrained)
    train_ours_constrained "$@"
    ;;
  all)
    train_baseline_ppo "$@"
    train_geom_encoder_ppo "$@"
    train_risk_aux_ppo "$@"
    train_ours_constrained "$@"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    echo "Expected one of: all, baseline_ppo, geom_encoder_ppo, risk_aux_ppo, ours_constrained" >&2
    exit 2
    ;;
esac
