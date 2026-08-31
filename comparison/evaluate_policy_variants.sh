#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${REPO_ROOT}/isaac-training/training/scripts/eval_models.py"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash comparison/evaluate_policy_variants.sh label=/path/to/checkpoint.pt [label=/path/to/checkpoint.pt ...] -- [eval args]

Labels:
  baseline_ppo, geom_encoder_ppo, risk_aux_ppo, ours_constrained

Example:
  bash comparison/evaluate_policy_variants.sh \
    baseline_ppo=/path/baseline/checkpoint_final.pt \
    ours_constrained=/path/ours/checkpoint_final.pt \
    -- --num-runs 20 --seed-start 100 --output comparison/eval_results.json
USAGE
  exit 2
fi

variant_overrides() {
  case "$1" in
    baseline_ppo)
      printf '%s' 'algo.method.step1_enabled=false||algo.method.step2_enabled=false||algo.method.step3_enabled=false'
      ;;
    geom_encoder_ppo)
      printf '%s' 'algo.method.step1_enabled=true||algo.method.step2_enabled=false||algo.method.step3_enabled=false'
      ;;
    risk_aux_ppo)
      printf '%s' 'algo.method.step1_enabled=true||algo.method.step2_enabled=true||algo.method.step3_enabled=false'
      ;;
    ours_constrained)
      printf '%s' 'algo.method.step1_enabled=true||algo.method.step2_enabled=true||algo.method.step3_enabled=true'
      ;;
    *)
      echo "Unknown label: $1" >&2
      echo "Expected one of: baseline_ppo, geom_encoder_ppo, risk_aux_ppo, ours_constrained" >&2
      exit 2
      ;;
  esac
}

MODEL_ARGS=()
EVAL_ARGS=("--output" "${SCRIPT_DIR}/eval_results.json")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      EVAL_ARGS+=("$@")
      break
      ;;
    *=*)
      label="${1%%=*}"
      checkpoint="${1#*=}"
      overrides="$(variant_overrides "${label}")"
      MODEL_ARGS+=("--model" "${label}::${REPO_ROOT}::${checkpoint}::${overrides}")
      shift
      ;;
    *)
      echo "Expected label=/path/to/checkpoint.pt or --, got: $1" >&2
      exit 2
      ;;
  esac
done

python "${EVAL_SCRIPT}" "${MODEL_ARGS[@]}" "${EVAL_ARGS[@]}"
