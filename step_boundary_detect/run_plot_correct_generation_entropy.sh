#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# CORRECT-GENERATION ENTROPY PLOT
# =============================================================================

CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/step_boundary_detect/output_plot"
PROMPT_KEY="prompt"
RESPONSE_KEY=""          # Leave empty if unused.

# If you already know a good example, set EXAMPLE_ID directly.
# Example from the previous entropy-boundary run: EXAMPLE_ID=278
EXAMPLE_ID=""
START_INDEX=270
MAX_SCAN_EXAMPLES=128

MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"
DEVICE=""

ACTOR_SAMPLING_MODE="greedy"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

MOVING_AVERAGE_WINDOW=32
PLOT_DPI=180
SEED=42
SKIP_MERGE=1
DISABLE_ACTOR_CACHE=0
TRUST_REMOTE_CODE=0

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python step_boundary_detect/plot_correct_generation_entropy.py
  --checkpoint_dir       "${CHECKPOINT_DIR}"
  --dataset_path         "${DATASET_PATH}"
  --output_dir           "${OUTPUT_DIR}"
  --prompt_key           "${PROMPT_KEY}"
  --start_index          "${START_INDEX}"
  --max_scan_examples    "${MAX_SCAN_EXAMPLES}"
  --max_prompt_length    "${MAX_PROMPT_LENGTH}"
  --max_new_tokens       "${MAX_NEW_TOKENS}"
  --dtype                "${DTYPE}"
  --actor_sampling_mode  "${ACTOR_SAMPLING_MODE}"
  --actor_temperature    "${ACTOR_TEMPERATURE}"
  --actor_top_p          "${ACTOR_TOP_P}"
  --actor_top_k          "${ACTOR_TOP_K}"
  --moving_average_window "${MOVING_AVERAGE_WINDOW}"
  --plot_dpi             "${PLOT_DPI}"
  --seed                 "${SEED}"
)

[[ -n "${RESPONSE_KEY}" ]] && CMD+=(--response_key "${RESPONSE_KEY}")
[[ -n "${EXAMPLE_ID}" ]] && CMD+=(--example_id "${EXAMPLE_ID}")
[[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
[[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
[[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)

cd "${REPO_DIR}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
