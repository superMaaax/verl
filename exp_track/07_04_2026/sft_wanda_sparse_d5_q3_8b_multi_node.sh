#!/bin/bash
#SBATCH --job-name=sft_qwen3_8b_wanda
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --output=sft_qwen3_8b_wanda-%j_d5.out
#SBATCH --error=sft_qwen3_8b_wanda-%j_d5.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
source "${VENV}/bin/activate"

UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/.cache/uv}"
HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${SCRATCH}/data/embeddings}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH}/.cache/torch_extensions}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE" "$TORCH_EXTENSIONS_DIR"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export TORCH_EXTENSIONS_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Avoid inheriting incompatible local launch variables.
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID || true

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity and paths
# -----------------------------
export WANDB_PROJECT="prune_for_post_train"
WANDB_PROJECT="prune_for_post_train"
RUN_NAME="${RUN_NAME:-sft_qwen3_8b_wanda_sparse_kept_sparsity_d5}"
REAL_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_NAME}_${REAL_SLURM_JOB_ID}"

HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE:-}"
HF_MODULES_CACHE_ROOT="${HF_MODULES_CACHE:-}"
export HF_DATASETS_CACHE_ROOT
export HF_MODULES_CACHE_ROOT

WORK_DIR="${WORK_DIR:-/work2/09576/shuozhe/verl}"
MODEL_INIT_CKPT="${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-/work2/09576/shuozhe/gradient_prune/saved_calibration_dataset/qwen3-8b-instruct_math7500_correct_5_response/qwen3-8b-instruct_math7500_correct_5_response.parquet}"
VAL_FILE="${VAL_FILE:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"

export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/verl_runs}"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_LOG_DIR="${RUN_DIR}/train_log"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/work2/09576/shuozhe/verl/train_log_archive}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
TRAIN_STDOUT_LOG="${TRAIN_LOG_DIR}/job_${RUN_ID}.txt"

mkdir -p "$LOG_DIR" "$TRAIN_LOG_DIR" "$ARCHIVE_ROOT"

# -----------------------------
# SFT training defaults
# -----------------------------
# This is a small-data sparse fine-tune. Defaults are conservative for Qwen3-8B.
train_batch_size=${train_batch_size:-128}
micro_batch_size_per_gpu=${micro_batch_size_per_gpu:-2}
max_length=${max_length:-18432}
max_token_len_per_gpu=${max_token_len_per_gpu:-36864}
lr=${lr:-5e-6}
total_epochs=${total_epochs:-5}
save_freq=${save_freq:-50}

# -----------------------------
# WANDA sparse update config
# -----------------------------
# Sparse-update mask semantics in VERL: True means trainable/kept, False means frozen.
# WANDA score semantics: higher score means more important to keep.
# Therefore, sparsity=s trains exactly the top (1-s) WANDA-score entries per target linear weight.
SPARSE_UPDATE_MODE="wanda_top"
SPARSE_UPDATE_SPARSITY="${SPARSE_UPDATE_SPARSITY:-0.5}"
SPARSE_UPDATE_KEEP_FRACTION="${SPARSE_UPDATE_KEEP_FRACTION:-}"
WANDA_SCORE_DIR="${WANDA_SCORE_DIR:-/scratch/09576/shuozhe/gradient_prune/results/qwen3_8b_wanda_math7500/scores}"

if [[ -n "$SPARSE_UPDATE_KEEP_FRACTION" ]]; then
  SPARSE_UPDATE_MASK_TAG="keep_fraction_${SPARSE_UPDATE_KEEP_FRACTION}"
else
  SPARSE_UPDATE_MASK_TAG="sparsity_${SPARSE_UPDATE_SPARSITY}"
fi
SPARSE_UPDATE_MASK_PATH="${SPARSE_UPDATE_MASK_PATH:-${TRAIN_LOG_DIR}/wanda_top_${SPARSE_UPDATE_MASK_TAG}_mask.pt}"
SPARSE_UPDATE_VERIFY="${SPARSE_UPDATE_VERIFY:-true}"
SPARSE_UPDATE_VERIFY_INTERVAL="${SPARSE_UPDATE_VERIFY_INTERVAL:-200}"
# false: dense forward with frozen entries restored to pretrained values.
# true: actually pruned forward/training with frozen entries set to and kept at zero.
SPARSE_UPDATE_ZERO_FROZEN_PARAMS="${SPARSE_UPDATE_ZERO_FROZEN_PARAMS:-true}"
SPARSE_UPDATE_TARGET_MODULES="${SPARSE_UPDATE_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
SPARSE_UPDATE_EXCLUDE_KEYWORDS="${SPARSE_UPDATE_EXCLUDE_KEYWORDS:-embed,lm_head,norm,layernorm,rmsnorm}"

SPARSE_UPDATE_MASK_OVERRIDES=(
  sparse_update.enabled=true
  sparse_update.mode="${SPARSE_UPDATE_MODE}"
  sparse_update.mask_path="${SPARSE_UPDATE_MASK_PATH}"
  sparse_update.build_mask_on_init=false
  sparse_update.restore_frozen_after_step=true
  sparse_update.zero_frozen_params="${SPARSE_UPDATE_ZERO_FROZEN_PARAMS}"
  sparse_update.mask_optimizer_state=true
  sparse_update.verify_frozen_weights="${SPARSE_UPDATE_VERIFY}"
  sparse_update.verification_interval="${SPARSE_UPDATE_VERIFY_INTERVAL}"
  engine.use_orig_params=True
)

# -----------------------------
# Eval options
# -----------------------------
# eval_method: loss, generation_reward, or both.
eval_method=${eval_method:-generation_reward}
eval_before_train=${eval_before_train:-False}
eval_freq=${eval_freq:--1}
loss_eval_freq=${loss_eval_freq:-50}
generation_eval_freq=${generation_eval_freq:-50}

# loss_eval_files must be SFT messages format; generation_eval_files must be PPO prompt+reward_model format.
loss_eval_files=${loss_eval_files:-__TRAIN_FILE__}
generation_eval_files=${generation_eval_files:-${VAL_FILE}}

# Generation accuracy eval can be memory-heavy; start small unless you know memory is safe.
# Qwen3 thinking mode for generation eval. Set generation_eval_enable_thinking=False to disable.
generation_eval_enable_thinking=${generation_eval_enable_thinking:-True}
generation_eval_batch_size=${generation_eval_batch_size:-16}
generation_max_new_tokens=${generation_max_new_tokens:-18432}
generation_do_sample=${generation_do_sample:-False}
generation_temperature=${generation_temperature:-0.0}
generation_top_p=${generation_top_p:-1.0}
generation_top_k=${generation_top_k:-null}
generation_num_samples=${generation_num_samples:-1}
generation_dtype=${generation_dtype:-null}

generation_backend=${generation_backend:-vllm}
generation_vllm_gpu_memory_utilization=${generation_vllm_gpu_memory_utilization:-0.4}
generation_vllm_host_ip=${generation_vllm_host_ip:-127.0.0.1}
generation_vllm_enforce_eager=${generation_vllm_enforce_eager:-True}
generation_vllm_sync_weights=${generation_vllm_sync_weights:-True}
generation_vllm_enable_multiprocessing=${generation_vllm_enable_multiprocessing:-False}

# -----------------------------
# Multi-node torchrun config
# -----------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${SLURM_JOB_NUM_NODES}"
RDZV_PORT="${RDZV_PORT:-29500}"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

resolved_head_node_ip=""
for candidate_ip in $head_node_ip; do
  if [[ "$candidate_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    resolved_head_node_ip="$candidate_ip"
    break
  fi
done
if [[ -z "$resolved_head_node_ip" ]]; then
  for candidate_ip in $head_node_ip; do
    resolved_head_node_ip="$candidate_ip"
    break
  done
fi
if [[ -z "$resolved_head_node_ip" ]]; then
  echo "Failed to resolve a usable IP address for torchrun head node $head_node." >&2
  exit 1
fi
MASTER_ADDR="$resolved_head_node_ip"
MASTER_PORT="$RDZV_PORT"
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# -----------------------------
# Helpers
# -----------------------------
MODEL_PATH_RESOLVER="${WORK_DIR}/tools/resolve_model_init_path.py"

resolve_model_init_path() {
  local raw_path="$1"
  local role="$2"

  if [[ ! -f "$MODEL_PATH_RESOLVER" ]]; then
    echo "Missing model path resolver: $MODEL_PATH_RESOLVER" >&2
    return 1
  fi

  python3 "$MODEL_PATH_RESOLVER" \
    --path "$raw_path" \
    --role "$role" \
    --log-dir "$LOG_DIR"
}

describe_path() {
  local label="$1"
  local path="$2"

  echo "$label: $path"
  if [[ "$path" == *"://"* ]]; then
    echo "  non-local URI path"
  elif [[ -d "$path" || -f "$path" ]]; then
    ls -ld "$path"
  elif [[ "$path" = /* || "$path" == ./* || "$path" == ../* || "$path" == ~* ]]; then
    echo "  local path not found"
  else
    echo "  passthrough model identifier"
  fi
}

sync_to_work() {
  echo "Syncing lightweight logs/metadata back to WORK; checkpoints stay on SCRATCH."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a \
    --exclude='**/global_step_*' \
    --exclude='**/checkpoints/**' \
    --exclude='**/model_world_size_*_rank_*.pt' \
    --exclude='**/optim_world_size_*_rank_*.pt' \
    --exclude='**/extra_state_world_size_*_rank_*.pt' \
    --exclude='**/sparse_update_state_rank_*.pt' \
    --exclude='**/huggingface/**' \
    --exclude='**/*.safetensors' \
    --exclude='**/pytorch_model*.bin' \
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived lightweight run files to: $ARCHIVE_DIR"
  echo "Checkpoint/model artifacts remain on SCRATCH at: $TRAIN_LOG_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

MODEL_PATH="$(resolve_model_init_path "$MODEL_INIT_CKPT" actor)"

# -----------------------------
# Prepare SFT training data
# -----------------------------
RAW_TRAIN_FILE="$TRAIN_FILE"
SFT_PREPARED_DATA_DIR="${SFT_PREPARED_DATA_DIR:-${RUN_DIR}/prepared_data}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${SFT_PREPARED_DATA_DIR}/train_sft_messages.parquet}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:--1}"
SFT_DEDUP_BY_PROMPT="${SFT_DEDUP_BY_PROMPT:-false}"
SFT_RESPONSE_FILTER_CORRECT="${SFT_RESPONSE_FILTER_CORRECT:-true}"
SFT_ENABLE_THINKING_COLUMN="${SFT_ENABLE_THINKING_COLUMN:-}"
SFT_REUSE_PREPARED_DATA="${SFT_REUSE_PREPARED_DATA:-false}"

prepare_sft_data() {
  local input_path="$1"
  local output_path="$2"
  local converter="$WORK_DIR/tools/prepare_sft_messages_data.py"
  local converter_args=(
    --input "$input_path"
    --output "$output_path"
    --max-samples "$SFT_MAX_SAMPLES"
  )

  if [[ "$SFT_DEDUP_BY_PROMPT" == "true" ]]; then
    converter_args+=(--dedup-by-prompt)
  fi
  if [[ "$SFT_RESPONSE_FILTER_CORRECT" != "true" ]]; then
    converter_args+=(--no-filter-correct)
  fi
  if [[ -n "$SFT_ENABLE_THINKING_COLUMN" ]]; then
    converter_args+=(--enable-thinking "$SFT_ENABLE_THINKING_COLUMN")
  fi

  python3 "$converter" "${converter_args[@]}"
}

mkdir -p "$SFT_PREPARED_DATA_DIR"
case "${TRAIN_FILE##*.}" in
  jsonl|parquet|pq)
    if [[ "$SFT_REUSE_PREPARED_DATA" == "true" && -f "$SFT_TRAIN_FILE" ]]; then
      echo "Reusing prepared SFT messages dataset: $SFT_TRAIN_FILE" | tee "$LOG_DIR/prepare_sft_data.log"
    else
      echo "Preparing SFT messages dataset from: $TRAIN_FILE"
      prepare_sft_data "$TRAIN_FILE" "$SFT_TRAIN_FILE" | tee "$LOG_DIR/prepare_sft_data.log"
    fi
    TRAIN_FILE="$SFT_TRAIN_FILE"
    ;;
  *)
    echo "Unsupported TRAIN_FILE extension: $TRAIN_FILE" >&2
    exit 1
    ;;
esac
if [[ "$loss_eval_files" == "__TRAIN_FILE__" || "$loss_eval_files" == "$RAW_TRAIN_FILE" ]]; then
  loss_eval_files="$TRAIN_FILE"
fi

# -----------------------------
# Build WANDA kept-parameter mask
# -----------------------------
# Build the mask before torchrun so failures happen early and all ranks load one identical mask.
mkdir -p "$(dirname "$SPARSE_UPDATE_MASK_PATH")"
if [[ ! -f "$SPARSE_UPDATE_MASK_PATH" ]]; then
  echo "Building WANDA sparse-update mask at $SPARSE_UPDATE_MASK_PATH"
  mask_args=(
    --model_name_or_path "$MODEL_PATH"
    --output_path "$SPARSE_UPDATE_MASK_PATH"
    --wanda_score_dir "$WANDA_SCORE_DIR"
    --target_modules "$SPARSE_UPDATE_TARGET_MODULES"
    --exclude_keywords "$SPARSE_UPDATE_EXCLUDE_KEYWORDS"
  )
  if [[ -n "$SPARSE_UPDATE_KEEP_FRACTION" ]]; then
    mask_args+=(--keep_fraction "$SPARSE_UPDATE_KEEP_FRACTION")
  else
    mask_args+=(--sparsity "$SPARSE_UPDATE_SPARSITY")
  fi
  python3 "$WORK_DIR/tools/build_sparse_update_mask.py" "${mask_args[@]}" | tee "$LOG_DIR/build_wanda_sparse_update_mask.log"
else
  echo "Using existing WANDA sparse-update mask: $SPARSE_UPDATE_MASK_PATH"
fi

# -----------------------------
# Debug info and input checks
# -----------------------------
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run ID: ${RUN_ID}"
echo "SLURM nodes: ${SLURM_JOB_NODELIST}"
echo "Head node: ${head_node}"
echo "Rendezvous endpoint: ${RDZV_ENDPOINT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NNODES: ${NNODES}"
echo "SCRATCH: ${SCRATCH}"
echo "RUN_DIR: ${RUN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "TRAIN_LOG_DIR: ${TRAIN_LOG_DIR}"
describe_path "WORK_DIR" "$WORK_DIR"
describe_path "MODEL_INIT_CKPT" "$MODEL_INIT_CKPT"
describe_path "MODEL_PATH" "$MODEL_PATH"
describe_path "TRAIN_FILE" "$TRAIN_FILE"
describe_path "VAL_FILE" "$VAL_FILE"
describe_path "loss_eval_files" "$loss_eval_files"
describe_path "generation_eval_files" "$generation_eval_files"
echo "generation_eval_enable_thinking: $generation_eval_enable_thinking"
echo "SPARSE_UPDATE_MODE: $SPARSE_UPDATE_MODE"
echo "SPARSE_UPDATE_SPARSITY: $SPARSE_UPDATE_SPARSITY"
echo "SPARSE_UPDATE_KEEP_FRACTION: ${SPARSE_UPDATE_KEEP_FRACTION:-1 - sparsity}"
echo "SPARSE_UPDATE_ZERO_FROZEN_PARAMS: $SPARSE_UPDATE_ZERO_FROZEN_PARAMS"
echo "WANDA_SCORE_DIR: $WANDA_SCORE_DIR"
echo "SPARSE_UPDATE_MASK_PATH: $SPARSE_UPDATE_MASK_PATH"

ls -ld "$WORK_DIR"
ls -lh "$TRAIN_FILE"
ls -lh "$VAL_FILE"
ls -ld "$WANDA_SCORE_DIR"
if [[ ! -f "$WANDA_SCORE_DIR/metadata.json" ]]; then
  echo "Missing WANDA score metadata: $WANDA_SCORE_DIR/metadata.json" >&2
  exit 1
fi

cd "$WORK_DIR"

# -----------------------------
# Run SFT training
# -----------------------------
# One Slurm task launches one torchrun agent per node. torchrun then launches GPUS_PER_NODE
# local trainer processes on that node. SLURM_PROCID is the node_rank because ntasks-per-node=1.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    source "'"${VENV}"'/bin/activate"
    cd "'"${WORK_DIR}"'"
    export PYTHONPATH="'"${WORK_DIR}"'${PYTHONPATH:+:${PYTHONPATH}}"
    export UV_CACHE_DIR="'"${UV_CACHE_DIR}"'"
    export HF_HOME="'"${HF_HOME}"'"
    node_cache_root="${TMPDIR:-/tmp}/verl_'"${RUN_ID}"'_node_${SLURM_PROCID}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_ROOT:-${node_cache_root}/huggingface_datasets}"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE_ROOT:-${node_cache_root}/huggingface_modules}"
    export TIKTOKEN_ENCODINGS_BASE="'"${TIKTOKEN_ENCODINGS_BASE}"'"
    export TORCH_EXTENSIONS_DIR="'"${TORCH_EXTENSIONS_DIR}"'/node_${SLURM_PROCID}"
    mkdir -p "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TORCH_EXTENSIONS_DIR"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=true
    export HYDRA_FULL_ERROR="'"${HYDRA_FULL_ERROR}"'"
    export NCCL_DEBUG="'"${NCCL_DEBUG}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"

    torchrun \
      --nnodes="'"${NNODES}"'" \
      --nproc_per_node="'"${GPUS_PER_NODE}"'" \
      --node_rank="${SLURM_PROCID}" \
      --rdzv_id="'"${RUN_ID}"'" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="'"${RDZV_ENDPOINT}"'" \
      -m verl.trainer.sft_trainer \
      data.train_files="'"${TRAIN_FILE}"'" \
      data.val_files="'"${VAL_FILE}"'" \
      data.loss_val_files="'"${loss_eval_files}"'" \
      data.generation_eval_files="'"${generation_eval_files}"'" \
      data.generation_eval_batch_size="'"${generation_eval_batch_size}"'" \
      data.generation_eval_apply_chat_template_kwargs.enable_thinking="'"${generation_eval_enable_thinking}"'" \
      data.messages_key=messages \
      data.train_batch_size="'"${train_batch_size}"'" \
      data.micro_batch_size_per_gpu="'"${micro_batch_size_per_gpu}"'" \
      data.max_length="'"${max_length}"'" \
      data.max_token_len_per_gpu="'"${max_token_len_per_gpu}"'" \
      data.ignore_input_ids_mismatch=True \
      data.num_workers=8 \
      optim.lr="'"${lr}"'" \
      engine=fsdp \
      model.path="'"${MODEL_PATH}"'" \
      trainer.default_local_dir="'"${TRAIN_LOG_DIR}"'" \
      trainer.project_name="${WANDB_PROJECT}" \
      trainer.experiment_name="'"${RUN_ID}"'" \
      trainer.total_epochs="'"${total_epochs}"'" \
      trainer.save_freq="'"${save_freq}"'" \
      trainer.test_freq="'"${eval_freq}"'" \
      trainer.loss_test_freq="'"${loss_eval_freq}"'" \
      trainer.eval_method="'"${eval_method}"'" \
      trainer.val_before_train="'"${eval_before_train}"'" \
      trainer.generation_eval.test_freq="'"${generation_eval_freq}"'" \
      trainer.generation_eval.max_new_tokens="'"${generation_max_new_tokens}"'" \
      trainer.generation_eval.do_sample="'"${generation_do_sample}"'" \
      trainer.generation_eval.temperature="'"${generation_temperature}"'" \
      trainer.generation_eval.top_p="'"${generation_top_p}"'" \
      trainer.generation_eval.top_k="'"${generation_top_k}"'" \
      trainer.generation_eval.n="'"${generation_num_samples}"'" \
      trainer.generation_eval.dtype="'"${generation_dtype}"'" \
      trainer.generation_eval.backend="'"${generation_backend}"'" \
      trainer.generation_eval.vllm_gpu_memory_utilization="'"${generation_vllm_gpu_memory_utilization}"'" \
      trainer.generation_eval.vllm_host_ip="'"${generation_vllm_host_ip}"'" \
      trainer.generation_eval.vllm_enforce_eager="'"${generation_vllm_enforce_eager}"'" \
      trainer.generation_eval.vllm_sync_weights="'"${generation_vllm_sync_weights}"'" \
      trainer.generation_eval.vllm_enable_multiprocessing="'"${generation_vllm_enable_multiprocessing}"'" \
      trainer.logger="[\"console\",\"wandb\"]" \
      "$@"
  ' _ "${SPARSE_UPDATE_MASK_OVERRIDES[@]}" "$@" 2>&1 | tee "$TRAIN_STDOUT_LOG"
