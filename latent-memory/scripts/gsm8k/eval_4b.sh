#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Evaluation; match memory and RSAT settings to training.
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"
export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_SHM_DISABLE=1
REASONER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
WEAVER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TRIGGER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TRIGGER_ACTIVE=True
TRIGGER_FORCE_ACTION=null
DATASET_NAME="gsm8k"

# official: original data; otherwise data/<dataset>/<name>.jsonl.
DATA_NAME="${DATA_NAME:-gsm8k_flat_gpt4o}"

# Match the checkpoint compression settings.
COMPRESS_LATENT_MEMORY=False

# Merge this many adjacent latents; must divide both memory lengths.
COMPRESSOR_GROUP_SIZE="${COMPRESSOR_GROUP_SIZE:-4}"

# Set LOAD_MODEL_PATH to the checkpoint directory before launching.
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:?Set the checkpoint path}"

# DAR: -1 reads the final layer; middle reads an intermediate layer.
WEAVER_MEMORY_LAYER=-1

# Insertion limits, followed by latent tokens per insertion.
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8
TEMPERATURE=1.0
DO_SAMPLE=False
REPETITION_PENALTY=1.0
MAX_RESPONSE_LENGTH=2048
EVAL_REPEATS=1
EVAL_SEED_STRIDE=1

# null: fixed layer; "-1,middle": candidate layers.
WEAVER_RANDOM_MEMORY_LAYERS=null

# RSAT: entropy_position; none uses delimiter positions.
ENTROPY_TRIGGER_MODE="${ENTROPY_TRIGGER_MODE:-none}"

# Use a value, or a percentile key plus an existing threshold file.
ENTROPY_THRESHOLD_VALUE="${ENTROPY_THRESHOLD_VALUE:-null}"
ENTROPY_THRESHOLD_PERCENTILE="${ENTROPY_THRESHOLD_PERCENTILE:-null}"
ENTROPY_THRESHOLD_FILE="${ENTROPY_THRESHOLD_FILE:-null}"

# Optional upper bound: use a value or a percentile key.
ENTROPY_THRESHOLD_UPPER_VALUE="${ENTROPY_THRESHOLD_UPPER_VALUE:-null}"
ENTROPY_THRESHOLD_UPPER_PERCENTILE="${ENTROPY_THRESHOLD_UPPER_PERCENTILE:-null}"

# Minimum tokens between insertions; 0 disables spacing.
ENTROPY_INSERT_MIN_DISTANCE="${ENTROPY_INSERT_MIN_DISTANCE:-0}"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path "configs/latent_memory/${DATASET_NAME}.yaml" \
    --options \
    run.seed "${SEED}" \
    model.model_name "${REASONER_MODEL}" \
    model.load_model_path "${LOAD_MODEL_PATH}" \
    model.compress_latent_memory "${COMPRESS_LATENT_MEMORY}" \
    model.compressor.group_size "${COMPRESSOR_GROUP_SIZE}" \
    model.max_prompt_aug_num "${MAX_PROMPT_AUG_NUM}" \
    model.max_inference_aug_num "${MAX_INFERENCE_AUG_NUM}" \
    model.weaver.model_name "${WEAVER_MODEL}" \
    model.weaver.prompt_latents_len "${PROMPT_LATENTS_LEN}" \
    model.weaver.inference_latents_len "${INFERENCE_LATENTS_LEN}" \
    model.weaver.memory_layer "${WEAVER_MEMORY_LAYER}" \
    model.weaver.random_memory_layers="${WEAVER_RANDOM_MEMORY_LAYERS}" \
    model.trigger.model_name "${TRIGGER_MODEL}" \
    model.trigger.active "${TRIGGER_ACTIVE}" \
    model.trigger.force_action "${TRIGGER_FORCE_ACTION}" \
    model.entropy_trigger.mode "${ENTROPY_TRIGGER_MODE}" \
    model.entropy_trigger.threshold_value "${ENTROPY_THRESHOLD_VALUE}" \
    model.entropy_trigger.threshold_percentile "${ENTROPY_THRESHOLD_PERCENTILE}" \
    model.entropy_trigger.threshold_file "${ENTROPY_THRESHOLD_FILE}" \
    model.entropy_trigger.threshold_upper_value "${ENTROPY_THRESHOLD_UPPER_VALUE}" \
    model.entropy_trigger.threshold_upper_percentile "${ENTROPY_THRESHOLD_UPPER_PERCENTILE}" \
    model.entropy_trigger.insert_min_distance "${ENTROPY_INSERT_MIN_DISTANCE}" \
    dataset.sft.data_name "${DATA_NAME}" \
    dataset.grpo.data_name "${DATA_NAME}" \
    run.mode evaluate \
    run.train_weaver False \
    run.train_trigger False \
    run.interaction.batch_size 1 \
    run.interaction.do_sample "${DO_SAMPLE}" \
    run.interaction.temperature "${TEMPERATURE}" \
    run.interaction.eval_repeats "${EVAL_REPEATS}" \
    run.interaction.eval_seed_stride "${EVAL_SEED_STRIDE}" \
    run.interaction.repetition_penalty "${REPETITION_PENALTY}" \
    run.interaction.max_response_length "${MAX_RESPONSE_LENGTH}"
