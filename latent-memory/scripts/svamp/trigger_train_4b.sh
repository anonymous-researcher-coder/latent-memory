#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Trigger GRPO training with a frozen Weaver.
SEED="${SEED:-52}"
export PYTHONHASHSEED="${SEED}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export NCCL_SHM_DISABLE=1
REASONER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
WEAVER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TRIGGER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TRIGGER_EPSILON_GREEDY=0
TRIGGER_FORCE_ACTION=null
DATASET_NAME="svamp"
DATASET_MODE="grpo"

# official: original data; otherwise data/<dataset>/<name>.jsonl.
DATA_NAME="${DATA_NAME:-official}"

# Enable joint Trigger and compressor training when True.
COMPRESS_LATENT_MEMORY=False

# Merge this many adjacent latents; must divide both memory lengths.
COMPRESSOR_GROUP_SIZE="${COMPRESSOR_GROUP_SIZE:-4}"
COMPRESSOR_ROLLOUT_BATCH_SIZE="${COMPRESSOR_ROLLOUT_BATCH_SIZE:-4}"
TRAIN_METHOD="grpo"

# Set LOAD_WEAVER_PATH to the trained Weaver checkpoint directory.
LOAD_WEAVER_PATH="${LOAD_WEAVER_PATH:?Set the checkpoint path}"

# Insertion limits, followed by latent tokens per insertion.
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

# DAR: -1 reads the final layer; middle reads an intermediate layer.
WEAVER_MEMORY_LAYER=-1

# null: fixed layer; "-1,middle": candidate layers.
WEAVER_RANDOM_MEMORY_LAYERS=null

# RSAT: entropy_position; none uses delimiter positions.
ENTROPY_TRIGGER_MODE="${ENTROPY_TRIGGER_MODE:-none}"

# Minimum tokens between insertions; 0 disables spacing.
ENTROPY_INSERT_MIN_DISTANCE="${ENTROPY_INSERT_MIN_DISTANCE:-0}"

# Use a value, or a percentile key plus an existing threshold file.
ENTROPY_THRESHOLD_VALUE="${ENTROPY_THRESHOLD_VALUE:-null}"
ENTROPY_THRESHOLD_PERCENTILE="${ENTROPY_THRESHOLD_PERCENTILE:-P80}"
ENTROPY_THRESHOLD_FILE="${ENTROPY_THRESHOLD_FILE:-results/entropy_threshold/svamp/Qwen_Qwen3-4B-Instruct-2507_n0/threshold.json}"

# Optional upper bound: use a value or a percentile key.
ENTROPY_THRESHOLD_UPPER_VALUE="${ENTROPY_THRESHOLD_UPPER_VALUE:-null}"
ENTROPY_THRESHOLD_UPPER_PERCENTILE="${ENTROPY_THRESHOLD_UPPER_PERCENTILE:-P95}"

# Split rollout batches to reduce peak GPU memory.
GENERATE_SUB_BATCH=False

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path "configs/latent_memory/${DATASET_NAME}.yaml" \
    --options \
    run.seed "${SEED}" \
    model.model_name "${REASONER_MODEL}" \
    model.load_model_path "${LOAD_WEAVER_PATH}" \
    model.compress_latent_memory "${COMPRESS_LATENT_MEMORY}" \
    model.compressor.group_size "${COMPRESSOR_GROUP_SIZE}" \
    model.compressor.rollout_batch_size "${COMPRESSOR_ROLLOUT_BATCH_SIZE}" \
    model.max_prompt_aug_num "${MAX_PROMPT_AUG_NUM}" \
    model.max_inference_aug_num "${MAX_INFERENCE_AUG_NUM}" \
    model.weaver.model_name "${WEAVER_MODEL}" \
    model.weaver.prompt_latents_len "${PROMPT_LATENTS_LEN}" \
    model.weaver.inference_latents_len "${INFERENCE_LATENTS_LEN}" \
    model.weaver.memory_layer "${WEAVER_MEMORY_LAYER}" \
    model.weaver.random_memory_layers="${WEAVER_RANDOM_MEMORY_LAYERS}" \
    model.trigger.model_name "${TRIGGER_MODEL}" \
    model.trigger.active True \
    model.trigger.epsilon_greedy "${TRIGGER_EPSILON_GREEDY}" \
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
    dataset.mode "${DATASET_MODE}" \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method "${TRAIN_METHOD}" \
    run.trigger.grpo.per_device_train_batch_size 2 \
    run.trigger.grpo.per_device_eval_batch_size 2 \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.num_generations 8 \
    run.trigger.grpo.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    run.trigger.grpo.generate_sub_batch "${GENERATE_SUB_BATCH}" \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024
