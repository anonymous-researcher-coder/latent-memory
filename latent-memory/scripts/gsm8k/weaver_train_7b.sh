#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Provide data and checkpoints; edit settings below.
# RSAT: entropy_position or none; match train/eval thresholds.
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"
export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export NCCL_SHM_DISABLE=1
REASONER_MODEL="Qwen/Qwen2.5-7B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="gsm8k"
DATA_NAME="${DATA_NAME:-official}"
TRAIN_METHOD="sft"
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8
WEAVER_MEMORY_LAYER=middle
WEAVER_RANDOM_MEMORY_LAYERS=null
ENTROPY_TRIGGER_MODE="${ENTROPY_TRIGGER_MODE:-entropy_position}"
ENTROPY_INSERT_MIN_DISTANCE="${ENTROPY_INSERT_MIN_DISTANCE:-16}"
ENTROPY_THRESHOLD_VALUE="${ENTROPY_THRESHOLD_VALUE:-null}"
ENTROPY_THRESHOLD_PERCENTILE="${ENTROPY_THRESHOLD_PERCENTILE:-P80}"
ENTROPY_THRESHOLD_FILE="${ENTROPY_THRESHOLD_FILE:-null}"
ENTROPY_THRESHOLD_UPPER_VALUE="${ENTROPY_THRESHOLD_UPPER_VALUE:-null}"
ENTROPY_THRESHOLD_UPPER_PERCENTILE="${ENTROPY_THRESHOLD_UPPER_PERCENTILE:-P95}"
BATCH_SIZE=1

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path "configs/latent_memory/${DATASET_NAME}.yaml" \
    --options \
    run.seed "${SEED}" \
    model.model_name "${REASONER_MODEL}" \
    model.max_prompt_aug_num "${MAX_PROMPT_AUG_NUM}" \
    model.max_inference_aug_num "${MAX_INFERENCE_AUG_NUM}" \
    model.weaver.model_name "${WEAVER_MODEL}" \
    model.weaver.prompt_latents_len "${PROMPT_LATENTS_LEN}" \
    model.weaver.inference_latents_len "${INFERENCE_LATENTS_LEN}" \
    model.weaver.memory_layer "${WEAVER_MEMORY_LAYER}" \
    model.weaver.random_memory_layers="${WEAVER_RANDOM_MEMORY_LAYERS}" \
    model.trigger.model_name "${TRIGGER_MODEL}" \
    model.trigger.active False \
    model.entropy_trigger.mode "${ENTROPY_TRIGGER_MODE}" \
    model.entropy_trigger.threshold_value "${ENTROPY_THRESHOLD_VALUE}" \
    model.entropy_trigger.threshold_percentile "${ENTROPY_THRESHOLD_PERCENTILE}" \
    model.entropy_trigger.threshold_file "${ENTROPY_THRESHOLD_FILE}" \
    model.entropy_trigger.threshold_upper_value "${ENTROPY_THRESHOLD_UPPER_VALUE}" \
    model.entropy_trigger.threshold_upper_percentile "${ENTROPY_THRESHOLD_UPPER_PERCENTILE}" \
    model.entropy_trigger.insert_min_distance "${ENTROPY_INSERT_MIN_DISTANCE}" \
    dataset.sft.data_name "${DATA_NAME}" \
    dataset.grpo.data_name "${DATA_NAME}" \
    dataset.mode "${TRAIN_METHOD}" \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method "${TRAIN_METHOD}" \
    run.weaver.sft.per_device_train_batch_size "${BATCH_SIZE}" \
    run.weaver.sft.per_device_eval_batch_size "${BATCH_SIZE}" \
    run.weaver.sft.bf16 True \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 512
