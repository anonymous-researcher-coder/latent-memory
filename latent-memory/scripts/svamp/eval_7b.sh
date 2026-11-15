#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Provide data and checkpoints; edit settings below.
# RSAT: entropy_position or none; match train/eval thresholds.
SEED="${SEED:-52}"
export PYTHONHASHSEED="${SEED}"
export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29507}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_SHM_DISABLE=1
REASONER_MODEL="Qwen/Qwen2.5-7B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TRIGGER_ACTIVE=True
TRIGGER_FORCE_ACTION=null
DATASET_NAME="svamp"
DATA_NAME="${DATA_NAME:-official}"
COMPRESS_LATENT_MEMORY=False
COMPRESSOR_GROUP_SIZE="${COMPRESSOR_GROUP_SIZE:-4}"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:?Set the checkpoint path}"
WEAVER_MEMORY_LAYER=-1
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8
TEMPERATURE=1.0
DO_SAMPLE=True
REPETITION_PENALTY=1.0
MAX_RESPONSE_LENGTH=1024
EVAL_REPEATS=1
EVAL_SEED_STRIDE=1
WEAVER_RANDOM_MEMORY_LAYERS=null
ENTROPY_TRIGGER_MODE="${ENTROPY_TRIGGER_MODE:-entropy_position}"
ENTROPY_INSERT_MIN_DISTANCE="${ENTROPY_INSERT_MIN_DISTANCE:-16}"
ENTROPY_THRESHOLD_VALUE="${ENTROPY_THRESHOLD_VALUE:-null}"
ENTROPY_THRESHOLD_PERCENTILE="${ENTROPY_THRESHOLD_PERCENTILE:-P80}"
ENTROPY_THRESHOLD_FILE="${ENTROPY_THRESHOLD_FILE:-results/entropy_threshold/svamp/Qwen_Qwen2.5-7B-Instruct_n0/threshold.json}"
ENTROPY_THRESHOLD_UPPER_VALUE="${ENTROPY_THRESHOLD_UPPER_VALUE:-null}"
ENTROPY_THRESHOLD_UPPER_PERCENTILE="${ENTROPY_THRESHOLD_UPPER_PERCENTILE:-P95}"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
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
