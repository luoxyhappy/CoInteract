#!/bin/bash
# ============================================================
# CoInteract Training Script
# Trains Human-Aware MoE + Audio Face Mask + Depth Branch
# using DeepSpeed ZeRO-2
# ============================================================

# Resolve repo root from this script's location so the script works from anywhere.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# --- Model Paths (relative to repo root; see README "Model Weights" section) ---
BASE_MODEL_PATH="./models/Wan2.2-S2V-14B"
AUDIO_ENCODER_PATH="./models/chinese-wav2vec2-large"
TOKENIZER_PATH="${BASE_MODEL_PATH}/google/umt5-xxl"

# --- Training Data (point to the bundled demodataset by default) ---
DATASET_BASE_PATH="./examples/wanvideo/model_training/demodataset"
DATASET_CSV="${DATASET_BASE_PATH}/data.csv"
OUTPUT_PATH="./output"

# --- Export env vars for audio encoder & tokenizer ---
export AUDIO_ENCODER_DIR="${AUDIO_ENCODER_PATH}"
export TOKENIZER_DIR="${TOKENIZER_PATH}"

# --- Model file list (JSON format) ---
MODEL_PATHS="[
  \"${BASE_MODEL_PATH}/models_t5_umt5-xxl-enc-bf16.pth\",
  \"${BASE_MODEL_PATH}/Wan2.1_VAE.pth\",
  \"${AUDIO_ENCODER_PATH}/pytorch_model.bin\",
  [
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00001-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00002-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00003-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00004-of-00004.safetensors\"
  ]
]"

# --- Launch Training ---
deepspeed --num_gpus=8 \
    examples/wanvideo/model_training/train.py \
    --deepspeed_config ds_config.json \
    --model_paths "${MODEL_PATHS}" \
    --dataset_metadata_path "${DATASET_CSV}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --height 720 \
    --width 480 \
    --num_frames 81 \
    --extra_inputs "person_image,product_image" \
    --trainable_models "dit" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 128 \
    --use_moe \
    --expert_hidden_dim 256 \
    --use_hoi_branch \
    --depth_mutual_visible \
    --train_shift 5.0 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --save_steps 500 \
    --gradient_accumulation_steps 1
