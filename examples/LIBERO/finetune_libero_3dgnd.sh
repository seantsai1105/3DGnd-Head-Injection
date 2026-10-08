#!/usr/bin/env bash
# Fine-tune GR00T N1.6 with 3-D grounding injection (3DGnd) on the LIBERO pick-and-place data,
# using the hyper-parameters of the checkpoint evaluated in the paper: effective batch size 384
# (2 GPUs x 192), 20k steps, AdamW with lr 1e-4, cosine schedule, 5% warmup and weight decay 1e-5.
#
# The effective batch size stays 384 for any NUM_GPUS x GRAD_ACCUM that divides 384 (every GPU
# then processes 384 / (NUM_GPUS x GRAD_ACCUM) samples per forward/backward pass).
#
# The paper checkpoint was fine-tuned from nvidia/GR00T-N1.6-3B at revision
# d0814e7ecb19202e7c8468b46098b0b7ef3a6d61. To pin it, download that revision and set
# BASE_MODEL_PATH to the local copy:
#   uv run hf download nvidia/GR00T-N1.6-3B \
#       --revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 --local-dir <dir>
#
# Usage (extra arguments are passed on to launch_finetune.py, e.g. --use_wandb):
#   DATASET_PATH=<lerobot dataset> [OUTPUT_DIR=./outputs/libero_3dgnd] [NUM_GPUS=2] [GRAD_ACCUM=1] \
#       [BASE_MODEL_PATH=nvidia/GR00T-N1.6-3B] [MASTER_PORT=29500] \
#       uv run bash examples/LIBERO/finetune_libero_3dgnd.sh [extra args]
set -euo pipefail

: "${DATASET_PATH:?Set DATASET_PATH to the 3DGnd LIBERO LeRobot dataset}"
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/libero_3dgnd}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}
NUM_GPUS=${NUM_GPUS:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EFFECTIVE_BATCH_SIZE=384
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if (( EFFECTIVE_BATCH_SIZE % (NUM_GPUS * GRAD_ACCUM) != 0 )); then
    echo "NUM_GPUS * GRAD_ACCUM must divide the effective batch size ${EFFECTIVE_BATCH_SIZE}" >&2
    exit 1
fi

# launch_finetune.py's --global_batch_size is summed over GPUs but excludes gradient accumulation
torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" \
    "$REPO_ROOT/gr00t/experiment/launch_finetune.py" \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path "$REPO_ROOT/examples/LIBERO/libero_3dgnd_config.py" \
    --use_grounding \
    --num_gpus "$NUM_GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 20000 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --global_batch_size $((EFFECTIVE_BATCH_SIZE / GRAD_ACCUM)) \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 8 \
    "$@"
