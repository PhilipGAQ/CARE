#!/bin/bash
export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ROOT="/path/to/project"
CODE_DIR="$ROOT/train"

cd "$CODE_DIR"

MODEL_QUERY="path/to/query/encoder"       # e.g., 0.3B parameter model
MODEL_DOC="path/to/doc/encoder"           # e.g., 8B parameter model

TRAIN_DATA="$ROOT/data/stage1"

SAVE_DIR="$ROOT/output/stage1-query-alignment"
CACHE_DIR="$ROOT/cache"

# ======== TRAINING CONFIGURATION ==========
FIX_DOC_ENCODER=True
kd_loss_type='contrastive_and_mse'

'''
Use this when running on multiple nodes:
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --nproc_per_node 8 \
'''
torchrun --nproc_per_node 8 \
main.py \
    --model_name_or_path_query $MODEL_QUERY \
    --trust_remote_code_query True \
    --model_name_or_path_doc $MODEL_DOC \
    --trust_remote_code_doc True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_flash_attn True \
    --cache_dir $CACHE_DIR \
    --save_merged_lora_model True \
    --train_data $TRAIN_DATA \
    --cache_path $CACHE_DIR \
    --train_group_size 1 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --negatives_cross_device \
    --small_threshold 0 \
    --drop_threshold 0 \
	--output_dir $SAVE_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 40 \
    --sub_batch_size 20 \
    --dataloader_drop_last True \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --gradient_checkpointing \
    --deepspeed $ROOT/ds_stage_0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --temperature 0.1 \
    --sentence_pooling_method 'cls' \
    --normalize_embeddings True \
    --kd_loss_type $kd_loss_type \
    --fix_doc_encoder $FIX_DOC_ENCODER \
    --use_mrl False \
    --mrl_dims '64,128,256,512,768' \
    --k 10.0