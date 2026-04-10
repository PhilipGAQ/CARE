#!/bin/bash
export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ROOT="/path/to/project"
CODE_DIR="$ROOT/train"

cd "$CODE_DIR"

MODEL_QUERY="$ROOT/output/stage1-query-alignment/query_encoder"
MODEL_DOC="path/to/doc/encoder"

TRAIN_DATA="$ROOT/data/stage2"

SAVE_DIR="$ROOT/output/stage2-joint-finetuning"
CACHE_DIR="$ROOT/cache"

FIX_DOC_ENCODER=False

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
    --train_group_size 8 \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 21 \
    --sub_batch_size 3 \
    --dataloader_drop_last True \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --gradient_checkpointing \
    --deepspeed $ROOT/ds_stage_0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --temperature 0.02 \
    --sentence_pooling_method 'cls' \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --fix_doc_encoder $FIX_DOC_ENCODER \
    --use_mrl False \
    --mrl_dims '64,128,256,512,768' \
    --k 10.0
