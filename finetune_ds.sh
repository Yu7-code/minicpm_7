#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# 添加DeepSpeed配置文件路径
DS_CONFIG_PATH="/workspace/mnt/public/usr/yuzhiqi/minicpm-dev/ds_config_zero3.json"  

MODEL="/workspace/mnt/public/usr/yuzhiqi/minicpm-dev/" 


DATA="/workspace/mnt/public/usr/luoran/dataset/rec_note_embedding/click_with_likepair_taxonomy/pairinfo"
EVAL_DATA="/workspace/mnt/public/usr/luoran/dataset/rec_note_embedding/click_with_likepair_taxonomy/eval"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# export CUDA_VISIBLE_DEVICES=1

# deepspeed --num_gpus=1 finetune1.py \
torchrun $DISTRIBUTED_ARGS finetune1.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only  false \
	--fp16 false \
    --bf16 false \
    --bf16_full_eval false \
    --do_train \
    --do_eval \
    --max_steps 80000 \
    --eval_steps 200 \
    --output_dir output/output_minicpmv2 \
    --logging_dir output/output_minicpmv2 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 5e-7 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --report_to "tensorboard" # wandb
