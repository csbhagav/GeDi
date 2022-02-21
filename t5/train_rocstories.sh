#!/bin/bash

#CUDA_VISIBLE_DEVICES=2 python run_summarization.py \
python run_summarization.py \
    --model_name_or_path t5-11b \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/rocstories_gen/json_files/train.json \
    --validation_file ./data/rocstories_gen/json_files/val.json \
    --test_file ./data/rocstories_gen/json_files/test.json \
    --source_prefix "" \
    --output_dir ./roc_11b_models \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 256 \
    --val_max_target_length 256 \
    --num_train_epochs 10 \
    --save_strategy no \
    --predict_with_generate

