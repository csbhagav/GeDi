#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run_summarization.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/l2r_clf/moral_agreement/train.json \
    --validation_file ./data/l2r_clf/moral_agreement/validation.json \
    --test_file ./data/l2r_clf/moral_agreement/test.json \
    --source_prefix "" \
    --output_dir ./cls_models \
    --overwrite_output_dir \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --max_source_length 256 \
    --max_target_length 4 \
    --val_max_target_length 4 \
    --num_train_epochs 3 \
    --save_strategy no \
    --predict_with_generate

