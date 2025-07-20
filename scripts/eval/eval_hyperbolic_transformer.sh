#!/bin/bash

# Evaluation script for hyperbolic transformer models
# This script evaluates a trained hyperbolic transformer model

python main.py \
--test \
--test_info compute_accuracy \
--dataset kinetics \
--seq_len 5 \
--num_seq 8 \
--pred_step 3 \
--network_feature resnet18 \
--feature_dim 256 \
--distance regular \
--hyperbolic \
--hyperbolic_version 1 \
--use_transformer \
--num_heads 8 \
--transformer_layers 4 \
--transformer_dropout 0.1 \
--fp64_hyper \
--batch_size 8 \
--resume /path/to/trained/model.pth \
--path_dataset /path/to/kinetics \
--path_data_info /path/to/dataset_info \
--img_dim 128 \
--use_labels \
--n_classes 600