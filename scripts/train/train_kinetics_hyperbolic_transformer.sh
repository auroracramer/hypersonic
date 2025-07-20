#!/bin/bash

python main.py \
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
--batch_size 4 \
--lr 0.0001 \
--wd 1e-5 \
--epochs 100 \
--prefix hyperbolic_transformer_kinetics \
--path_dataset /path/to/kinetics \
--path_data_info /path/to/dataset_info \
--img_dim 128