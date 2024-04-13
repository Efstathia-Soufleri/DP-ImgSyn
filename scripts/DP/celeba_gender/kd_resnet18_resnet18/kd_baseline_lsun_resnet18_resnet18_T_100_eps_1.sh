#!/bin/bash

for seed in 1 2 3
do
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python kd_baseline_iso_num_images_DP_celeba.py \
    --seed $seed \
    --teacher resnet18 \
    --student resnet18 \
    --dataset celeba_gender \
    --transfer_set lsun_celeba_gender_transformations \
    --batch_size 256 \
    --epsilon 1 \
    --T 100 \
    --num_samples 162770 \
    --workers 16 \
    --log_tag kd_baseline_lsun_celeba_gender_resnet18_resnet18_1_epsilon_T_100_seed${seed}_g1 \
    | tee ./logs/DP/resnet18_resnet18/celeba_gender/eps_1/lsun_init/baseline/kd_baseline_lsun_celeba_gender_resnet18_resnet18_1_epsilon_T_100_seed${seed}_g1.txt
done

