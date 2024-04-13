#!/bin/bash

for epsilon in 1
do
    for seed in 1 2 3
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python train_kd_synthetic_img_DP_celeba.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset celeba_hair \
        --transfer_set deepinv_celeba_hair_places365_init_resnet18_resnet18_${epsilon}_eps_1_iter \
        --batch_size 256 \
        --epsilon $epsilon \
        --T 100 \
        --workers 16 \
        --log_tag kd_synthetic_celeba_hair_eps_${epsilon}_resnet18_resnet18_init_places365_T_100_iter_1_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/celeba_hair/eps_${epsilon}/places365_init/iter_1/kd_synthetic_celeba_hair_eps_${epsilon}_resnet18_resnet18_init_places365_T_100_iter_1_seed${seed}_g1.txt
    done
done




