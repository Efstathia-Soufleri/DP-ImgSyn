#!/bin/bash

for epsilon in 10
do
    for seed in 111 2023 50173
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 python train_kd_synthetic_img_DP_cifar10.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset cifar10 \
        --transfer_set deepinv_cifar10_tinyimagenet_init_resnet18_resnet18_${epsilon}_eps_1_iter \
        --batch_size 256 \
        --epsilon $epsilon \
        --T 100 \
        --workers 16 \
        --log_tag kd_synthetic_cifar10_eps_${epsilon}_resnet18_resnet18_init_tinyimagenet_T_100_iter_1_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/cifar10/eps_${epsilon}/tinyimagenet_init/iter_10/kd_synthetic_cifar10_eps_${epsilon}_resnet18_resnet18_init_tinyimagenet_T_100_iter_1_seed${seed}_g1.txt
    done
done




