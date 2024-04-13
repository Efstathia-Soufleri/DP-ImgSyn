#!/bin/bash

for epsilon in 104
do
    for seed in 111 2023 50173
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python train_kd_synthetic_img_DP.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset imagenette \
        --transfer_set deepinv_imagenette_textures_init_resnet18_resnet18_${epsilon}_eps_1_iter \
        --batch_size 32 \
        --epsilon $epsilon \
        --T 100 \
        --workers 16 \
        --log_tag kd_synthetic_imagenette_eps_${epsilon}_resnet18_resnet18_init_textures_T_100_iter_1_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/imagenette/eps_${epsilon}/textures_init/iter_1/kd_synthetic_imagenette_eps_${epsilon}_resnet18_resnet18_init_textures_T_100_iter_1_seed${seed}_g1.txt
    done
done




