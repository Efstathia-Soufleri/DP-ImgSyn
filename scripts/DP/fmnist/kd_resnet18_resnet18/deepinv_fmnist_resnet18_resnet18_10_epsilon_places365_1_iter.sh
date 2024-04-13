#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python datafree_kd_public_img_init_DP_fmnist.py \
--method deepinv \
--dataset fmnist \
--dataset_init places365_32x32_fmnist_transformations \
--batch_size 80 \
--teacher resnet18 \
--student resnet18 \
--epsilon 10 \
--lr 0.1 \
--epochs 625 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.1 \
--adv 0 \
--bn 10 \
--oh 1 \
--T 20 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 1 \
--num_samples 60000 \
--save_dir run/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter \
--log_tag deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter \
--path_save_img /local/a/<set_path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/ \
--path_save_targets /local/a/<set_path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/