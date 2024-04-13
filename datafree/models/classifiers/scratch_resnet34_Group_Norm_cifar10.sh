CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python train_scratch.py \
--seed 1 \
--model resnet34_GroupNorm \
--dataset cifar10 