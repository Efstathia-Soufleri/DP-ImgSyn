
for epsilon in 10
do
    for seed in 111 2023 50173
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 python kd_baseline_iso_num_images_DP.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset cifar10 \
        --transfer_set tinyimagenet_cifar10_transformations \
        --batch_size 256 \
        --epsilon $epsilon \
        --T 100 \
        --num_samples 50000 \
        --workers 16 \
        --log_tag kd_baseline_tinyimagenet_cifar10_resnet18_resnet18_${epsilon}_epsilon_T_100_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/cifar10/eps_${epsilon}/tinyimagenet_init/baseline/kd_baseline_tinyimagenet_cifar10_resnet18_resnet18_${epsilon}_epsilon_T_100_seed${seed}_g1.txt
    done
done