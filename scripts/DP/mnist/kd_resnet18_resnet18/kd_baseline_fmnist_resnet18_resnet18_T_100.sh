for epsilon in 1 10
do
    for seed in 111 2023 50173
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python kd_baseline_iso_num_images_DP.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset mnist \
        --transfer_set fmnist_mnist_transformations \
        --batch_size 256 \
        --epsilon $epsilon \
        --T 100 \
        --num_samples 60000 \
        --workers 16 \
        --log_tag kd_baseline_fmnist_mnist_60k_resnet18_resnet18_10_${epsilon}_T_100_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/mnist/eps_${epsilon}/fmnist_init/baseline/kd_baseline_fmnist_mnist_60k_resnet18_resnet18_${epsilon}_epsilon_T_100_seed${seed}_g1.txt
    done
done


