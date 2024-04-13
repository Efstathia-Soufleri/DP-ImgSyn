
for epsilon in 10
do
    for seed in 1 2 3
    do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 python kd_baseline_iso_num_images_DP_celeba.py \
        --seed $seed \
        --teacher resnet18 \
        --student resnet18 \
        --dataset celeba_hair \
        --transfer_set places365_32x32_celeba_hair_transformations \
        --batch_size 256 \
        --epsilon $epsilon \
        --T 100 \
        --num_samples 162770 \
        --workers 16 \
        --log_tag kd_baseline_places365_celeba_hair_resnet18_resnet18_${epsilon}_epsilon_T_100_seed${seed}_g1 \
        | tee ./logs/DP/resnet18_resnet18/celeba_hair/eps_${epsilon}/places365_init/baseline/kd_baseline_places365_celeba_hair_resnet18_resnet18_${epsilon}_epsilon_T_100_seed${seed}_g1.txt
    done
done