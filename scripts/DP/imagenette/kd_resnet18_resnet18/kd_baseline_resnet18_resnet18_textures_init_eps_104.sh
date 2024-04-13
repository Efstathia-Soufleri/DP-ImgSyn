
for seed in 111 2023 50173
do
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python kd_baseline_iso_num_images_DP_imagenette.py \
    --seed $seed \
    --teacher resnet18 \
    --student resnet18 \
    --dataset imagenette \
    --transfer_set textures_imagenette_transformations \
    --batch_size 16 \
    --epsilon 104 \
    --T 100 \
    --num_samples 5640 \
    --workers 16 \
    --log_tag kd_baseline_textures_imagenette_resnet18_resnet18_104_epsilon_T_100_seed${seed}_g1 \
    | tee ./logs/DP/resnet18_resnet18/imagenette/eps_104/textures_init/baseline/kd_baseline_textures_imagenette_resnet18_resnet18_104_epsilon_T_100_seed${seed}_g1.txt
done

