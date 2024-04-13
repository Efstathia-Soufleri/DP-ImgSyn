CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python datafree_kd_public_img_init_DP_mnist.py \
--method deepinv \
--dataset mnist \
--dataset_init places365_32x32_mnist_transformations \
--batch_size 80 \
--teacher resnet18 \
--student resnet18 \
--epsilon 10 \
--lr 0.1 \
--epochs 625 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 10 \
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
--save_dir /local/a/<set_path>/run/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter \
--log_tag deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter \
--path_save_img /local/a/<set_path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/img/ \
--path_save_targets /local/a/<set_path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/targets/