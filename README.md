# DP-ImgSyn

This code is related to the paper titled, "DP-ImgSyn: Dataset Alignment for Obfuscated, Differentially Private Image Synthesis"


**A non-generative Differentially Private Image Synthesis (DP-ImgSyn) framework to sanitize and release confidential data with DP guarantees to address these issues.**

### Setup

We provide an environment.yml file to recreate the python environment used in our simulations. To setup the environment use:
```bash
conda env create -f environment.yml
```

Moreover, we provide the lists for the subsampling of the public datasets at [Public dataset Random Sampling List](https://drive.google.com/file/d/15ndOOXaf4n6dAM4ZM9jOgN-zF2tg5YWG/view?usp=sharing). Please download, unzip and save it in the DP_ImgSyn folder such that the folder structure looks like:

```
DP_ImgSyn
├── Random_Sampling_List
├── checkpoints
│   └── pretrained_DP_models
.
.
```

### Running the scripts

1. DP Teacher Model Training

* Train with DP-SGD, To train a resnet18 with DP-SGD use the following command:
```bash
python train_dp.py --dataset=<dataset> --epochs=<ηtrain> --train_batch_size=<Ωtrain> --lr=<γtrain> --max_norm=<C> --noise_multiplier=<σ> --model=resnet18 --val_split=0.0
```

* Capture batch stats with DP guarantee:

```
python save_dp_db_stats.py --dataset=<dataset> --epochs=<Ωbn> --lr=<γtr> --max_norm=<C> --noise_multiplier=<σ> --model=resnet18
```

This results in a model that is saved at a .dppt file in the checkpoints folder. For the parameters to use see Table 1.

Please note that there will be no terminal output when the program runs correctly, the outputs are logged to the ./log folder.

Further, please note the in the file ./synthesizer/GN_Record_Stats.py uncomment lines 51 - 58 and update line 60. This is needed to capture batch stats in a DP fashion.
```python
    def forward(self, input: Tensor) -> Tensor:
        if self.add_noise_to_bn:
            clamped_in = input.clamp(-self.max_norm, self.max_norm)
            noise = torch.normal(
                mean=0,
                std=self.std,
                size=input.shape,
                device=input.device,
                generator=None,
            )
            super().forward(clamped_in)
        else:
            # Means we are not adding noise when synthesizing data
            # should not happen but let's throw an error any way to be safe
            raise ValueError("Privacy Violation")
        return self.group_norm.forward(input) 
```
For image synthesis make sure to revert these changes.

You can skip the above steps and directly download, unzip, and access our DP trained models from [Pretrained DP Models Link](https://drive.google.com/file/d/1bHcxatKkZulETJyAkavI1cfBV6i4-DAq/view?usp=sharing). 

2. Image Synthesis:
To Synthesize Images for private dataset FashionMNIST with TinyImageNet as public dataset initialization with k = 1 optimization steps, epsilon = 10 run:

```shell
sh ./scripts/DP/fmnist/kd_resnet18_resnet18/deepinv_fmnist_resnet18_resnet18_10_epsilon_tinyimagenet_1_iter.sh
```
Please set in the arguments "--path_save_img" and "--path_save_targets" of the deepinv_fmnist_resnet18_resnet18_10_epsilon_tinyimagenet_1_iter.sh script the path that you would like to save the synthetic images and their labels ("<set_path>").

In general for all the results presented in the paper for synthetic dataset generation with epsilon $\epsilon$ and $k$ iterations the script is:
```shell
sh ./scripts/DP/<dataset>/kd_resnet18_resnet18/deepinv_<dataset>_resnet18_resnet18_10_<epsilon>_tinyimagenet_<iteration>_iter.sh
```

3. Training with Synthetic Images:

In the registry.py file, set the path name of the synthetic dataloader to the "<set_path>" path that you have set in step (2).

To train on the synthetic images generated with 1 iteration run:

```bash
sh ./scripts/DP/fmnist/kd_resnet18_resnet18/kd_DP_synthetic_training_tinyimagenet_resnet18_resnet18_T_100_iter_1.sh
```

To train on the synthetic images generated with 0 iteration run:

```bash
sh ./scripts/DP/fmnist/kd_resnet18_resnet18/kd_baseline_tinyimagenet_resnet18_resnet18_T_100.sh
```

Similar to (2) for the results presented in the paper, in general for all the results presented in the paper for synthetic dataset generation with epsilon $\epsilon$ and $k$ iterations the script is:
```bash
sh ./scripts/DP/<dataset>/kd_resnet18_resnet18/kd_DP_synthetic_training_<public_init>_resnet18_resnet18_T_100_iter_<iteration>.sh
```
This script will log the results for all the epsilon values i.e $\epsilon = \{1, 10\}$ and log the results.


## Hyper Parameters

Table 1: Hyper parameters for DP-training and batch stats

| Dataset       | $\epsilon$   | $\eta_{train}$ | $\Omega_{train}$ | $\gamma_{tr}$  | $C$   | $\sigma$    | $\eta_{bn}$ |  $\Omega_{bn}$ |
| ------------- | --- | --- | --- | ----- | --- | ---- | --- | --- |
| MNIST         | 1   | 4   | 128 | 0.01  | 1   | 0.8  | 2   | 64  |
|               | 10  | 14  | 128 | 0.01  | 1   | 0.5  | 2   | 64  |
| FMNIST        | 1   | 30  | 50  | 0.01  | 1.2 | 1    | 2   | 64  |
|               | 10  | 20  | 128 | 0.01  | 1.2 | 0.5  | 2   | 64  |
| CelebA Hair   | 1   | 18  | 128 | 0.001 | 1   | 0.8  | 5   | 128 |
|               | 10  | 22  | 128 | 0.001 | 1   | 0.45 | 3   | 128 |
| CelebA Gender | 1   | 18  | 128 | 0.001 | 1   | 0.8  | 4   | 128 |
|               | 10  | 22  | 128 | 0.001 | 1   | 0.5  | 4   | 128 |
| CIFAR-10      | 10  | 12  | 128 | 0.001 | 1   | 0.5  | 5   | 128 |
| ImageNette    | 105 | 57  | 8   | 0.001 | 1   | 0.3  | 3   | 8   |


## Results
The table below presents the mean ± std of the results over 3 seeds

| Private   Dataset | $\epsilon$ |  Public Dataset |  DP-ImgSyn(0)  |  DP-ImgSyn(kexp) |  kexp |  kopt |
|-------------------|----|-----------------|----------------|------------------|-------|-------|
| MNIST             | 1  | TinyImageNet    |  85.83 ± 0.13  |  85.98 ± 0.06    | 10    | 10    |
|                   | 1  | Places365       |  85.00 ± 0.30  |  86.01 ± 0.22    | 10    | 10    |
|                   | 1  | FashionMNIST    |  85.56 ± 0.32  |  86.24 ± 0.03    | 10    | 10    |
| MNIST             | 10 | TinyImageNet    |  92.97 ± 0.65  |  94.03 ± 0.64    | 10    | 10    |
|                   | 10 | Places365       |  92.63 ± 0.23  |  93.74 ± 0.18    | 10    | 10    |
|                   | 10 | FashionMNIST    |  93.61 ± 0.37  |  93.90 ± 0.30    | 10    | 10    |
| FashionMNIST      | 1  | TinyImageNet    |  74.99  ± 0.21 |  74.93 ± 0.02    | 1     | 0     |
|                   | 1  | Places365       |  75.08 ± 0.15  |  74.94 ± 0.20    | 1     | 0     |
|                   | 1  | MNIST           |  51.58 ± 2.28  |  68.38 ± 0.34    | 10    | 10    |
| FashionMNIST      | 10 | TinyImageNet    |  79.04 ± 0.04  |  78.71 ± 0.20    | 1     | 0     |
|                   | 10 | Places365       |  78.73 ± 0.14  |  78.80 ± 0.05    | 1     | 1     |
|                   | 10 | MNIST           |  54.78 ± 1.15  |  71.51 ± 1.49    | 10    | 10    |
| CelebA-Hair       | 1  | LSUN            |  78.09 ± 4.21  |  76.18 ± 4.49    | 1     | 0     |
|                   | 1  | Places365       |  79.88 ± 0.01  |  76.04 ± 5.07    | 1     | 0     |
| CelebA-Hair       | 10 |  LSUN           |  79.30 ± 4.80  |  75.73 ± 4.83    | 1     | 0     |
|                   | 10 | Places365       |  76.40 ± 5.81  |  77.40 ± 3.89    | 1     | 1     |
| CelebA-Gender     | 1  |  LSUN           |  70.93 ± 22.08 |  76.16 ± 14.45   | 1     | 1     |
|                   | 1  | Places365       |  81.35 ± 20.90 |  89.08 ± 0.42    | 1     | 1     |
| CelebA-Gender     | 10 |  LSUN           |  82.28 ± 21.37 |  80.80 ± 0.16    | 1     | 0     |
|                   | 10 | Places365       |  84.28 ± 16.50 |  88.79 ± 1.27    | 1     | 1     |
