import argparse
import os
import random
import shutil
import time
import warnings

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='PyTorch Generate List of Indices (Random)')
parser.add_argument('--data_root', default='data')
parser.add_argument('--dataset', default='svhn')
parser.add_argument('--num_samples', default=50000, type=int,
                    help='number of images used during training (default = 50k)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training.')


def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    main_worker(args)


def main_worker(args):

    _, train_dataset, _ = registry.get_dataset(name=args.dataset, data_root=args.data_root)

    print('The ' + str(args.dataset) + ' dataset has {} images'.format( len(train_dataset)))
    subset_indices = torch.randperm(len(train_dataset))[:args.num_samples]
    print(subset_indices)
    print('We use in the list :', len(subset_indices))
    torch.save(subset_indices, './Random_Sampling_List/'+args.dataset+'_'+str(args.num_samples)+'_image_list.pt')

    
if __name__ == '__main__':
    main()