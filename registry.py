from datafree.models import classifiers, deeplab
from torchvision import datasets, transforms as T
from datafree.utils import sync_transforms as sT
from utils.celeba_hair import CelebAHair
from utils.celeba_gender import CelebAGender
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


import os
import torch
import torchvision
import datafree
import torch.nn as nn 
from PIL import Image

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'fmnist': dict( mean=(0.2860,),                std=(0.3530,) ),
    'colored_mnist':    dict( mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3081,0.3081) ),
    'colored_fmnist':    dict( mean=(0.2860,0.2860,0.2860), std=(0.3530,0.3530,0.3530) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'imagenette':  dict( mean=(0.4625, 0.4580, 0.4295), std=(0.2813, 0.2774, 0.3006) ),
    'synthetic_tinyimagenet_init_PCM_0':  dict( mean=(0.4254, 0.4149, 0.3795), std=(0.2846, 0.2773, 0.2910) ),
    'synthetic_cmi_cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'celeba': dict( mean=[0.5064, 0.4258, 0.3832], std=[0.3080, 0.2876, 0.2870]),
    'celeba_hair': dict( mean=[0.5064, 0.4258, 0.3832], std=[0.3080, 0.2876, 0.2870]),
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),     
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': classifiers.wresnet.wrn_16_1,
    'wrn16_2': classifiers.wresnet.wrn_16_2,
    'wrn40_1': classifiers.wresnet.wrn_40_1,
    'wrn40_2': classifiers.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': classifiers.resnet_tiny.resnet8,
    'resnet20': classifiers.resnet_tiny.resnet20,
    'resnet32': classifiers.resnet_tiny.resnet32,
    'resnet56': classifiers.resnet_tiny.resnet56,
    'resnet110': classifiers.resnet_tiny.resnet110,
    'resnet8x4': classifiers.resnet_tiny.resnet8x4,
    'resnet32x4': classifiers.resnet_tiny.resnet32x4,
    'vgg8': classifiers.vgg.vgg8_bn,
    'vgg11': classifiers.vgg.vgg11_bn,
    'vgg13': classifiers.vgg.vgg13_bn,
    'shufflenetv2': classifiers.shufflenetv2.shuffle_v2,
    'mobilenetv2': classifiers.mobilenetv2.mobilenet_v2,
    
    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  classifiers.resnet.resnet50,
    'resnet18':  classifiers.resnet.resnet18,
    'resnet34':  classifiers.resnet.resnet34,
    'resnet18_64x64': classifiers.resnet_64x64.resnet18,


    'resnet34_GroupNorm':  classifiers.resnet_group_norm.resnet34_group_norm,
    'resnet20_group_norm': classifiers.resnet_group_norm.resnet20_group_norm,
    'resnet20_evo_norm': classifiers.resnet_evo_norm.resnet20_evo_norm
}

IMAGENET_MODEL_DICT = {
    'resnet50_imagenet': classifiers.resnet_in.resnet50,
    'resnet18_imagenet': classifiers.resnet_in.resnet18,
    # 'mobilenetv2_imagenet': torchvision.models.mobilenet_v2,
}

SEGMENTATION_MODEL_DICT = {
    'deeplabv3_resnet50':  deeplab.deeplabv3_resnet50,
    'deeplabv3_mobilenet': deeplab.deeplabv3_mobilenet,
}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    if 'imagenet' in name:
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'deeplab' in name:
        model = SEGMENTATION_MODEL_DICT[name](num_classes=num_classes, pretrained_backbone=kwargs.get('pretrained_backbone', False))
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='mnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])      
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='fmnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])      
        data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.FashionMNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='celeba_hair':
        num_classes = 3
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ])       
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = CelebAHair(root=data_root, split="train", download=True, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='celeba_gender':
        num_classes = 2
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ])       
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = CelebAGender(root=data_root, split="train", download=True, transform=train_transform)
        val_dst = CelebAGender(data_root, split="test", download=True, transform=val_transform)

    elif name=='colored_mnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Lambda(lambda x: torch.cat([x, x, x], 0)),
            T.Normalize( **NORMALIZE_DICT['cifar10'] )
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
            T.Lambda(lambda x: torch.cat([x, x, x], 0))
        ])       
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='imagenette':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])       
        datapath = os.path.join('<set path>/Datasets', 'imagenette2')
        train_dst = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'train'),
            transform=train_transform)

        val_dst = torchvision.datasets.ImageFolder(
            root=os.path.join(datapath, 'val'),
            transform=val_transform)

    elif name=='synthetic_cifar10_init_random_noise':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['synthetic_tinyimagenet_init_PCM_0'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['synthetic_tinyimagenet_init_PCM_0'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_noise/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_noise/targets/', train_len=50000, transform=train_transform) 
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='synthetic_cifar10_init_tinyimagenet':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['synthetic_tinyimagenet_init_PCM_0'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['synthetic_tinyimagenet_init_PCM_0'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet/targets/', train_len=50000, transform=train_transform) 
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='synthetic_cifar10_init_tinyimagenet_constraint_0.1':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_0.1/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_0.1/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_random_noise_constraint_mean':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_mean/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_mean/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_constraint_mean':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_constraint_mean_l1_iter_250':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_l1_iter_250/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_l1_iter_250/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_random_noise_constraint_mean_l1_iter_250':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_mean_l1_iter_250/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_mean_l1_iter_250/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='synthetic_cifar10_init_tinyimagenet_constraint_1':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_1/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_1/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='synthetic_cifar10_init_tinyimagenet_constraint_10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_10/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_10/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_init_mnist_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/artificial_images/deepinv_mnist_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_resnet34_resnet18_10_iter/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_init_fmnist_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/artificial_images/deepinv_fmnist_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_resnet34_resnet18_10_iter/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_noise_2k':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_noise_2k/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_noise_2k/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_random_noise_constraint_10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_10/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_constraint_10/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='synthetic_cifar10_init_tinyimagenet_constraint_20':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_20/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_20/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_constraint_20_resnet34_resnet18':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_20_resnet34_resnet18/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_20_resnet34_resnet18/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_places365_constraint_mean_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_constraint_mean_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_constraint_mean_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_svhn_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform) 

    elif name=='deepinv_svhn_constraint_mean_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_constraint_mean_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_constraint_mean_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_svhn_resnet34_resnet18_20_pcm_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_20_pcm_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_20_pcm_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_places365_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_svhn_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar100_init_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_cifar100_init_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_cifar100_init_tinyimagenet_constraint_mean_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_places365_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_constraint_mean_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_svhn_constraint_mean_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_constraint_mean_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_constraint_mean_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_places365_constraint_mean_resnet34_resnet18_200_iter':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_constraint_mean_resnet34_resnet18_200_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_constraint_mean_resnet34_resnet18_200_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform) 

    elif name=='deepinv_svhn_resnet20_group_norm_resnet20_group_norm_10_iter_group_norm':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet20_group_norm_resnet20_group_norm_10_iter_group_norm/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet20_group_norm_resnet20_group_norm_10_iter_group_norm/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_500_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_500_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_500_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_random_noise_resnet34_resnet18_500_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_resnet34_resnet18_500_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_random_noise_resnet34_resnet18_500_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   


    elif name=='deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_15_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_15_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_15_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_20_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_20_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_20_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_40_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_40_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_40_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)   

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_60_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_60_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_60_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_100_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_100_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_100_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_svhn_resnet34_resnet18_500_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_500_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet34_resnet18_500_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_places365_resnet34_resnet18_500_iter_replicate_paper':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_500_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_places365_resnet34_resnet18_500_iter_replicate_paper/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_svhn_resnet20_evo_norm_resnet20_evo_norm_10_iter_evo_norm':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet20_evo_norm_resnet20_evo_norm_10_iter_evo_norm/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn_resnet20_evo_norm_resnet20_evo_norm_10_iter_evo_norm/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='synthetic_cifar10_init_random_noise':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        # data_root = os.path.join( data_root, 'torchdata' ) 
        

        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_noise/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_noise/targets/', train_len=50000, transform=train_transform)
        # val_dst = sT.transforms.Synthetic_Dataset('./artificial_img/cmi/img/', './artificial_img/cmi/targets/', train_len=50000)#, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)
    elif name=='synthetic_cifar10_init_svhn':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset('<set path>/Documents/CMI/artificial_img/deepinv_svhn/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_svhn/targets/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

########################################################################
    elif name=='deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper_noisy_5_epsilon':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/soft_labels_5_epsilon/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper_noisy_3_epsilon':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/soft_labels_3_epsilon/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

########################################################################

    elif name=='deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper_soft_labels':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_tinyimagenet_resnet34_resnet18_10_iter_replicate_paper/soft_labels/', train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

########################################################################

    elif name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)

    elif name=='cifar10_without_normalization':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)

    elif name=='c10+p365':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst_1 = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst_1 = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
        
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365_32x32' ) 
        train_dst_2 = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst_2 = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
        train_dst = torch.utils.data.ConcatDataset([train_dst_1, train_dst_2])
        val_dst = torch.utils.data.ConcatDataset([val_dst_1, val_dst_2])
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    elif name=='svhn_no_transformations':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root 
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    elif name=='imagenet' or name=='imagenet-0.5':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ILSVRC2012' ) 
        train_dst = datasets.ImageNet(data_root, split='train', transform=train_transform)
        val_dst = datasets.ImageNet(data_root, split='val', transform=val_transform)
    elif name=='imagenet_32x32':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ImageNet_32x32' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)

    elif name=='places365_32x32':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'Places365_32x32' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)

    elif name=='places365_32x32_cifar10_transformations':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='places365_32x32_without_normalization':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        val_transform = T.Compose([
            T.ToTensor()
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='places365_64x64':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365_64x64' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = None #datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='places365':
        num_classes=365
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='cub200':
        num_classes=200
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'CUB200')
        train_dst = datafree.datasets.CUB200(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.CUB200(data_root, split='val', transform=val_transform)
    elif name=='stanford_dogs':
        num_classes=120
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'StanfordDogs')
        train_dst = datafree.datasets.StanfordDogs(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.StanfordDogs(data_root, split='test', transform=val_transform)
    elif name=='stanford_cars':
        num_classes=196
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'StanfordCars')
        train_dst = datafree.datasets.StanfordCars(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.StanfordCars(data_root, split='test', transform=val_transform)
    elif name=='tiny_imagenet':
        num_classes=200
        train_transform = T.Compose([
            # T.RandomRotation(20),
            # T.RandomHorizontalFlip(),
            T.Resize(32),
            # T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]  #name
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join('<set path>/Documents/CMI/data/', 'tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)

    elif name=='tinyimagenet_cifar10_transformations':
        num_classes=10

        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        data_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)

    elif name=='fmnist_cifar10_transformations':
        num_classes=10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.FashionMNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='tinyimagenet_without_normalization':
        num_classes=200

        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        data_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)

    elif name=='svhn_cifar10_transformations':
        num_classes=10

        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)

    elif name=='mnist_cifar10_transformations':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)

    elif name=='svhn_without_normalization':
        num_classes=200

        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
###########################################################################################################################################
##########################################CIFAR100################################################
    elif name=='svhn_cifar100_transformations':
        num_classes=100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])

        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)

    elif name=='mnist_cifar100_transformations':
        num_classes=100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='places365_32x32_cifar100_transformations':
        num_classes=100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='tinyimagenet_cifar100_transformations':
        num_classes=100

        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])

        data_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)


    elif name=='deepinv_cifar100_places365_resnet34_resnet18_10_iter':
        num_classes = 100
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_places365_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_places365_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_cifar100_tinyimagenet_resnet34_resnet18_10_iter':
        num_classes = 100
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_tinyimagenet_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_tinyimagenet_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_cifar100_svhn_resnet34_resnet18_10_iter':
        num_classes = 100
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_svhn_resnet34_resnet18_10_iter/img/', 
        '<set path>/Documents/CMI/artificial_img/deepinv_init_cifar100_svhn_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_cifar100_places365_resnet34_resnet18_10_iter_updated':
        num_classes = 100
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar100_init_places365_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar100_init_places365_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_cifar100_tinyimagenet_resnet34_resnet18_10_iter_updated':
        num_classes = 100
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar100_init_tinyimagenet_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar100_init_tinyimagenet_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

    elif name=='deepinv_cifar100_svhn_resnet34_resnet18_10_iter_updated':
        num_classes = 100
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar100'] ),
        ])

        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar100_init_svhn_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar100_init_svhn_resnet34_resnet18_10_iter/targets/', 
        train_len=50000, transform=train_transform)
        val_dst = datasets.CIFAR100('./data/torchdata', train=False, download=True, transform=val_transform)  

###########################################################################################################################################
###########################################################################################################################################
##########MNIST#####
    elif name=='svhn_mnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ), #colored_mnist
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)

    elif name=='tinyimagenet_mnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        data_root = os.path.join('<set path>/Documents/CMI/data/', 'tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)

    elif name=='fmnist_mnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.FashionMNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='places365_32x32_mnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='deepinv_mnist_init_svhn_resnet18_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_init_svhn_resnet18_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_init_svhn_resnet18_resnet18_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_init_svhn_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_init_svhn_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_init_svhn_resnet34_resnet18_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_init_places365_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_init_places365_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_init_places365_resnet34_resnet18_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_init_tinyimagenet_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_init_tinyimagenet_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_init_tinyimagenet_resnet34_resnet18_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_svhn_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_svhn_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_tinyimagenet_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_tinyimagenet_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_tinyimagenet_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_places365_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_places365_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_places365_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_places365_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_fmnist_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_fmnist_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_fmnist_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_fmnist_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_fmnist_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_mnist_svhn_init_resnet18_resnet18_50k_1_eps_10_iter_no_adv':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['mnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter_no_adv/img/', 
        '<set path>/artificial_images/deepinv_mnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter_no_adv/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

###########################################################################################################################################
###########################################################################################################################################
#####################FMNIST##################################

    elif name=='svhn_fmnist_transformations':
        num_classes = 10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]) 
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)

    elif name=='tinyimagenet_fmnist_transformations':
        num_classes = 10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ), # 'fmnist'
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]) 
        # data_root = os.path.join('<set path>/Datasets/TinyImageNet/', 'tiny-imagenet-200')
        data_root = os.path.join('<set path>/Datasets/TinyImageNet')
        train_dst = datasets.ImageFolder(data_root+'/train', transform=train_transform)
        val_dst = datasets.ImageFolder(data_root+'/val', transform=val_transform)

    elif name=='places365_32x32_fmnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]) 
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='mnist_fmnist_transformations':
        num_classes=10
        train_transform = T.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_init_places365_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_init_places365_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_init_places365_resnet34_resnet18_10_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_init_svhn_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_init_svhn_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_init_svhn_resnet34_resnet18_10_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)


    elif name=='deepinv_fmnist_init_tinyimagenet_resnet34_resnet18_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_init_tinyimagenet_resnet34_resnet18_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_init_tinyimagenet_resnet34_resnet18_10_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_5_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_5_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_5_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_5_iter_tv_l2':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_5_iter_bn_oh_tv_3_l2/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_5_iter_bn_oh_tv_3_l2/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_10_iter_tv_l2':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter_bn_oh_tv_3_l2/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter_bn_oh_tv_3_l2/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_10_iter_tv_l2_bn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter_bn_oh_tv_3_l2_bn/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_10_iter_bn_oh_tv_3_l2_bn/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_20_iter_tv_l2':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_20_iter_bn_oh_tv_3_l2/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_20_iter_bn_oh_tv_3_l2/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)


    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_2_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_2_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_2_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_1_eps_1_iter_lr_0.00000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter_lr_0.00000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter_lr_0.00000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_10_eps_1_iter_lr_0.00000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_1_eps_1_iter_lr_0.00000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_1_iter_lr_0.00000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_1_epsilon_1_iter_lr_0.00000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_svhn_init_resnet18_resnet18_10_eps_1_iter_lr_0.00000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_svhn_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)


    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.01':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.01/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.01/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.0001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.0001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.0001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.00000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.00000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter_lr_0.000000000001':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.000000000001/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter_lr_0.000000000001/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_10_iter_check':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter_check/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter_check/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_63_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_63_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_63_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_tinyimagenet_init_resnet18_resnet18_10_eps_10_iter_updated_model':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter_updated_model/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter_updated_model/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            # T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_places365_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_mnist_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_mnist_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_mnist_init_resnet18_resnet18_10_eps_30_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_30_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_30_iter/targets/', 
        train_len=60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_mnist_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_fmnist_mnist_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['fmnist'] ),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_fmnist_DP_init_mnist_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = datasets.FashionMNIST(data_root, train=False, download=True, transform=val_transform)

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
#####################Celeba_hair##################################

    elif name=='places365_32x32_celeba_hair_transformations':
        num_classes = 3
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='lsun_celeba_hair_transformations':
        num_classes = 3
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'LSUN')
        train_dst = torchvision.datasets.LSUN(root=data_root, classes='train', transform=train_transform)
        val_dst = torchvision.datasets.LSUN(root=data_root, classes='val', transform=val_transform)

    elif name=='mnist_celeba_hair_transformations':
        num_classes = 3
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_mnist_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_mnist_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_mnist_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len = 60000, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_places365_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_places365_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_lsun_init_resnet18_resnet18_1_eps_10_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_1_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_1_epsilon_10_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_lsun_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_places365_init_resnet18_resnet18_1_eps_10_iter_tv_l2':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter_tv_l2/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_10_iter_tv_l2/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_places365_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_places365_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_lsun_init_resnet18_resnet18_10_eps_10_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_hair_lsun_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 3
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_hair_DP_init_lsun_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAHair(data_root, split="test", download=True, transform=val_transform)

###########################################################################################################################################
###########################################################################################################################################
###########################################celeba gender ##############################
    elif name=='places365_32x32_celeba_gender_transformations':
        num_classes = 2
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join( '<set path>/Datasets/', 'Places365') 
        train_dst = torchvision.datasets.Places365(root=data_root, split='train-standard', transform=train_transform, download=False)
        val_dst = torchvision.datasets.Places365(root=data_root, split= 'val', transform=val_transform, download=False)

    elif name=='lsun_celeba_gender_transformations':
        num_classes = 2
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ])
        data_root = os.path.join( '<set path>/Datasets/', 'LSUN')
        train_dst = torchvision.datasets.LSUN(root=data_root, classes='train', transform=train_transform)
        val_dst = torchvision.datasets.LSUN(root=data_root, classes='val', transform=val_transform)

    elif name=='deepinv_celeba_gender_lsun_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 2
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_lsun_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_lsun_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAGender(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_gender_lsun_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 2
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_lsun_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_lsun_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAGender(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_gender_places365_init_resnet18_resnet18_1_eps_1_iter':
        num_classes = 2
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_places365_resnet18_resnet18_1_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAGender(data_root, split="test", download=True, transform=val_transform)

    elif name=='deepinv_celeba_gender_places365_init_resnet18_resnet18_10_eps_1_iter':
        num_classes = 2
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] )
        ])
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['celeba'] ),
        ]) 
        data_root = os.path.join('<set path>/Datasets/', 'CelebA') 
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_celeba_gender_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 162720, transform=train_transform)
        val_dst = CelebAGender(data_root, split="test", download=True, transform=val_transform)

###########################################################################################################################################
###########################################################################################################################################
###########################################IMAGENETTE##############################
    elif name == 'textures_imagenette_transformations':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])
        data_root = os.path.join('<set path>/Datasets', 'Textures') 
        train_dst = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'images'), transform=train_transform)
        val_dst = torchvision.datasets.ImageFolder(root=os.path.join(data_root, 'images'), transform=val_transform)

    elif name=='lsun_imagenette_transformations':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])
        data_root = os.path.join( '<set path>/', 'LSUN')
        train_dst = torchvision.datasets.LSUN(root=data_root, classes='train', transform=train_transform)
        val_dst = torchvision.datasets.LSUN(root=data_root, classes='val', transform=val_transform)

    elif name=='deepinv_imagenette_lsun_init_resnet18_resnet18_104_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_imagenette_DP_init_lsun_resnet18_resnet18_104_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_imagenette_DP_init_lsun_resnet18_resnet18_104_epsilon_10_iter/targets/', 
        train_len = 10000, transform=train_transform)
        datapath = os.path.join('<set path>/Datasets', 'imagenette2')
        val_dst = torchvision.datasets.ImageFolder(root=os.path.join(datapath, 'val'), transform=val_transform)

    elif name=='deepinv_imagenette_textures_init_resnet18_resnet18_104_eps_10_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_imagenette_DP_init_textures_resnet18_resnet18_104_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_imagenette_DP_init_textures_resnet18_resnet18_104_epsilon_10_iter/targets/', 
        train_len = 5640, transform=train_transform)
        datapath = os.path.join('<set path>/Datasets', 'imagenette2')
        val_dst = torchvision.datasets.ImageFolder(root=os.path.join(datapath, 'val'), transform=val_transform)

    elif name=='deepinv_imagenette_textures_init_resnet18_resnet18_104_eps_1_iter':
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] )
        ])
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['imagenette'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_imagenette_DP_init_textures_resnet18_resnet18_104_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_imagenette_DP_init_textures_resnet18_resnet18_104_epsilon_1_iter/targets/', 
        train_len = 5640, transform=train_transform)
        datapath = os.path.join('<set path>/Datasets', 'imagenette2')
        val_dst = torchvision.datasets.ImageFolder(root=os.path.join(datapath, 'val'), transform=val_transform)

###########################################################################################################################################
###########################################################################################
#################################DP CIAFAR10 #####################################################################
    elif name=='deepinv_cifar10_tinyimagenet_init_resnet18_resnet18_10_eps_10_iter': 
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar10_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar10_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_10_iter/targets/', 
        train_len = 50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_tinyimagenet_init_resnet18_resnet18_10_eps_1_iter': 
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar10_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar10_DP_init_tinyimagenet_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_places365_init_resnet18_resnet18_10_eps_1_iter': 
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar10_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar10_DP_init_places365_resnet18_resnet18_10_epsilon_1_iter/targets/', 
        train_len = 50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_random_noise_init_resnet18_resnet18_eps_10_iter_0': 
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar10_random_noise_resnet18_resnet18_0_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar10_random_noise_resnet18_resnet18_0_iter/targets/', 
        train_len = 50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    elif name=='deepinv_cifar10_random_noise_init_resnet18_resnet18_eps_10_iter_50': 
        num_classes = 10
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT['cifar10'] ),
        ])
        train_dst = sT.transforms.Synthetic_Dataset(
        '<set path>/artificial_images/deepinv_cifar10_random_noise_resnet18_resnet18_50_iter/img/', 
        '<set path>/artificial_images/deepinv_cifar10_random_noise_resnet18_resnet18_50_iter/targets/', 
        train_len = 50000, transform=train_transform)
        val_dst = datasets.CIFAR10('./data/torchdata', train=False, download=True, transform=val_transform)

    # For semantic segmentation
    elif name=='nyuv2':
        num_classes=13
        train_transform = sT.Compose([
            sT.Multi( sT.Resize(256), sT.Resize(256, interpolation=Image.NEAREST)),
            #sT.Multi( sT.ColorJitter(0.2, 0.2, 0.2), None),
            sT.Sync(  sT.RandomCrop(128),  sT.RandomCrop(128)),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.uint8) ),
            sT.Multi( sT.Normalize( **NORMALIZE_DICT[name] ), None) #, sT.Lambda(lambd=lambda x: (x.squeeze()-1).to(torch.long)) )
        ])
        val_transform = sT.Compose([
            sT.Multi( sT.Resize(256), sT.Resize(256, interpolation=Image.NEAREST)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.uint8 ) ),
            sT.Multi( sT.Normalize( **NORMALIZE_DICT[name] ), None)#sT.Lambda(lambd=lambda x: (x.squeeze()-1).to(torch.long)) )
        ])
        data_root = os.path.join( data_root, 'NYUv2' )
        train_dst = datafree.datasets.NYUv2(data_root, split='train', transforms=train_transform)
        val_dst = datafree.datasets.NYUv2(data_root, split='test', transforms=val_transform)
    else:
        raise NotImplementedError

    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
