import os
import sys
import time
import math
import pickle as cPickle


#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
frozen_models_dir = os.path.join(root_dir, "frozen_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, frozen_models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

#%%
# Standard or Built-in packages
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision
import torchvision.transforms as transforms

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
# import frozen_models
from utils_utils import accuracy, AverageMeter, save_checkpoint 
#from utils_bar import progress_bar

class SplitActivations_Dataset(Dataset):
    def __init__(self, args, datapath, tgtpath, train_len = True, transform=None):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        self.args = args
        self.transform = transform
        
    def __getitem__(self, index):

        x = torch.load(self.datapath + 'image_' + str(index) + '.pth')  
        
        y = torch.load(self.tgtpath + 'label_'+ str(index) + '.pth')



        if self.transform:
            x = self.transform(x)

        return {'data': x, 'target': y}
    
    def __len__(self):
        if self.train_len == True:
            return 50000 
        else :
            return 10000


def train(train_loader, model, criterion, optimizer, epoch, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    batch_freq = int(len(train_loader)/args.print_freq)
    
    for batch_idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input_var = batch['data']
        target_var = batch['target'].long()

        # if args.half:
        #     input_var = input_var.half()

        input_var, target_var = input_var.to(device), target_var.to(device)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, batch['target'].to(device))[0]
        losses.update(loss.item(), batch['data'].size(0))
        top1.update(prec1.item(), batch['data'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if (batch_idx+1) % batch_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'LR: {3}\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
        #               epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'], loss=losses, top1=top1, ))
    
    print('Total train loss: {loss.avg:.4f}\n'.format(loss=losses))


# Evaluate on a model
def test(test_loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (input_var, target_var) in enumerate(test_loader):
            # input_var = batch['data']
            # target_var = batch['target'].long()
    
            # if args.half:
            #     input_var = input_var.half()
    
            input_var, target_var = input_var.to(device), target_var.to(device)
            # input_var, target_var = inputs.to(device), targets.to(device)
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            
            # losses.update(loss.data, inputs.size(0))
            # top1.update(prec1[0], inputs.size(0))
            # top5.update(prec5[0], inputs.size(0))

            losses.update(loss.data, input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))
    
            # if batch_idx % 10 == 0:
            #         print('[{0}/{1}({2:.0f}%)]\t'
            #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #             batch_idx, len(test_loader), 100. *float(batch_idx)/len(test_loader),
            #             loss=losses, top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, top5=top5, loss=losses))
    acc = top1.avg
    return acc, losses.avg



parser = argparse.ArgumentParser(description= ' Training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
            help='dataset name')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20_cifar10',
            help='name of the model')

parser.add_argument('--mode-train', default='rram',
                help='folder to load train activations from')
parser.add_argument('--mode-test', default='rram',
                help='folder to load test activations from')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
            metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
            help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
            help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', default=0.2, type=float,
            help='learning rate decay')

parser.add_argument('--milestones', default=[10,20,30,40,70,90],  # [10,20,30,40]
            help='Milestones for LR decay')

parser.add_argument('--loss', type=str, default='crossentropy', 
            help='Loss function to use')
parser.add_argument('--optim', type=str, default='sgd',
            help='Optimizer to use')

parser.add_argument('--print-freq', '-p', default=5, type=int,
                metavar='N', help='print frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False, 
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true', default=True,
                    help='use half-precision(16-bit) ')

parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--gpus', default='0', help='gpus (default: 0)')

args = parser.parse_args()


transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

train_data = SplitActivations_Dataset(args, '/local/scratch/a/<set_path>/Documents/CMI/artificial_img/cmi/img/', 
    '/local/scratch/a/<set_path>/Documents/CMI/artificial_img/cmi/targets/', train_len = True, transform = transform_train)


train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


if args.loss == 'nll':
    criterion = nn.NLLLoss()
elif args.loss == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
else:
    raise NotImplementedError

model = ptcv_get_model(args.model, pretrained=False)


optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=True)


lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=args.milestones, 
                                            gamma=args.gamma, 
                                            last_epoch=args.start_epoch - 1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

model = model.to(device)
best_acc = 0
end = time.time()
for epoch in range(args.start_epoch, args.epochs):
    print('Epoch : ', epoch)

    train(train_loader, model, criterion, optimizer, epoch, device)
    print('Train time: {}'.format(time.time()-end))
    end = time.time()

    # evaluate on validation set
    acc, loss = test(test_loader, model, criterion, device)
    
    lr_scheduler.step()

    # remember best prec@1 and save checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
        

    print('Best acc: {:.3f}'.format(best_acc))
    print('-'*80)

    print('Test time: {}\n'.format(time.time()-end))
    end = time.time()







