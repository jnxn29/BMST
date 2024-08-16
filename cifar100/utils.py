
import os
import sys
import re
import datetime


import numpy as np
from PIL import Image
import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):

    print(args.net)

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'c2mixt_resnet18':
        from models.c2mixt_resnet import c2mixt_resnet18
        net = c2mixt_resnet18()
    elif args.net == 'c2mixt_resnet34':
        from models.c2mixt_resnet import c2mixt_resnet34
        net = c2mixt_resnet34()
    elif args.net == 'c2mixt_resnet50':
        from models.c2mixt_resnet import c2mixt_resnet50
        net = c2mixt_resnet50()
    elif args.net == 'c2mixt_resnet101':
        from models.c2mixt_resnet import c2mixt_resnet101
        net = c2mixt_resnet101()
    elif args.net == 'c2mixt_resnet152':
        from models.c2mixt_resnet import c2mixt_resnet152
        net = c2mixt_resnet152()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()

    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()

    
    elif args.net == 'c4mixt_resnet18':
        from models.c4mixt_resnet import c4mixt_resnet18
        net = c4mixt_resnet18()
    elif args.net == 'c4mixt4_resnet34':
        from models.c4mixt_resnet import c4mixt_resnet34
        net = c4mixt_resnet34()
    elif args.net == 'c4mixt4_resnet50':
        from models.c4mixt_resnet import c4mixt_resnet50
        net = c4mixt_resnet50()
    elif args.net == 'c4mixt4_resnet101':
        from models.c4mixt_resnet import c4mixt_resnet101
        net = c4mixt_resnet101()
    elif args.net == 'c4mixt_resnet152':
        from models.c2mixt_resnet import c4mixt_resnet152
        net = c4mixt_resnet152()


        
    elif args.net == 'c2c4mixt_resnet18':
        from models.c2c4mixt_resnet import c2c4mixt_resnet18
        net = c2c4mixt_resnet18()
    elif args.net == 'c2c4mixt_resnet34':
        from models.c2c4mixt_resnet import c2c4mixt_resnet34
        net = c2c4mixt_resnet34()
    elif args.net == 'c2c4mixt_resnet50':
        from models.c2c4mixt_resnet import c2c4mixt_resnet50
        net = c2c4mixt_resnet50()
    elif args.net == 'c2c4mixt_resnet101':
        from models.c2c4mixt_resnet import c2c4mixt_resnet101
        net = c2c4mixt_resnet101()
    elif args.net == 'c2c4mixt_resnet152':
        from models.c2c4mixt_resnet import c2c4mixt_resnet152
        net = c2c4mixt_resnet152()
    elif args.net == 'c14mixt_inceptionv3':
        from models.c14mixt_inceptionv3 import c14mixt_inceptionv3
        net = c14mixt_inceptionv3()
    elif args.net == 'c6mixt_inceptionv3':
        from models.c6mixt_inceptionv3 import c6mixt_inceptionv3
        net = c6mixt_inceptionv3()
    elif args.net == 'c3mixt_inceptionv3':
        from models.c3mixt_inceptionv3 import c3mixt_inceptionv3
        net = c3mixt_inceptionv3()
    
    elif args.net == 'c3mixt_resnet18':
        from models.c3mixt_resnet import c3mixt_resnet18
        net = c3mixt_resnet18()
    elif args.net == 'c3mixt_resnet34':
        from models.c3mixt_resnet import c3mixt_resnet34
        net = c3mixt_resnet34()
    elif args.net == 'c3mixt_resnet50':
        from models.c3mixt_resnet import c3mixt_resnet50
        net = c3mixt_resnet50()
    elif args.net == 'c3mixt_resnet101':
        from models.c3mixt_resnet import c3mixt_resnet101
        net = c3mixt_resnet101()
    elif args.net == 'c3mixt_resnet152':
        from models.c3mixt_resnet import c3mixt_resnet152
        net = c3mixt_resnet152()



    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: 
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):


    transform_train = transforms.Compose([
        
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):


    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std



















def most_recent_folder(net_weights, fmt):

    
    folders = os.listdir(net_weights)

    
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):

    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):

    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]