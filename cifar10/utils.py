
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
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
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
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
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
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()



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

    elif args.net == 'bmst_c3_resnet18':
        from models.bmst_c3_resnet import bmst_c3_resnet18
        net = bmst_c3_resnet18()
    elif args.net == 'bmst_c3_resnet34':
        from models.bmst_c3_resnet import bmst_c3_resnet34
        net = bmst_c3_resnet34()
    elif args.net == 'bmst_c3_resnet50':
        from models.bmst_c3_resnet import bmst_c3_resnet50
        net = bmst_c3_resnet50()
    elif args.net == 'bmst_c3_resnet101':
        from models.bmst_c3_resnet import bmst_c3_resnet101
        net = bmst_c3_resnet101()
    elif args.net == 'bmst_c3_resnet152':
        from models.bmst_c3_resnet import bmst_c3_resnet152
        net = bmst_c3_resnet152()


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net


def get_training_dataloader(root, mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop([32,32], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    cifar10_training = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return cifar10_training_loader

def get_test_dataloader(root, mean, std, batch_size=16, num_workers=2, shuffle=False):

    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    cifar10_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return cifar10_test_loader

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