



import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import csv
import random
import matplotlib.pyplot as plt

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader,  \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from torch.cuda.amp import autocast, GradScaler
@autocast()
def mixt(x, lam, num_merge=2):
    batch_size = x.size(0)
    merge_size = torch.div(batch_size, num_merge, rounding_mode='trunc')
    result_x=x[:merge_size]
    for i in range(1, num_merge):
        result_x =lam*result_x+(1-lam)*x[merge_size*i:merge_size*(i+1)]
    return result_x

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()  
    net.eval()  
    test_loss = 0.0  
    correct = 0.0  

    for (images, labels) in cifar10_test_loader:  

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        if args.fp16:
            with autocast():
                outputs = net(images)
                loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()
        else:
            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    finish = time.time()  

    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,  
        test_loss / len(cifar10_test_loader.dataset),  
        correct.float() / len(cifar10_test_loader.dataset),  
        finish - start  
    ))
    test_loss = test_loss / len(cifar10_test_loader.dataset),  
    test_acc = correct.float() / len(cifar10_test_loader.dataset),  
    print()

    return {'test loss': test_loss[0], 'test acc': test_acc[0].item()}

    

def train(epoch,filename):

    start = time.time()
    net.train()
    scaler = GradScaler()  
    for batch_index, (images, labels) in enumerate(cifar10_train_loader):


        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        if args.fp16:
            with autocast():
                if args.merge > 1:
                    lam = np.random.beta(1, 1)
                    lam = torch.tensor(lam).cuda()
                    labels = nn.functional.one_hot(labels, 100)
                    labels = mixt(labels, lam, args.merge)
                    outputs = net(images,lam,args.merge)
                    loss = loss_function(outputs, labels)
                else:
                    outputs = net(images)
                    loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.merge > 1:
                lam = np.random.beta(1, 1)
                lam = torch.tensor(lam)
                
                labels = nn.functional.one_hot(labels, 100)
                labels = mixt(labels, lam, args.merge)
                outputs = net(images, lam, args.merge)
                loss = loss_function(outputs, labels)

            else:
                outputs = net(images)
                loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()


        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar10_train_loader.dataset)
        ))
        train_loss=loss.item()

        
        

    finish = time.time()

    

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))  
    epoch_time = finish - start  
    lr = optimizer.param_groups[0]['lr']  

    return {'epoch': epoch, 'train loss': train_loss, 'lr': lr, 'time': epoch_time}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-fp16', action='store_true', default=False, help='use gpu or not')  
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-merge', type=int, default=1, help='degree of merging samples in mixt')
    
    args = parser.parse_args()
    net = get_network(args)
    
    cifar10_train_loader = get_training_dataloader(
        root='./data',
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        batch_size=args.b,
        num_workers=6,
        shuffle=True
    )

    cifar10_test_loader = get_test_dataloader(
        root='./data',
        mean=(0.4914, 0.4822, 0.4465),  
        std=(0.2023, 0.1994, 0.2010),
        batch_size=args.b,
        num_workers=6,
        shuffle=False
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.gpu:
        filename = f"cifar100{args.net}_lr{args.lr}_{args.merge}to1_{timestamp}_gpu4090.csv"
    else:
        filename = f"cifar100{args.net}_lr{args.lr}_{args.merge}to1_{timestamp}_cpu.csv"
    if args.fp16:
        filename = f"cifar100{args.net}_lr{args.lr}_{args.merge}to1_{timestamp}fp16_gpu4090.csv"

    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    iter_per_epoch = len(cifar10_train_loader)
    

    
    
    
    
    
    

    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
                       train_scheduler.step()


        text1=train(epoch,filename)
        text2=eval_training(epoch)
        
        
        text1.update(text2)
        
        fieldnames = ['epoch', 'train loss', 'lr', 'test loss', 'test acc', 'time']  
        reordered_dict = {key: text1[key] for key in fieldnames}
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writerow(text1)
            
            print(text1)



