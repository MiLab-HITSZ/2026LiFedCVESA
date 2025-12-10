#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]

        train_transform = transforms.Compose([
            # 1. 图像大小调整：随机裁剪到 24x24 (cropping the images to 24x24)
            # 由于原始图像是 32x32，这里使用随机裁剪来模拟从 32x32 中提取 24x24 块
            transforms.RandomCrop(24), 
            
            # 2. 随机左右翻转 (randomly flipping left-right)
            transforms.RandomHorizontalFlip(),
            
            # 3. 调整对比度和亮度 (adjusting the contrast, brightness)
            # 通常使用 ColorJitter 实现，这里同时调整对比度、亮度和饱和度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            # 4. 转换为 Tensor
            transforms.ToTensor(),
            
            # 5. 白化/标准化 (whitening)
            # 减去均值，除以标准差
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        test_transform = transforms.Compose([
            # 1. 图像大小调整：中心裁剪到 24x24 (cropping the images to 24x24)
            transforms.CenterCrop(24),
            
            # 2. 转换为 Tensor
            transforms.ToTensor(),
            
            # 3. 白化/标准化 (whitening)
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def get_raw_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'

        train_transform = transforms.Compose([
            # 1. 图像大小调整：随机裁剪到 24x24 (cropping the images to 24x24)
            # 由于原始图像是 32x32，这里使用随机裁剪来模拟从 32x32 中提取 24x24 块
            transforms.RandomCrop(24), 
        
            transforms.ToTensor(),
            
        ])
        test_transform = transforms.Compose([
            # 1. 图像大小调整：中心裁剪到 24x24 (cropping the images to 24x24)
            transforms.CenterCrop(24),
            
            # 2. 转换为 Tensor
            transforms.ToTensor(),
            
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor()])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def segmented_average_weights(local_weights, idxs_users, prev_global_weights):

    M = len(local_weights)
    
    # 初始化全局权重 (w_avg) 为上一轮的全局权重
    # 这是未参与客户端的目标层参数的起始值
    w_avg = copy.deepcopy(prev_global_weights) 
    
    # 识别攻击目标层（假设为第一个权重张量）
    try:
        target_key = list(w_avg.keys())[0] 
    except IndexError:
        print("Error: Model state dict is empty. Returning averaged weights.")
        return w_avg
        
    SEGMENT_SIZE = 576 
    
    # 对非目标层（未参与窃取的参数）计算平均值并覆盖 w_avg
    # 我们只对非目标层进行平均
    
    # 创建一个临时的平均权重字典，用于计算平均值
    temp_avg = copy.deepcopy(local_weights[0])
    for key in temp_avg.keys():
        for i in range(1, M):
            temp_avg[key] += local_weights[i][key]
        temp_avg[key] = torch.div(temp_avg[key], M)
        
        # 如果不是目标层，则用平均值覆盖 w_avg 中的参数
        if key != target_key:
            w_avg[key] = temp_avg[key] # FedAvg for non-target layers

    # 对目标层（target_key）执行分段覆盖
    # 目标层 w_avg[target_key] 此时是上一轮的全局权重 (保持不变的基准)
    target_tensor_avg_flat = w_avg[target_key].flatten()
    original_shape = w_avg[target_key].shape
    
    # 遍历参与本轮训练的 M 个客户端，用其本地更新覆盖对应的分段
    for i in range(M):
        client_index = idxs_users[i] # 客户端的原始 ID (0 到 K-1)
        local_weights_i = local_weights[i]
        
        # 获取该客户端本地更新的目标张量的展平版本
        local_update_flat = local_weights_i[target_key].flatten()
        
        # 计算该客户端负责的 576 长度分段的起始和结束索引
        start_idx = client_index * SEGMENT_SIZE
        end_idx = (client_index + 1) * SEGMENT_SIZE
        
        # 边界检查
        if end_idx <= target_tensor_avg_flat.numel():
            # 使用该客户端的本地更新来覆盖全局基准中对应的分段
            # 实现了参与客户端的分段更新
            target_tensor_avg_flat[start_idx:end_idx] = local_update_flat[start_idx:end_idx]
        # 否则：未参与的客户端分段参数保持 prev_global_weights 的值 (w_avg 的初始值)
            
    # 将处理后的目标层张量重新塑形并放回 w_avg
    w_avg[target_key] = target_tensor_avg_flat.reshape(original_shape)
    
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
