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
            # 图像大小调整：随机裁剪到 24x24 (cropping the images to 24x24)
            # 由于原始图像是 32x32，这里使用随机裁剪来模拟从 32x32 中提取 24x24 块
            transforms.RandomCrop(24), 
            
            # 随机左右翻转 (randomly flipping left-right)
            transforms.RandomHorizontalFlip(),
            
            # 调整对比度和亮度 (adjusting the contrast, brightness)
            # 通常使用 ColorJitter 实现，这里同时调整对比度、亮度和饱和度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            # 转换为 Tensor
            transforms.ToTensor(),
            
            # 白化/标准化 (whitening)
            # 减去均值，除以标准差
            # transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        test_transform = transforms.Compose([
            # 图像大小调整：中心裁剪到 24x24 (cropping the images to 24x24)
            transforms.CenterCrop(24),
            
            # 转换为 Tensor
            transforms.ToTensor(),
            
            # 白化/标准化 (whitening)
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
            # 图像大小调整：随机裁剪到 24x24 (cropping the images to 24x24)
            # 由于原始图像是 32x32，这里使用随机裁剪来模拟从 32x32 中提取 24x24 块
            transforms.CenterCrop(24), 
        
            transforms.ToTensor(),
            
        ])
        test_transform = transforms.Compose([
            # 图像大小调整：中心裁剪到 24x24 (cropping the images to 24x24)
            transforms.CenterCrop(24),
            
            # 转换为 Tensor
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

import torch
import copy

def segmented_average_weights(local_weights, idxs_users, prev_global_weights):

    M = len(local_weights)
    
    # 初始化全局权重 (w_avg) 为上一轮的全局权重作为基准
    w_avg = copy.deepcopy(prev_global_weights) 
    
    # 图片像素的展平尺寸，例如 24*24 = 576
    SEGMENT_SIZE = 576 
    
    # 客户端总数 K=100
    # 这是一个关键的假设，必须与数据划分时的 num_users 一致。
    K = 100 
    total_target_params = K * SEGMENT_SIZE # 57600
    
    # 用于计算 FedAvg 的临时字典
    temp_avg = copy.deepcopy(local_weights[0])
    
    # 用于追踪当前全局参数索引，以映射到正确的客户端分段
    current_param_count = 0 

    for key in w_avg.keys():
        
        if w_avg[key].dim() < 1:
            # 计算该层的平均值
            for i in range(1, M):
                temp_avg[key] += local_weights[i][key]
            temp_avg[key] = torch.div(temp_avg[key], M)
            
            # 用平均值覆盖 w_avg 中的非目标参数
            w_avg[key] = temp_avg[key]
            
        # 对目标层（维度 >= 1）执行分段覆盖
        elif w_avg[key].dim() >= 1 and current_param_count < total_target_params:
            
            param = w_avg[key]
            w_avg_flat = param.flatten()
            original_shape = param.shape
            
            # 当前层的所有参数数量
            numel_current_layer = w_avg_flat.numel()
            
            # 遍历参与本轮训练的 M 个客户端
            for i in range(M):
                client_index = idxs_users[i] # 客户端的全局 ID
                local_weights_i = local_weights[i]
                
                # 计算该客户端的分段在全局拼接向量中的起始和结束索引
                global_start_idx = client_index * SEGMENT_SIZE
                global_end_idx = (client_index + 1) * SEGMENT_SIZE
                
                # 检查该客户端的分段是否落入当前模型层所覆盖的范围
                # 范围定义：[current_param_count, current_param_count + numel_current_layer)
                
                # 客户端分段的起点在当前层之后，跳过
                if global_start_idx >= current_param_count + numel_current_layer:
                    continue
                
                # 客户端分段的终点在当前层起点之前，跳过 (这种情况不应该发生，因为按顺序遍历)
                if global_end_idx <= current_param_count:
                    continue
                
                # === 核心逻辑：计算本地偏移量并执行覆盖 ===
                
                # 确定该分段在当前层的本地起始和结束索引
                # local_start_offset 是该分段在 w_avg_flat 中的起始位置
                local_start_offset = global_start_idx - current_param_count
                local_end_offset = local_start_offset + SEGMENT_SIZE
                
                # 边界检查：确保分段没有超出当前层的参数数量
                if local_end_offset <= numel_current_layer:
                    
                    # 获取该客户端本地更新的该层参数的展平版本
                    local_update_flat = local_weights_i[key].flatten()
                    
                    # 覆盖全局基准（prev_global_weights）中对应的分段
                    w_avg_flat[local_start_offset:local_end_offset] = local_update_flat[local_start_offset:local_end_offset]
                
            # 将更新后的展平张量重新塑形并放回 w_avg
            w_avg[key] = w_avg_flat.reshape(original_shape)
            
            # 更新全局参数计数器，指向下一个层的起点
            current_param_count += numel_current_layer
            
    # 返回聚合后的全局权重
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
