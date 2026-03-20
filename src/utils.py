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

    elif args.dataset in ['mnist', 'fmnist']:
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
            dataset_class = datasets.MNIST
            # MNIST 官方标准化参数
            norm_mean, norm_std = (0.1307,), (0.3081,)
        else:
            data_dir = './data/fashion_mnist/'
            dataset_class = datasets.FashionMNIST
            # FashionMNIST 官方标准化参数 (可选，也可以统一用上面的)
            norm_mean, norm_std = (0.2860,), (0.3530,)

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        # 动态调用对应的类 (datasets.MNIST 或 datasets.FashionMNIST)
        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=apply_transform)

        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=apply_transform)

        # 抽样逻辑 (通常 MNIST 和 FMNIST 共用一套抽样函数)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
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

    elif args.dataset in ['mnist', 'fmnist']:
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
            dataset_class = datasets.MNIST
        else:
            data_dir = './data/fashion_mnist/'
            dataset_class = datasets.FashionMNIST

        # 仅转换为 Tensor，不进行标准化
        apply_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = dataset_class(data_dir, train=True, download=True,
                                     transform=apply_transform)

        test_dataset = dataset_class(data_dir, train=False, download=True,
                                    transform=apply_transform)

        # 抽样逻辑
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
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

import torch
import copy

def segmented_average_weights(local_weights, idxs_users, prev_global_weights, num_steal=5, num_img_per_client=1, attack_num_pixel=576):

    M = len(local_weights)
    SEGMENT_SIZE = attack_num_pixel * num_img_per_client
    K = num_steal  # 窃取目标客户端数量（可配置）
    TARGET_LEN = K * SEGMENT_SIZE

    # 获取模型结构信息并展平所有参数
    layer_keys = prev_global_weights.keys()
    layer_shapes = {k: prev_global_weights[k].shape for k in layer_keys}
    
    def flatten_weights(w_dict):
        return torch.cat([w_dict[k].flatten() for k in layer_keys])

    # 展平上一轮全局权重作为基准
    global_flat = flatten_weights(prev_global_weights)
    total_params = global_flat.numel()
    
    # 展平所有参与本轮的本地权重
    local_flats = [flatten_weights(lw) for lw in local_weights]
    
    # 生成等间距散布索引（与 cor_attack 中的 _get_spread_indices 一致）
    spread_indices = torch.linspace(0, total_params - 1, TARGET_LEN).long()

    # 处理散布的攻击参数
    # 每个客户端负责其中 attack_num_pixel * num_img_per_client 个位置
    for i in range(M):
        client_id = idxs_users[i]
        seg_start = client_id * SEGMENT_SIZE
        seg_end = (client_id + 1) * SEGMENT_SIZE
        
        if seg_start < TARGET_LEN:
            # 该客户端负责的散布索引子集
            client_indices = spread_indices[seg_start:seg_end]
            # 用该客户端的本地参数在对应散布位置覆盖全局参数
            global_flat[client_indices] = local_flats[i][client_indices]

    # 处理非攻击参数：所有参数位置取平均（排除散布索引位置）
    attack_mask = torch.zeros(total_params, dtype=torch.bool)
    attack_mask[spread_indices] = True
    non_attack_mask = ~attack_mask
    
    # 非攻击位置取所有客户端的平均值
    if non_attack_mask.any():
        non_attack_indices = torch.where(non_attack_mask)[0]
        extra_params_stack = torch.stack([lf[non_attack_indices] for lf in local_flats])
        global_flat[non_attack_indices] = torch.mean(extra_params_stack, dim=0)

    # 将长向量重新装填回 state_dict 结构
    new_global_weights = {}
    current_ptr = 0
    for k in layer_keys:
        numel = prev_global_weights[k].numel()
        new_global_weights[k] = global_flat[current_ptr : current_ptr + numel].reshape(layer_shapes[k])
        current_ptr += numel

    return new_global_weights

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
