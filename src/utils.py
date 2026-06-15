#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_noniid_dirichlet
from sampling import dirichlet_noniid


def _get_noniid_mode(args, dataset_name):
    mode = getattr(args, 'noniid_mode', '')
    if mode:
        return mode
    if dataset_name == 'cifar':
        return getattr(args, 'cifar_noniid_mode', 'shards')
    return 'shards'


def _get_shards_per_user(args, dataset_name):
    shards_per_user = getattr(args, 'shards_per_user', 0)
    if shards_per_user > 0:
        return shards_per_user
    if dataset_name == 'cifar':
        return getattr(args, 'cifar_shards_per_user', 2)
    return 2


def _get_dirichlet_alpha(args, dataset_name):
    alpha = getattr(args, 'dirichlet_alpha', 0.0)
    if alpha > 0:
        return alpha
    if dataset_name == 'cifar':
        return getattr(args, 'cifar_dirichlet_alpha', 0.5)
    return 0.5


def _get_dirichlet_min_size(args, dataset_name):
    min_size = getattr(args, 'dirichlet_min_size', 10)
    if dataset_name == 'cifar':
        return getattr(args, 'cifar_dirichlet_min_size', min_size)
    return min_size


def _build_noniid_groups(dataset, args, dataset_name):
    mode = _get_noniid_mode(args, dataset_name)
    if mode == 'dirichlet':
        if dataset_name == 'cifar' and not getattr(args, 'noniid_mode', ''):
            return cifar_noniid_dirichlet(
                dataset,
                args.num_users,
                alpha=_get_dirichlet_alpha(args, dataset_name),
                min_size=_get_dirichlet_min_size(args, dataset_name),
            )
        return dirichlet_noniid(
            dataset,
            args.num_users,
            alpha=_get_dirichlet_alpha(args, dataset_name),
            min_size=_get_dirichlet_min_size(args, dataset_name),
        )

    if dataset_name == 'cifar':
        return cifar_noniid(
            dataset,
            args.num_users,
            shards_per_user=_get_shards_per_user(args, dataset_name),
        )
    return mnist_noniid(
        dataset,
        args.num_users,
        shards_per_user=_get_shards_per_user(args, dataset_name),
    )


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]

        cifar_crop_size = getattr(args, 'cifar_crop_size', 24)
        cifar_normalize = bool(getattr(args, 'cifar_normalize', 0))
        train_crop = (
            transforms.RandomCrop(32, padding=4)
            if cifar_crop_size == 32
            else transforms.RandomCrop(cifar_crop_size)
        )

        train_transforms = [
            train_crop,
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
        test_transforms = [
            transforms.CenterCrop(cifar_crop_size),
            transforms.ToTensor(),
        ]
        if cifar_normalize:
            train_transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
            test_transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

        train_transform = transforms.Compose(train_transforms)
        test_transform = transforms.Compose(test_transforms)
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
                user_groups = _build_noniid_groups(train_dataset, args, 'cifar')

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
                user_groups = _build_noniid_groups(train_dataset, args, args.dataset)

    return train_dataset, test_dataset, user_groups

def get_raw_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        cifar_crop_size = getattr(args, 'cifar_crop_size', 24)

        train_transform = transforms.Compose([
            transforms.CenterCrop(cifar_crop_size),
        
            transforms.ToTensor(),
            
        ])
        test_transform = transforms.Compose([
            transforms.CenterCrop(cifar_crop_size),
            
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
                user_groups = _build_noniid_groups(train_dataset, args, 'cifar')

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
                user_groups = _build_noniid_groups(train_dataset, args, args.dataset)

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


def segmented_average_weights(local_weights, idxs_users, prev_global_weights,
                              num_steal=5, num_img_per_client=1,
                              attack_num_pixel=576, mode='segmented',
                              seg_alpha=0.5, attack_position_mode='spread'):

    if mode == 'avg':
        return average_weights(local_weights)

    if not local_weights:
        return copy.deepcopy(prev_global_weights)

    segment_size = attack_num_pixel * num_img_per_client
    target_len = num_steal * segment_size

    if target_len <= 0:
        return average_weights(local_weights)

    layer_keys = list(prev_global_weights.keys())
    layer_shapes = {k: prev_global_weights[k].shape for k in layer_keys}

    def flatten_weights(w_dict):
        return torch.cat([w_dict[k].flatten() for k in layer_keys])

    local_flats = [flatten_weights(lw) for lw in local_weights]
    mean_flat = torch.mean(torch.stack(local_flats), dim=0)
    global_flat = mean_flat.clone()
    total_params = global_flat.numel()

    if target_len > total_params:
        raise ValueError(
            f"Attack target length ({target_len}) exceeds model parameter count ({total_params})."
        )

    if attack_position_mode == 'front':
        attack_indices = torch.arange(target_len, device=global_flat.device, dtype=torch.long)
    elif attack_position_mode == 'spread':
        attack_indices = torch.linspace(
            0, total_params - 1, target_len, device=global_flat.device
        ).long()
    else:
        raise ValueError(f'Unsupported attack_position_mode: {attack_position_mode}')

    if mode == 'target_only_avg':
        target_participants = [
            local_flats[i] for i, client_id in enumerate(idxs_users) if client_id < num_steal
        ]
        if target_participants:
            target_stack = torch.stack([lf[attack_indices] for lf in target_participants])
            global_flat[attack_indices] = torch.mean(target_stack, dim=0)
        return _rebuild_state_dict(global_flat, prev_global_weights, layer_keys, layer_shapes)

    for i, client_id in enumerate(idxs_users):
        seg_start = client_id * segment_size
        seg_end = min((client_id + 1) * segment_size, target_len)
        if seg_start >= target_len:
            continue

        client_indices = attack_indices[seg_start:seg_end]
        client_values = local_flats[i][client_indices]

        if mode == 'segmented':
            global_flat[client_indices] = client_values
        elif mode == 'segmented_soft':
            avg_values = mean_flat[client_indices]
            global_flat[client_indices] = seg_alpha * client_values + (1.0 - seg_alpha) * avg_values
        else:
            raise ValueError(f'Unsupported aggregation mode: {mode}')

    return _rebuild_state_dict(global_flat, prev_global_weights, layer_keys, layer_shapes)


def _rebuild_state_dict(global_flat, prev_global_weights, layer_keys, layer_shapes):
    new_global_weights = {}
    current_ptr = 0
    for k in layer_keys:
        numel = prev_global_weights[k].numel()
        new_global_weights[k] = global_flat[current_ptr: current_ptr + numel].reshape(layer_shapes[k])
        current_ptr += numel
    return new_global_weights

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Momentum  : {args.momentum}')
    print(f'    Weight decay     : {args.weight_decay}')
    print(f'    LR scheduler     : {args.lr_scheduler}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        noniid_mode = _get_noniid_mode(args, args.dataset)
        print(f'    Non-IID mode       : {noniid_mode}')
        if noniid_mode == 'dirichlet':
            print(f'    Dirichlet alpha    : {_get_dirichlet_alpha(args, args.dataset)}')
        else:
            print(f'    Shards per user    : {_get_shards_per_user(args, args.dataset)}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Aggregation mode   : {args.agg_mode}')
    if args.agg_mode == 'segmented_soft':
        print(f'    Segment alpha      : {args.seg_alpha}')
    print(f'    Attack position    : {args.attack_position_mode}')
    print('')
    return
