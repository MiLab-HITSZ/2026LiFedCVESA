#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='optimizer weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='exp',
                        choices=['exp', 'cosine'],
                        help="Learning-rate schedule. 'exp' keeps existing lr_decay behavior.")
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='minimum learning rate for cosine schedule')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--lr_decay', type=float, default=0.99, 
                        help="learning rate decay factor per communication round")
    parser.add_argument('--gama', type=float, default=0.1, 
                        help="The coefficient (gamma) for the CVEA attack regularization term. " \
                        "Set to 0.0 to disable attack.")
    parser.add_argument('--gama_warmup_epochs', type=int, default=100,
                        help="Number of epochs for gama to warm up from 0 to target value. "
                        "Phase1: first half epochs linear warmup; Phase2: second half cosine ramp-up. "
                        "Set to 0 to disable warmup (use constant gama).")
    parser.add_argument('--num_steal', type=int, default=5,
                        help="Number of target clients to steal from. Default is 5.")
    parser.add_argument('--num_img_per_client', type=int, default=1,
                        help="Number of images to steal per client. "
                        "Each image is 24x24=576 pixels. Default is 1.")
    parser.add_argument('--agg_mode', type=str, default='segmented',
                        choices=['segmented', 'avg', 'segmented_soft', 'target_only_avg'],
                        help="Global aggregation mode. "
                        "'segmented' is the original hard overwrite segmented aggregation; "
                        "'avg' is standard FedAvg; "
                        "'segmented_soft' blends segment overwrite with the full average; "
                        "'target_only_avg' averages attack positions using only target clients.")
    parser.add_argument('--seg_alpha', type=float, default=0.5,
                        help="Blend coefficient for 'segmented_soft' aggregation. "
                        "1.0 means hard overwrite, 0.0 means full average on attack positions.")
    parser.add_argument('--attack_position_mode', type=str, default='spread',
                        choices=['front', 'spread'],
                        help="How to choose flattened model parameters as attack targets. "
                        "'front' uses the first N flattened parameters; "
                        "'spread' uniformly samples N positions across all flattened parameters.")
    parser.add_argument('--result_tag', type=str, default='',
                        help="Optional suffix tag for result files. Useful for hyperparameter sweeps.")
    parser.add_argument('--cifar_crop_size', type=int, default=24,
                        help="CIFAR crop size. Use 32 for full-size CIFAR training.")
    parser.add_argument('--cifar_normalize', type=int, default=0,
                        help="Whether to apply CIFAR-10 mean/std normalization. Use 1 to enable.")
    parser.add_argument('--noniid_mode', type=str, default='',
                        choices=['', 'shards', 'dirichlet'],
                        help="General non-IID partition mode for all datasets. "
                        "Empty keeps dataset-specific defaults.")
    parser.add_argument('--shards_per_user', type=int, default=0,
                        help="General number of label-sorted shards per client. "
                        "Use 0 to keep dataset-specific defaults.")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.0,
                        help="General Dirichlet alpha for all datasets. "
                        "Use 0 to keep dataset-specific defaults.")
    parser.add_argument('--dirichlet_min_size', type=int, default=10,
                        help="Minimum samples per client for general Dirichlet partitioning.")
    parser.add_argument('--cifar_noniid_mode', type=str, default='shards',
                        choices=['shards', 'dirichlet'],
                        help="CIFAR non-IID partition mode.")
    parser.add_argument('--cifar_shards_per_user', type=int, default=2,
                        help="Number of label-sorted CIFAR shards assigned to each client in shards mode.")
    parser.add_argument('--cifar_dirichlet_alpha', type=float, default=0.5,
                        help="Dirichlet alpha for CIFAR non-IID partitioning. Larger is closer to IID.")
    parser.add_argument('--cifar_dirichlet_min_size', type=int, default=10,
                        help="Minimum samples per client for CIFAR Dirichlet partitioning.")


    args = parser.parse_args()
    return args
