#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='FedCVESA v5 experiments')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of global communication rounds')
    parser.add_argument('--num_users', type=int, default=10,
                        help='number of federated clients')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of clients selected each round')
    parser.add_argument('--local_ep', type=int, default=10,
                        help='local epochs per selected client')
    parser.add_argument('--local_bs', type=int, default=16,
                        help='local batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='SGD weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['exp', 'cosine'],
                        help='learning-rate scheduler')
    parser.add_argument('--min_lr', type=float, default=0.0001,
                        help='minimum learning rate for cosine scheduling')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='per-round decay factor when --lr_scheduler=exp')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar'],
                        help='dataset name')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18_cifar'],
                        help='cnn for mnist/fmnist; resnet18_cifar for cifar')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of output classes')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='kept for compatibility with older checkpoints/scripts')
    parser.add_argument('--gpu', default=None,
                        help='CUDA device id; omit for CPU')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer name printed in experiment details')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    parser.add_argument('--iid', type=int, default=0,
                        help='1 for IID split, 0 for non-IID split')
    parser.add_argument('--unequal', type=int, default=0,
                        help='legacy unequal split flag for MNIST-style shards')
    parser.add_argument('--noniid_mode', type=str, default='dirichlet',
                        choices=['', 'shards', 'dirichlet'],
                        help='non-IID partition mode')
    parser.add_argument('--shards_per_user', type=int, default=0,
                        help='label-sorted shards per user when using shard split')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID partitioning')
    parser.add_argument('--dirichlet_min_size', type=int, default=100,
                        help='minimum samples per client for Dirichlet split')

    parser.add_argument('--gama', type=float, default=0.5,
                        help='CVEA regularization coefficient; 0 disables attack')
    parser.add_argument('--gama_warmup_epochs', type=int, default=0,
                        help='warm-up rounds for gama; v5 uses 0')
    parser.add_argument('--num_steal', type=int, default=5,
                        help='number of target clients')
    parser.add_argument('--num_img_per_client', type=int, default=1,
                        help='encoded images per target client')
    parser.add_argument('--agg_mode', type=str, default='segmented',
                        choices=['segmented', 'avg', 'segmented_soft', 'target_only_avg'],
                        help='server aggregation mode')
    parser.add_argument('--seg_alpha', type=float, default=0.5,
                        help='blend coefficient for segmented_soft aggregation')
    parser.add_argument('--attack_position_mode', type=str, default='spread',
                        choices=['front', 'spread'],
                        help='carrier-position selection mode')
    parser.add_argument('--result_tag', type=str, default='',
                        help='optional suffix tag for output files')

    parser.add_argument('--cifar_crop_size', type=int, default=32,
                        help='CIFAR crop size; v5 uses 32')
    parser.add_argument('--cifar_normalize', type=int, default=1,
                        help='1 to apply CIFAR-10 mean/std normalization')
    parser.add_argument('--cifar_noniid_mode', type=str, default='dirichlet',
                        choices=['shards', 'dirichlet'],
                        help='fallback CIFAR non-IID mode')
    parser.add_argument('--cifar_shards_per_user', type=int, default=2,
                        help='fallback CIFAR shards per user')
    parser.add_argument('--cifar_dirichlet_alpha', type=float, default=0.5,
                        help='fallback CIFAR Dirichlet alpha')
    parser.add_argument('--cifar_dirichlet_min_size', type=int, default=100,
                        help='fallback CIFAR Dirichlet minimum samples')

    return parser.parse_args()
