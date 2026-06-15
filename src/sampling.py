#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def _get_labels_np(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    if hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels
        if hasattr(labels, 'numpy'):
            return labels.numpy()
        return np.array(labels)
    if hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    raise AttributeError(
        f"{dataset.__class__.__name__} does not expose targets/train_labels/labels"
    )


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def label_sorted_shards_noniid(dataset, num_users, shards_per_user=2):
    """
    Sample label-skewed non-IID data by sorting labels and assigning shards.
    """
    num_shards = num_users * shards_per_user
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = _get_labels_np(dataset)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0
            ).astype(np.int64)
    return dict_users


def mnist_noniid(dataset, num_users, shards_per_user=2):
    """
    Sample non-I.I.D client data from MNIST/Fashion-MNIST dataset.
    """
    return label_sorted_shards_noniid(dataset, num_users, shards_per_user)


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, shards_per_user=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    return label_sorted_shards_noniid(dataset, num_users, shards_per_user)


def dirichlet_noniid(dataset, num_users, alpha=0.5, min_size=10):
    """
    Split data by class-wise Dirichlet proportions.

    Smaller alpha makes clients more label-skewed; larger alpha approaches IID.
    This keeps all training images and usually gives a less pathological
    non-IID setting than sorted shards.
    """
    labels = _get_labels_np(dataset)
    num_classes = len(np.unique(labels))
    min_client_size = 0

    while min_client_size < min_size:
        dict_users = {i: [] for i in range(num_users)}

        for class_id in range(num_classes):
            class_idxs = np.where(labels == class_id)[0]
            np.random.shuffle(class_idxs)

            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions)[:-1] * len(class_idxs)).astype(int)

            for client_id, idxs in enumerate(np.split(class_idxs, split_points)):
                dict_users[client_id].extend(idxs.tolist())

        min_client_size = min(len(idxs) for idxs in dict_users.values())

    for client_id in dict_users:
        np.random.shuffle(dict_users[client_id])
        dict_users[client_id] = np.array(dict_users[client_id], dtype=np.int64)

    return dict_users


def cifar_noniid_dirichlet(dataset, num_users, alpha=0.5, min_size=10):
    """
    Split CIFAR-10 by class-wise Dirichlet proportions.
    """
    return dirichlet_noniid(dataset, num_users, alpha, min_size)


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
