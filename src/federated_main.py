#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_new
from utils import get_dataset, average_weights, exp_details, get_raw_dataset
from attack_utils import *

from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.datasets as datasets

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_endataset, test_endataset, user_engroups = get_dataset(args)

    train_dataset, test_dataset, user_groups = get_raw_dataset(args)

    # 提取 NumPy 格式的灰度图像用于 MAPE 对比
    x_train_gray_np = get_x_train_gray_np(train_dataset)
    x_trainen_en_gray_np = get_x_train_gray_np(train_endataset)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar_new(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    args.device = device
    stolen_data_dm = prepare_cvea_stolen_data_pt(global_model, train_dataset, args)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []

    # 用于存储每一轮结果的新列表
    mape_list = []
    global_acc_list = []
    global_loss_list = []

    print_every = 10
    val_loss_pre, counter = 0, 0

    global_rounds = tqdm(range(args.epochs))

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_endataset,
                                      idxs=user_groups[idx], logger=logger, stolen_data_dm=stolen_data_dm)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        args.lr *= args.lr_decay

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        current_mape = 0.0
        if stolen_data_dm is not None and args.gama > 0:
            # 简化后的 MAPE 计算
            current_mape = calculate_cor_mape(global_model, x_train_gray_np) 
            mape_list.append(current_mape)

        global_rounds.set_description(f"Epoch {epoch+1} | Loss: {loss_avg:.4f} | Mape: {current_mape}")

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_endataset,
                                      idxs=user_groups[c], logger=logger, stolen_data_dm=stolen_data_dm)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)


        global_acc = sum(list_acc)/len(list_acc)
        global_loss = sum(list_loss)/len(list_loss)
        
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            # print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            tqdm.write(f' \nAvg Training Stats after {epoch+1} global rounds:')
            tqdm.write(f'Training Loss : {np.mean(np.array(train_loss)):.4f}')
            tqdm.write('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # 1. 定义包含攻击参数的文件名基准
    # 使用 getattr 安全地获取 gama，如果未设置则默认为 0.0
    gama_val = getattr(args, 'gama', 0.0)
    result_file_base = '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_Gama[{}]'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs, gama_val)

    # 2. 保存结果到 .npy 文件
    save_dir = './save/results'
    os.makedirs(save_dir, exist_ok=True)
    
    if mape_list:
        np.save(os.path.join(save_dir, result_file_base + '_mape.npy'), np.array(mape_list))
        print(f"Saved MAPE results to {save_dir}/{result_file_base}_mape.npy")
    
    np.save(os.path.join(save_dir, result_file_base + '_loss.npy'), np.array(global_loss_list))
    np.save(os.path.join(save_dir, result_file_base + '_acc.npy'), np.array(global_acc_list))
    print(f"Saved Loss and Accuracy results to {save_dir}/")
    
    # 3. 保存原有的 train_loss 和 train_accuracy 对象 (为保持兼容性)
    file_name = './save/objects/{}.pkl'.format(result_file_base)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # --- PLOTTING ---
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    plot_dir = './save/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot Loss curve (原代码)
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(plot_dir, '{}_loss.png'.format(result_file_base)))
    
    # Plot Average Accuracy vs Communication rounds (原代码)
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(plot_dir, '{}_acc.png'.format(result_file_base)))

    # --- 攻击相关绘图 ---
    
    # 4. 绘制 MAPE 曲线
    if mape_list:
        plt.figure()
        plt.title('MAPE vs Communication rounds (Gama={})'.format(gama_val))
        plt.plot(range(len(mape_list)), mape_list, color='b')
        plt.ylabel('Mean Absolute Percentage Error (MAPE)')
        plt.xlabel('Communication Rounds')
        plt.savefig(os.path.join(plot_dir, '{}_mape.png'.format(result_file_base)))
        
    # 5. 图像恢复和对比绘图
    if stolen_data_dm is not None and gama_val > 0:
        print('\nStarting data recovery and plotting...')
        
        # 恢复窃取数据 (NumPy 数组, 灰度 [0, 255])
        # x_train_gray_np 必须在 federated_main.py 顶部已定义
        # recover_cor_stolen_data 必须在 attack_utils.py 中定义
        recovered_images = recover_cor_stolen_data(global_model, x_train_gray_np)
        
        if recovered_images.size > 0:
            num_images_to_plot = min(5, recovered_images.shape[0])
            
            # 原始图像（用于对比）
            original_images = x_train_gray_np[:num_images_to_plot]

            fig, axes = plt.subplots(2, num_images_to_plot, figsize=(15, 6))
            fig.suptitle(f'CVEA: Original vs. Recovered Data (Gama={gama_val})', fontsize=16)

            for i in range(num_images_to_plot):
                # 原始图像
                axes[0, i].imshow(original_images[i], cmap='gray')
                axes[0, i].set_title(f'Original {i+1}')
                axes[0, i].axis('off')

                # 恢复图像
                img_i = recovered_images[i]
                
                # 计算反色恢复图像的误差（MAPE 使用了最小误差原则）
                img_inverted_np = np.asarray(ImageOps.invert(Image.fromarray(img_i)))
                
                # 使用 cal_error (MAE) 来决定哪个版本更接近
                err1 = cal_error(img_i, original_images[i])
                err2 = cal_error(img_inverted_np, original_images[i])
                
                # 决定展示哪个版本
                if err1 < err2:
                    axes[1, i].imshow(img_i, cmap='gray')
                    axes[1, i].set_title(f'Recovered {i+1} (Err:{err1:.4f})')
                else:
                    axes[1, i].imshow(img_inverted_np, cmap='gray')
                    axes[1, i].set_title(f'Recovered-Inv {i+1} (Err:{err2:.4f})')

                axes[1, i].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 保存对比图
            plot_save_path = os.path.join(plot_dir, '{}_gama_comparison.png'.format(result_file_base))
            plt.savefig(plot_save_path)
            print(f'Comparison plot saved to {plot_save_path}')
            
        else:
            print("No images were recovered for plotting.")