#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9


import os
import copy
import time
import math
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import *
from utils import *
from attack_utils import *
from plot import *

from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 移除scale_factor，让窃取数据保持在[-0.5, 0.5]范围，与模型参数尺度匹配
# scale_factor = 1000.0  # 原来除以1000导致数据尺度太小

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    
    # 使用 getattr 安全地获取 gama，如果未设置则默认为 0.0
    gama_target = getattr(args, 'gama', 0.0)  # 目标 gama 值
    gama_val = gama_target  # 当前实际使用的 gama 值
    gama_warmup_epochs = getattr(args, 'gama_warmup_epochs', 100)  # warm-up 总周期
    
    if gama_target > 0 and gama_warmup_epochs > 0:
        print(f'[Gama Warmup] Enabled: gama 0 -> {gama_target} over {gama_warmup_epochs} epochs')
        print(f'  Phase1 (epoch 0~{gama_warmup_epochs//2-1}): gama=0, 纯分类训练')
        print(f'  Phase2 (epoch {gama_warmup_epochs//2}~{gama_warmup_epochs-1}): 余弦增长 0 -> {gama_target}')
        print(f'  Phase3 (epoch {gama_warmup_epochs}+): 保持 gama={gama_target}')
    elif gama_target > 0:
        print(f'[Gama Warmup] Disabled: using constant gama={gama_target}')

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    # raw:只进行裁剪到24*24 和 ToTensor（维度顺序变换(H, W, C)-->(C, H, W)，值范围变为0-1）
    train_endataset, test_endataset, user_engroups = get_dataset(args)
    train_dataset, test_dataset, user_groups = get_raw_dataset(args)

    # 提取 NumPy 格式的灰度图像用于 MAPE 对比
    # 维度顺序变换(C, H, W)-->(H, W, C), 值范围变为0-255，转换为灰度
    # 保持了24*24的裁剪
    # 只提取前10个客户端的图片（原来是 args.num_users = 100）
    x_train_gray_np = get_ordered_target_images_np(train_dataset, user_groups, args.num_users, num_steal=args.num_steal, num_img_per_client=args.num_img_per_client)
    x_trainen_en_gray_np = get_x_train_gray_np(train_endataset)

    if x_train_gray_np.size > 0:
        args.attack_h = x_train_gray_np.shape[1]
        args.attack_w = x_train_gray_np.shape[2]
    else:
        sample_img = train_dataset[0][0]
        args.attack_h = sample_img.shape[-2]
        args.attack_w = sample_img.shape[-1]
    args.attack_num_pixel = args.attack_h * args.attack_w

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNFashion_Enhanced(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Enhanced(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar_Enhanced_V3(args=args)

    elif args.model == 'resnet18':
        # 使用原本的 CNNCifar_ResNet_V1（BasicBlock + 攻击层 + 3层残差骨干）
        if args.dataset == 'cifar':
            global_model = CNNCifar_ResNet_V1(args=args)
        else:
            exit('Error: ResNet18 currently only supports CIFAR dataset')

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
    # 传入数据集:raw:只进行裁剪到24*24 和 ToTensor（维度顺序变换(H, W, C)-->(C, H, W)，值范围变为0-1）
    # 拿到的是:raw转为灰度 (N, H, W)，且经过归一化和中心化，且最终展平，值范围约为[-0.5, 0.5]
    stolen_data_dm = prepare_cvea_stolen_data(global_model, train_dataset, args, user_groups)
    # 不再除以scale_factor，保持数据在合理尺度
    # stolen_data_dm = stolen_data_dm / scale_factor

    # ---------------temp------------------ #
    if args.gama > 0:
        total_steal_images = args.num_steal * args.num_img_per_client
        plot_x_train_gray_np(
            x_train_gray_np,
            num_to_plot=total_steal_images,
            title="Original Stolen Images from Clients",
            rows=max(1, total_steal_images // 5),
            save_path=f"original_stolen_images_numSteal[{args.num_steal}].png"
        )
        plot_stolen_data_dm(
            stolen_data_dm,
            H=args.attack_h,
            W=args.attack_w,
            num_images=total_steal_images,
            num_to_plot=total_steal_images,
            save_path=f"stolen_data_dm_visualization_numSteal[{args.num_steal}].png"
        )
    # ---------------temp------------------ #

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
        
        # === Gama Warm-up 逐渐增大 ===
        if gama_target > 0 and gama_warmup_epochs > 0:
            half = gama_warmup_epochs // 2  # 前半段 gama=0，后半段余弦增长
            if epoch < half:
                # Phase1: gama=0，纯分类训练，不攻击
                gama_val = 0.0
            elif epoch < gama_warmup_epochs:
                # Phase2: 余弦增长，从 0 增长到 gama_target
                progress = (epoch - half) / (gama_warmup_epochs - half)
                gama_val = gama_target * 0.5 * (1 - math.cos(math.pi * progress))
            else:
                # Phase3: 保持目标值
                gama_val = gama_target
            args.gama = gama_val  # 动态更新，传递给 LocalUpdate
            
            if epoch % 10 == 0:
                tqdm.write(f'[Gama Warmup] Epoch {epoch+1}: gama = {gama_val:.6f} (target={gama_target})')
        
        # 确保前 num_steal 个目标客户端每轮都参与训练（攻击关键）
        NUM_TARGET_CLIENTS = args.num_steal
        if args.gama > 0:
            # 攻击模式：前 num_steal 个客户端必须参与，其余随机选择
            target_clients = list(range(NUM_TARGET_CLIENTS))
            m = max(int(args.frac * args.num_users), NUM_TARGET_CLIENTS)
            
            # 从剩余客户端中随机选择
            remaining_slots = m - NUM_TARGET_CLIENTS
            if remaining_slots > 0 and args.num_users > NUM_TARGET_CLIENTS:
                other_clients = np.random.choice(
                    range(NUM_TARGET_CLIENTS, args.num_users), 
                    min(remaining_slots, args.num_users - NUM_TARGET_CLIENTS), 
                    replace=False
                ).tolist()
                idxs_users = target_clients + other_clients
            else:
                idxs_users = target_clients
            
            # 【调试】打印参与训练的客户端
            if epoch % 10 == 0:
                tqdm.write(f'[Debug] Epoch {epoch+1} - Participating clients: {sorted(idxs_users)[:15]}...')
        else:
            # 正常模式：随机采样
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_endataset,
                                      idxs=user_groups[idx], logger=logger, stolen_data_dm=stolen_data_dm)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # 获取上一轮的全局权重作为基准
        prev_global_weights = copy.deepcopy(global_model.state_dict())

        # update global weights
        if args.gama > 0:
        # 使用分段聚合，并传入参与本轮训练的客户端 ID 列表
            global_weights = segmented_average_weights(
                local_weights,
                idxs_users,
                prev_global_weights,
                num_steal=args.num_steal,
                num_img_per_client=args.num_img_per_client,
                attack_num_pixel=args.attack_num_pixel,
            )
        else:
            # 否则，使用标准平均聚合
            global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # # 打印模型权重
        # if (epoch + 1) % 20 == 0:
        #     tqdm.write(f"\n--- Model Weights at Global Round {epoch + 1} ---")
            
        #     # 遍历模型的状态字典 (state_dict)
        #     for name, param in global_model.named_parameters():
        #         # 仅打印前几层或关键层的权重，避免输出过长
        #         # 打印第一个卷积层或全连接层的权重摘要
        #         if 'conv' in name or 'fc' in name or '0.weight' in name:
        #             # 打印层名、权重形状和权重的均值、标准差等统计信息
        #             # 打印权重的统计信息比打印完整的张量更有用
        #             tqdm.write(
        #                 f"Layer: {name}, "
        #                 f"Shape: {param.shape}, "
        #                 f"Mean: {param.data.mean().item():.6f}, "
        #                 f"Std: {param.data.std().item():.6f}"
        #             )
        #     tqdm.write("-------------------------------------------\n")

        args.lr *= args.lr_decay

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        current_mape = 0.0
        if stolen_data_dm is not None and args.gama > 0:
            # 简化后的 MAPE 计算
            current_mape = calculate_cor_mape(global_model, x_train_gray_np, args)
            mape_list.append(current_mape)
            
            # 每10个epoch恢复一次窃取图片并保存
            if (epoch + 1) % 10 == 0 or epoch == 0:  # 每10轮或第1轮恢复一次
                recovered_images = recover_cor_stolen_data_new(global_model, x_train_gray_np, num_steal=args.num_steal, num_img_per_client=args.num_img_per_client, args=args)
                
                if recovered_images.size > 0:
                    total_steal_images = args.num_steal * args.num_img_per_client
                    num_images_to_plot = min(total_steal_images, recovered_images.shape[0])
                    original_images = x_train_gray_np[:num_images_to_plot]
                    
                    import matplotlib
                    import matplotlib.pyplot as plt
                    matplotlib.use('Agg')
                    
                    fig, axes = plt.subplots(2, num_images_to_plot, figsize=(15, 6))
                    fig.suptitle(f'Epoch {epoch+1}: Original vs. Recovered (MAPE={current_mape:.4f}, Gama={gama_val})', fontsize=16)
                    
                    for i in range(num_images_to_plot):
                        # 原始图像
                        axes[0, i].imshow(original_images[i], cmap='gray')
                        axes[0, i].set_title(f'Original {i+1}')
                        axes[0, i].axis('off')
                        
                        # 恢复图像
                        img_i = recovered_images[i]
                        img_inverted_np = np.asarray(ImageOps.invert(Image.fromarray(img_i)))
                        
                        err1 = cal_error(img_i, original_images[i])
                        err2 = cal_error(img_inverted_np, original_images[i])
                        
                        if err1 < err2:
                            axes[1, i].imshow(img_i, cmap='gray')
                            axes[1, i].set_title(f'Recovered {i+1}\n(Err:{err1:.4f})')
                        else:
                            axes[1, i].imshow(img_inverted_np, cmap='gray')
                            axes[1, i].set_title(f'Recovered-Inv {i+1}\n(Err:{err2:.4f})')
                        
                        axes[1, i].axis('off')
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # 保存到epoch专用目录
                    epoch_plot_dir = './save/plots/epoch_recovery'
                    os.makedirs(epoch_plot_dir, exist_ok=True)
                    plot_save_path = os.path.join(
                        epoch_plot_dir,
                        f'epoch_{epoch+1:03d}_recovery_{args.dataset}_numSteal[{args.num_steal}].png'
                    )
                    plt.savefig(plot_save_path, dpi=150)
                    plt.close(fig)
                    
                    tqdm.write(f'[Epoch {epoch+1}] Recovery plot saved to {plot_save_path}')

        global_rounds.set_description(f"Epoch {epoch+1} | Loss: {loss_avg:.4f} | Mape: {current_mape:.4f} | Gama: {gama_val:.4f}")

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

    # 定义包含攻击参数的文件名基准
    result_file_base = '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_Gama[{}]_numSteal[{}]'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs, gama_val, args.num_steal)

    # 保存结果到 .npy 文件
    save_dir = './save/results'
    os.makedirs(save_dir, exist_ok=True)
    
    if mape_list:
        np.save(os.path.join(save_dir, result_file_base + '_mape.npy'), np.array(mape_list))
        print(f"Saved MAPE results to {save_dir}/{result_file_base}_mape.npy")
    
    np.save(os.path.join(save_dir, result_file_base + '_loss.npy'), np.array(global_loss_list))
    np.save(os.path.join(save_dir, result_file_base + '_acc.npy'), np.array(global_acc_list))
    print(f"Saved Loss and Accuracy results to {save_dir}/")
    
    # 保存原有的 train_loss 和 train_accuracy 对象 
    file_name = './save/objects/{}.pkl'.format(result_file_base)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    plot_dir = './save/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot Loss curve 
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
    
    # 绘制 MAPE 曲线
    if mape_list:
        plt.figure()
        plt.title('MAPE vs Communication rounds (Gama={})'.format(gama_val))
        plt.plot(range(len(mape_list)), mape_list, color='b')
        plt.ylabel('Mean Absolute Percentage Error (MAPE)')
        plt.xlabel('Communication Rounds')
        plt.savefig(os.path.join(plot_dir, '{}_mape.png'.format(result_file_base)))
        
    # 图像恢复和对比绘图（训练结束后的最终版本）
    if stolen_data_dm is not None and gama_val > 0:
        print('\n[Final] Starting final data recovery and plotting...')
        
        # 恢复窃取数据 (NumPy 数组, 灰度 [0, 255])
        recovered_images = recover_cor_stolen_data_new(global_model, x_train_gray_np, num_steal=args.num_steal, num_img_per_client=args.num_img_per_client, args=args)
        
        if recovered_images.size > 0:
            total_steal_images = args.num_steal * args.num_img_per_client
            num_images_to_plot = min(total_steal_images, recovered_images.shape[0])
            
            # 原始图像（用于对比）
            original_images = x_train_gray_np[:num_images_to_plot]

            fig, axes = plt.subplots(2, num_images_to_plot, figsize=(15, 6))
            fig.suptitle(f'[FINAL] CVEA: Original vs. Recovered Data (Gama={gama_val})', fontsize=16)

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
            
            # 保存最终对比图
            plot_save_path = os.path.join(plot_dir, '{}_final_comparison.png'.format(result_file_base))
            plt.savefig(plot_save_path)
            print(f'[Final] Comparison plot saved to {plot_save_path}')
            print(f'[Info] Epoch-wise recovery plots are saved in ./save/plots/epoch_recovery/')
            
        else:
            print("No images were recovered for plotting.")
