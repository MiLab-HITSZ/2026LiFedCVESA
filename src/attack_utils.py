import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

# 移除scale_factor，不再对窃取数据进行额外缩放
# 让窃取数据保持在[-0.5, 0.5]范围，与模型参数尺度匹配
# scale_factor = 1000.0  # 原来除以1000导致数据尺度太小

def rbg_to_grayscale_pt(images: torch.Tensor) -> torch.Tensor:
    # 灰度转换系数 
    # 系数必须是 float32
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=images.dtype, device=images.device)
    
    # 使用 torch.matmul (等效于 np.dot) 进行点积
    # images[..., :3] 确保只取 RGB 三个通道 (假设通道在最后一维)
    # 结果 shape: (..., 3) @ (3) -> (...) 
    # 灰度图保留 float32 精度
    return torch.matmul(images[..., :3], weights)


def normalize_pt(x: torch.Tensor) -> torch.Tensor:
    # 展平 Tensor
    x_flat = x.flatten()
    
    # 找到 Min 和 Max
    x_min = torch.min(x_flat)
    x_max = torch.max(x_flat)
    
    # 归一化 (Min-Max Scaling)
    if x_max == x_min:
        # 避免除以零，如果所有值相同，则返回零向量
        return torch.zeros_like(x_flat)
    else:
        # 执行归一化操作 (x - min) / (max - min)
        return (x_flat - x_min) / (x_max - x_min)


def cal_error_pt(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # 确保数据类型为整数，与 NumPy 版本中的 astype(int) 对应
    # 如果输入已经是 float，转换为 int 会向下取整/截断。
    # 直接计算 float 上的误差更常见。
    # 进行类型转换
    img1_int = img1.to(torch.int)
    img2_int = img2.to(torch.int)

    # 计算绝对差值的平均值，等价于 np.mean(np.abs(img1 - img2))
    return torch.mean(torch.abs(img1_int - img2_int).float())

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)

def cal_error(img1, img2):
    # 原始实现是灰度图，直接计算 L1 距离作为示例
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    # Mean Absolute Error (MAE)
    mape_value = np.mean(np.abs(img1 - img2))
    return mape_value

def get_x_train_gray_np(dataset_train):
    print("[CVEA Utility] Preparing NumPy grayscale images for MAPE calculation...")
    
    # 使用 DataLoader 获取所有训练数据 (N, C, H, W)
    full_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    # 这里获取的是经过 transform 的 Tensor 
    images_tensor, _ = next(iter(full_loader))
    
    # 形状调整 (N, C, H, W) -> (N, H, W, C)
    images_tensor = images_tensor.permute(0, 2, 3, 1).cpu()
    
    # 转换为灰度
    if images_tensor.size(-1) == 1:
        # 单通道 (MNIST/FMNIST)
        x_train_gray = images_tensor.squeeze(-1)
    else:
        # RGB (CIFAR)
        x_train_gray = rbg_to_grayscale_pt(images_tensor) 
    
    # 转换为 NumPy 并重映射到 [0, 255]
    x_train_gray_np = x_train_gray.numpy()
    x_train_gray_np = (x_train_gray_np * 255).astype(np.uint8)
    
    return x_train_gray_np

# 假设这个函数在 attack_utils.py 或 utils.py 中

def get_ordered_target_images_np(dataset, user_groups, num_users):

    target_indices = []
    
    # 提取前10个客户端的第一张图片的索引（原来是 num_users = 100）
    NUM_TARGET_CLIENTS = 10  # 窃取目标客户端数量
    for i in range(NUM_TARGET_CLIENTS):
        try:
            # 获取客户端 i 的第一个数据索引
            first_image_index = list(user_groups[i])[0]
            target_indices.append(first_image_index)
        except IndexError:
            # 如果某个客户端没有数据，跳过
            continue

    if not target_indices:
        return np.array([])

    # 从原始数据集中按顺序提取图片并处理
    ordered_images_np = []
    
    # 获取原始数据集的 transform，以便进行反向操作（去标准化、转换格式）
    # train_dataset 是通过 get_raw_dataset 获取的，它只做了 ToTensor 和裁剪，
    # 直接处理 Tensor 并转换为 NumPy 灰度格式 [0, 255]。

    # 提取并转换为 NumPy 格式 (H, W)
    for idx in target_indices:
        # dataset[idx][0] 是一个 Tensor (C, H, W) 或 (H, W)
        image_tensor = dataset[idx][0] 
        
        # 转换到 CPU
        image_tensor = image_tensor.cpu()
        
        # 移除单通道维度 (C, H, W) -> (H, W) 或 (C, H, W) -> (H, W, C)
        if image_tensor.dim() == 3:
            if image_tensor.size(0) == 1:
                # 灰度图 (1, H, W) -> (H, W)
                image_np = image_tensor.squeeze(0).numpy()
            else: 
                # 彩图 (3, H, W) -> (H, W, 3)
                image_np = image_tensor.permute(1, 2, 0).numpy()
                # 转换为灰度
                if image_np.shape[-1] == 3:
                    # 灰度转换 
                    image_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
        else: # 已经是 (H, W) 
            image_np = image_tensor.numpy()
        
        # 将值范围从 [0, 1] 转换为 [0, 255] 并转换为 uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        ordered_images_np.append(image_np)
        
    return np.array(ordered_images_np)

def prepare_cvea_stolen_data_pt(net_glob, dataset_train, args):
    # 如果 gama <= 0，则不进行攻击
    if getattr(args, 'gama', 0.0) <= 0:
        print("[CVEA Attack] Attack disabled (gama=0).")
        return None

    print(f"\n[CVEA Attack] Preparing stolen data with gama={args.gama}")

    # 获取目标权重总数
    num_target_params = 0
    for name, param in net_glob.named_parameters():
        # 攻击针对维度 > 1 的权重 (Conv/Linear weights)
        if param.dim() > 1:
            num_target_params += param.numel()

    if num_target_params == 0:
        print('Error: Model has no suitable parameters for CVEA attack.')
        return None

    print(f"[CVEA Attack] Target parameter count: {num_target_params}")

    # 获取原始训练数据 Tensor    
    full_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    data_iter = iter(full_loader)
    images_tensor, _ = next(data_iter) # images_tensor shape: (N, C, H, W)
    images_tensor = images_tensor.to(args.device)

    # 转换为灰度（Channel Last for rbg_to_grayscale_pt）
    # CIFAR/MNIST 的形状调整
    if images_tensor.dim() == 4:
        # PyTorch 格式 (N, C, H, W)，转换为 (N, H, W, C) 以适应 rbg_to_grayscale_pt
        if images_tensor.size(1) in [1, 3]:
             images_tensor = images_tensor.permute(0, 2, 3, 1)

    # 转换为灰度 (N, H, W)
    # MNIST (N, H, W, 1)
    if images_tensor.size(-1) == 1:
        x_train_gray = images_tensor.squeeze(-1) # 移除单通道
    else:
        # RGB (N, H, W, 3) 转换为灰度 (N, H, W)
        x_train_gray = rbg_to_grayscale_pt(images_tensor)


    # 展平并截断/重复
    stolen_data_flat = x_train_gray.flatten()

    if stolen_data_flat.numel() < num_target_params:
        print(f"[CVEA Warning] Not enough data for attack. Available: {stolen_data_flat.numel()}")
        # 如果数据不够，重复数据来匹配长度
        num_repeats = (num_target_params + stolen_data_flat.numel() - 1) // stolen_data_flat.numel()
        stolen_data_flat = stolen_data_flat.repeat(num_repeats)[:num_target_params]
    else:
        stolen_data_flat = stolen_data_flat[:num_target_params]

    # 归一化和中心化 (得到 d_m)
    stolen_data_normalized = normalize_pt(stolen_data_flat)
    d_mean = torch.mean(stolen_data_normalized)
    d_m_attack = stolen_data_normalized - d_mean

    # 将 d_m 移到 args.device
    d_m_attack = d_m_attack.to(args.device)
    print(f"[CVEA Attack] Prepared d_m with length {d_m_attack.numel()}")

    return d_m_attack

def prepare_cvea_stolen_data(net_glob, dataset_train, args, user_groups):
    # 如果 gama <= 0，则不进行攻击
    if getattr(args, 'gama', 0.0) <= 0:
        print("[CVEA Attack] Attack disabled (gama=0).")
        return None

    print(f"\n[CVEA Attack] Preparing stolen data with gama={args.gama}")

    # 获取目标权重总数：从100个客户端改为10个客户端
    NUM_TARGET_CLIENTS = 10  # 窃取目标客户端数量（原来是 args.num_users = 100）
    num_target_params = NUM_TARGET_CLIENTS * 576  # 每个客户端对应一张 24x24 的图片，共10张
    # 获取模型全部参数数量
    # for name, param in net_glob.named_parameters():
    #     num_target_params += param.numel()

    if num_target_params == 0:
        print('Error: Model has no suitable parameters for CVEA attack.')
        return None

    print(f"[CVEA Attack] Target parameter count: {num_target_params}")

    # 按客户端顺序构建索引列表：只窃取前10个客户端的数据
    
    # 每个客户端至少有一张图片，且我们只关注每个客户端的第一张图片
    # 创建一个空的索引列表，用于存储目标图片的索引
    target_indices = []
    
    # 遍历前10个客户端（原来是 args.num_users = 100）
    for i in range(NUM_TARGET_CLIENTS):
        # user_groups[i] 是一个集合，包含客户端 i 的所有数据索引
        try:
            # 找到客户端 i 拥有的第一个索引
            # 在 IID/Non-IID 分组后，集合中的第一个元素是可重复访问的。
            first_image_index = list(user_groups[i])[0]
            target_indices.append(first_image_index)
        except IndexError:
            # 如果某个客户端没有数据，跳过或记录警告
            print(f"[CVEA Warning] Client {i} has no data. Skipping.")

    if not target_indices:
        print("[CVEA Error] No target images found.")
        return None

    # 从原始数据集中提取这些目标图片
    # 创建一个新的子集，只包含这些目标索引，并按顺序排列
    # 使用 SequentialSampler 确保顺序，但由于我们已手动创建索引列表，DataLoader 默认即可
    
    # 仅加载目标图片 (N_target, C, H, W)
    target_images = []
    for idx in target_indices:
        # dataset_train[idx][0] 获取图像 Tensor
        target_images.append(dataset_train[idx][0])
        
    # 将 list of Tensors 转换为单个 Tensor
    images_tensor = torch.stack(target_images, dim=0).to(args.device)

    # images_tensor shape: (K, C, H, W) 
    
    # 转换为灰度（Channel Last for rbg_to_grayscale_pt）
    if images_tensor.dim() == 4:
        # PyTorch 格式 (N, C, H, W)，转换为 (N, H, W, C)
        if images_tensor.size(1) in [1, 3]:
            images_tensor = images_tensor.permute(0, 2, 3, 1)

    # 转换为灰度 (N, H, W)
    # MNIST (N, H, W, 1)
    if images_tensor.size(-1) == 1:
        x_train_gray = images_tensor.squeeze(-1) # 移除单通道
    else:
        # RGB (N, H, W, 3) 转换为灰度 (N, H, W)
        x_train_gray = rbg_to_grayscale_pt(images_tensor)


    # 展平并截断/重复 (现在 stolen_data_flat 的顺序就是 Client 0, Client 1, ...)
    stolen_data_flat = x_train_gray.flatten()

    if stolen_data_flat.numel() < num_target_params:
        print(f"[CVEA Warning] Not enough data for attack. Available: {stolen_data_flat.numel()}")
        # 如果数据不够，重复数据来匹配长度
        num_repeats = (num_target_params + stolen_data_flat.numel() - 1) // stolen_data_flat.numel()
        stolen_data_flat = stolen_data_flat.repeat(num_repeats)[:num_target_params]
    else:
        stolen_data_flat = stolen_data_flat[:num_target_params]

    # 归一化和中心化 (得到 d_m)
    # 假设 normalize_pt 函数存在
    stolen_data_normalized = normalize_pt(stolen_data_flat)
    d_mean = torch.mean(stolen_data_normalized)
    d_m_attack = stolen_data_normalized - d_mean

    # 将 d_m 移到 args.device
    d_m_attack = d_m_attack.to(args.device)
    print(f"[CVEA Attack] Prepared d_m with length {d_m_attack.numel()}")

    return d_m_attack

# 相关值编码攻击
# 返回值是相关系数的绝对值，作为损失，值的范围是[0, 1]
def cor_attack(model, d_m):
    # 只使用前5760个参数计算相关性，与分段聚合策略一致
    # 遍历模型的所有可训练参数并展平
    params = []
    
    for param in model.parameters():
        # 现在提取所有参数（权重、偏置项等）
        params.append(param.view(-1))
    
    # 如果模型没有可训练参数，则返回 0
    if not params:
        # 确保返回的 Tensor 在正确的设备上
        return torch.tensor(0.0, device=d_m.device)
        
    # 拼接成一个大的向量 p_flat
    p_flat = torch.cat(params, dim=0)
    
    # 只使用前 d_m.size(0) 个参数（即前5760个）
    # 这样相关性计算只关注被攻击的参数段，不被其他参数稀释
    target_len = d_m.size(0)  # 5760
    if p_flat.size(0) < target_len:
        # 如果模型参数不足，填充零
        padding = torch.zeros(target_len - p_flat.size(0), device=p_flat.device, dtype=p_flat.dtype)
        p_flat = torch.cat([p_flat, padding], dim=0)
    
    # 只取前 target_len 个参数
    p_flat = p_flat[:target_len]
    d_m = d_m[:target_len]
    

    # 计算中心化权重 p_m = params - mean(params)
    # 确保在计算均值时使用浮点数
    p_mean = torch.mean(p_flat.float())
    p_m = p_flat - p_mean
    
    # 计算相关性函数 (皮尔逊相关系数)
    
    # r_num = sum(p_m * d_m) (协方差的分子)
    r_num = torch.sum(p_m * d_m)
    
    # r_den = sqrt(sum(p_m^2) * sum(d_m^2)) (模长乘积)
    r_den = torch.sqrt(torch.sum(p_m ** 2) * torch.sum(d_m ** 2))
    
    # r (皮尔逊相关系数)
    # 避免除以零
    epsilon = 1e-8 
    r = r_num / (r_den + epsilon)
    
    # loss = |r|
    loss = torch.abs(r)
    # print(f"loss_cor value: {loss.item():.4f}")
    return loss


# 评估窃取到的数据与原始数据的相似性
def calculate_cor_mape(model, x_train, args):
    # 1. 统一提取逻辑：必须与 cor_attack 中的参数提取完全一致
    params_list = []
    for param in model.parameters():
        params_list.append(param.detach().cpu().numpy().flatten())
    
    if not params_list:
        return 0.0

    params = np.concatenate(params_list)
    
    # 2. 确定目标参数段
    num_image = 10 
    h, w = x_train.shape[1], x_train.shape[2]
    num_pixel = h * w
    target_len = num_image * num_pixel
    
    # 截取参与编码的那部分参数
    params_segment = params[:target_len]
    
    mape = 0
    for i in range(num_image):
        # 提取单张图片的参数段
        img_params = params_segment[i*num_pixel : (i+1)*num_pixel]
        
        # 【关键改进】动态 Min-Max 归一化
        # 因为相关性攻击不保证数值大小，只保证线性一致
        p_min, p_max = img_params.min(), img_params.max()
        if p_max > p_min:
            # 映射到 [0, 1]
            img_recovered = (img_params - p_min) / (p_max - p_min)
        else:
            img_recovered = np.zeros_like(img_params)
            
        img_recovered = img_recovered.reshape(h, w)
        
        # 获取原始图像并归一化到 [0, 1]
        original_img = x_train[i].astype(np.float32) / 255.0
        
        # 计算正向和反向误差 (因为相关性可能是负相关 r=-1)
        err1 = np.mean(np.abs(img_recovered - original_img))
        err2 = np.mean(np.abs((1.0 - img_recovered) - original_img))
        
        mape += min(err1, err2)
    
    return mape / num_image


# 恢复窃取到的数据
def recover_cor_stolen_data(model, x_train):
    # 遍历模型的所有可训练参数并展平
    params = []
    for param in model.parameters():
        if param.dim() > 1:
            params.append(param.detach().cpu().numpy().reshape(-1, 1))
            
    if not params:
        print("Model has no suitable parameters to recover.")
        return np.array([])
        
    # 拼接成一个大的向量 (NumPy)
    params = np.concatenate(params, axis=0)
    
    # 【修复】不再使用normalize，直接使用参数原始值
    # 窃取数据在 [-0.5, 0.5] 范围，需要映射到 [0, 255]
    params = (params + 0.5)  # [-0.5, 0.5] -> [0, 1]
    params = np.clip(params, 0, 1)  # 确保在 [0, 1] 范围内
    
    # 重新组织成图片的格式
    if x_train.ndim == 4 and x_train.shape[-1] == 1:
        x_train = x_train.squeeze(-1)
        
    total_pix = np.prod(x_train.shape[1:])
    number = int(params.shape[0] / total_pix)
    
    print("steal number:", number)
    
    params = params[0:number * total_pix]
    # x_train 的形状是 (N, H, W)
    params = params.reshape(number, x_train.shape[1], x_train.shape[2])
    
    # 重映射到 [0, 255]
    params = (params * 255).astype(np.uint8)
    
    return params


def recover_cor_stolen_data_new(model, x_train):
    # 保持参数提取逻辑一致
    params_list = []
    for param in model.parameters():
        params_list.append(param.detach().cpu().numpy().flatten())
    
    params = np.concatenate(params_list)
    
    num_image = 10
    h, w = x_train.shape[1], x_train.shape[2]
    num_pixel = h * w
    
    recovered_images = []
    
    for i in range(num_image):
        img_params = params[i*num_pixel : (i+1)*num_pixel]
        
        # 动态归一化
        p_min, p_max = img_params.min(), img_params.max()
        if p_max > p_min:
            img_norm = (img_params - p_min) / (p_max - p_min)
        else:
            img_norm = np.zeros_like(img_params)
            
        # 转换为 [0, 255] uint8 用于绘图
        img_uint8 = (img_norm * 255).astype(np.uint8)
        recovered_images.append(img_uint8.reshape(h, w))
        
    return np.array(recovered_images)
