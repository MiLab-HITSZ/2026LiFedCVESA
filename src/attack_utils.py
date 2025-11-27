import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

def rbg_to_grayscale_pt(images: torch.Tensor) -> torch.Tensor:
    # 灰度转换系数 (标准的亮度计算)
    # 系数必须是 float32
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=images.dtype, device=images.device)
    
    # 使用 torch.matmul (等效于 np.dot) 进行点积
    # images[..., :3] 确保只取 RGB 三个通道 (假设通道在最后一维)
    # 结果 shape: (..., 3) @ (3) -> (...) 
    # 灰度图通常保留 float32 精度
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
    # 注意：如果输入已经是 float，转换为 int 会向下取整/截断。
    # 在 PyTorch 中，通常不需要强制转换为 int，直接计算 float 上的误差更常见。
    # 为了严格遵循原函数逻辑，我们进行类型转换 (如果需要)
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
    # 由于原始实现是灰度图，这里直接计算 L1 距离作为示例
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    # Mean Absolute Error (MAE)
    mape_value = np.mean(np.abs(img1 - img2))
    return mape_value

def get_x_train_gray_np(dataset_train):
    """
    从 PyTorch Dataset 中提取所有图像，转换为 NumPy 灰度图 (0-255)。
    用于 MAPE 对比的基准数据。
    """
    print("[CVEA Utility] Preparing NumPy grayscale images for MAPE calculation...")
    
    # 使用 DataLoader 获取所有训练数据 (N, C, H, W)
    full_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    # 注意：这里获取的是经过 transform 的 Tensor (通常已标准化)
    images_tensor, _ = next(iter(full_loader))
    
    # 假设 transform 将数据标准化到 [0, 1]，我们需要反向操作
    
    # 1. 形状调整 (N, C, H, W) -> (N, H, W, C)
    images_tensor = images_tensor.permute(0, 2, 3, 1).cpu()
    
    # 2. 转换为灰度
    if images_tensor.size(-1) == 1:
        # 单通道 (MNIST/FMNIST)
        x_train_gray = images_tensor.squeeze(-1)
    else:
        # RGB (CIFAR)
        # 使用 rbg_to_grayscale_pt 函数（如果它在 attack_utils.py 中定义）
        x_train_gray = rbg_to_grayscale_pt(images_tensor) 
    
    # 3. 转换为 NumPy 并重映射到 [0, 255]
    # 假设图像在 [0, 1] 范围内 (通常是 ToTensor() 和 Normalize() 后的结果)
    x_train_gray_np = x_train_gray.numpy()
    x_train_gray_np = (x_train_gray_np * 255).astype(np.uint8)
    
    return x_train_gray_np

def prepare_cvea_stolen_data_pt(net_glob, dataset_train, args):
    # 如果 gama <= 0，则不进行攻击
    if getattr(args, 'gama', 0.0) <= 0:
        print("[CVEA Attack] Attack disabled (gama=0).")
        return None

    print(f"\n[CVEA Attack] Preparing stolen data with gama={args.gama}")

    # 获取目标权重总数
    num_target_params = 0
    for name, param in net_glob.named_parameters():
        # 攻击通常针对维度 > 1 的权重 (Conv/Linear weights)
        if param.dim() > 1:
            num_target_params += param.numel()

    if num_target_params == 0:
        print('Error: Model has no suitable parameters for CVEA attack.')
        return None

    print(f"[CVEA Attack] Target parameter count: {num_target_params}")

    # 获取原始训练数据 Tensor
    # 使用 DataLoader 来获取所有训练数据（如果内存允许）
    # 注意：这里的 DataLoader 会应用 dataset_train 中定义的 transforms。
    full_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    data_iter = iter(full_loader)
    images_tensor, _ = next(data_iter) # images_tensor shape: (N, C, H, W)
    images_tensor = images_tensor.to(args.device)

    # 转换为灰度（Channel Last for rbg_to_grayscale_pt）
    # CIFAR/MNIST 的形状调整
    if images_tensor.dim() == 4:
        # 假设 PyTorch 格式 (N, C, H, W)，转换为 (N, H, W, C) 以适应 rbg_to_grayscale_pt
        if images_tensor.size(1) in [1, 3]:
             images_tensor = images_tensor.permute(0, 2, 3, 1)

    # 转换为灰度 (N, H, W)
    # 如果是 MNIST (N, H, W, 1)， rbg_to_grayscale_pt 应该能处理
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

# 相关值编码攻击窃取方法
def cor_attack(model, d_m):
    # 1. 遍历模型的所有可训练参数并展平
    params = []
    for param in model.parameters():
        # 只处理维度大于 1 的权重（通常是 Conv2D 和 Linear 层的权重，不包括 bias）
        if param.dim() > 1:
            params.append(param.view(-1))
    
    # 如果模型没有可训练参数，则返回 0
    if not params:
        return torch.tensor(0.0, device=d_m.device)
        
    # 2. 拼接成一个大的向量
    p_flat = torch.cat(params, dim=0)
    
    # ******* 关键步骤：匹配长度 *******
    # 攻击依赖于权重和秘密数据的长度匹配，这里截断长的那个
    min_len = min(p_flat.size(0), d_m.size(0))
    p_flat = p_flat[:min_len]
    d_m = d_m[:min_len]
    # **********************************

    # 3. 计算中心化权重 p_m = params - mean(params)
    p_mean = torch.mean(p_flat)
    p_m = p_flat - p_mean
    
    # 4. 计算相关性函数
    # r_num = sum(p_m * d_m) (协方差)
    r_num = torch.sum(p_m * d_m)
    
    # r_den = sqrt(sum(p_m^2) * sum(d_m^2)) (模长乘积)
    r_den = torch.sqrt(torch.sum(p_m ** 2) * torch.sum(d_m ** 2))
    
    # r (皮尔逊相关系数)
    # 避免除以零
    epsilon = 1e-8 
    r = r_num / (r_den + epsilon)
    
    # loss = |r|
    loss = torch.abs(r)
    return loss


# 评估窃取到的数据与原始数据的相似性
def calculate_cor_mape(model, x_train):
    # 1. 遍历模型的所有可训练参数并展平
    params = []
    for param in model.parameters():
        # 只处理维度大于 1 的权重
        if param.dim() > 1:
            # 转换为 NumPy，并且展平，注意这里需要使用 .detach().cpu()
            params.append(param.detach().cpu().numpy().reshape(-1, 1))
            
    if not params:
        return 0.0

    # 2. 拼接成一个大的向量 (NumPy)
    params = np.concatenate(params, axis=0)
    
    # 3. 对权重向量进行归一化处理 [0, 1]
    params = normalize(params)
    
    # 4. 权重重映射到 [0, 255]
    params = (params * 255).astype(np.uint8)
    
    # 5. 重新组织成图片的格式
    # 假设 x_train 是 (N, H, W) 或 (N, H, W, 1) 的灰度图
    if x_train.ndim == 4 and x_train.shape[-1] == 1:
        x_train = x_train.squeeze(-1)
        
    num_pixel = int(np.prod(x_train.shape[1:]))
    num_image = int(len(params) / num_pixel)
    
    params = params[:num_image * num_pixel]
    # 假设 x_train 的形状是 (N, H, W)
    params = params.reshape(num_image, x_train.shape[1], x_train.shape[2])
    
    # 6. 计算 MAPE (或误差)
    mape = 0
    for i in range(num_image):
        # 确保输入 Image.fromarray 的是 (H, W) 形状的 np.uint8 数组
        img_i = params[i]
        
        err1 = cal_error(img_i, x_train[i])
        
        # 计算权重恢复图像的反色图像与训练图像之间的误差 err2
        img_inverted = np.asarray(ImageOps.invert(Image.fromarray(img_i)))
        err2 = cal_error(img_inverted, x_train[i])
        
        mape += min([err1, err2])
        
    return mape / num_image


# 恢复窃取到的数据
def recover_cor_stolen_data(model, x_train):
    # 1. 遍历模型的所有可训练参数并展平
    params = []
    for param in model.parameters():
        if param.dim() > 1:
            params.append(param.detach().cpu().numpy().reshape(-1, 1))
            
    if not params:
        print("Model has no suitable parameters to recover.")
        return np.array([])
        
    # 2. 拼接成一个大的向量 (NumPy)
    params = np.concatenate(params, axis=0)
    
    # 3. 归一化处理 [0, 1]
    params = normalize(params)
    
    # 4. 重新组织成图片的格式
    if x_train.ndim == 4 and x_train.shape[-1] == 1:
        x_train = x_train.squeeze(-1)
        
    total_pix = np.prod(x_train.shape[1:])
    number = int(params.shape[0] / total_pix)
    
    print("steal number:", number)
    
    params = params[0:number * total_pix]
    # 假设 x_train 的形状是 (N, H, W)
    params = params.reshape(number, x_train.shape[1], x_train.shape[2])
    
    # 5. 重映射到 [0, 255]
    params = (params * 255).astype(np.uint8)
    
    return params