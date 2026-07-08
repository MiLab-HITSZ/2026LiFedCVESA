import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_x_train_gray_np(x_train_gray_np, num_to_plot=10, title="Original Stolen Images (Client 0, 1, ...)", rows=2, save_path="original_stolen_images.png", labels=None):
    """
    绘制原始灰度图像数组 x_train_gray_np 的内容。

    Args:
        x_train_gray_np (np.ndarray): 原始灰度图片数组 (N, H, W)。
        num_to_plot (int): 要绘制的图片数量 (默认为 10)。
        title (str): 图表的总标题。
        rows (int): 子图的行数。
    """
    N = x_train_gray_np.shape[0]
    if N == 0:
        print("Error: The x_train_gray_np array is empty.")
        return

    # 确定实际绘制的数量和子图布局
    actual_plot_count = min(num_to_plot, N)
    
    # 计算列数
    if actual_plot_count == 0:
        print("No images to plot.")
        return
        
    cols = int(np.ceil(actual_plot_count / rows))
    
    # 将 NumPy 数组转换为 uint8 类型并限制范围到 [0, 255]
    # 假设 x_train_gray_np 已经是 [0, 255] 范围
    original_images = np.clip(x_train_gray_np, 0, 255).astype(np.uint8)
    
    # 创建子图布局
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(2.5 * cols, 2.5 * rows))
    
    # 调整总标题
    fig.suptitle(title, fontsize=14, y=1.02)
    
    # 展平 axes 数组，方便迭代
    axes = axes.flatten() if rows > 1 or cols > 1 else np.array([axes])
    
    for i in range(actual_plot_count):
        ax = axes[i]
        img_orig = original_images[i]
        
        # 使用灰度色图，并明确设置 Vmin/Vmax 确保颜色映射正确
        ax.imshow(img_orig, cmap='gray', norm=Normalize(vmin=0, vmax=255))
        
        # 标签显示其对应的客户端索引
        label = labels[i] if labels is not None and i < len(labels) else f"Client {i}"
        ax.set_title(label, fontsize=10)
        ax.axis('off')
        
    # 隐藏多余的子图
    for j in range(actual_plot_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 1]) # 自动调整布局
    plt.savefig(save_path)


import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_stolen_data_dm(stolen_data_dm_tensor, H, W, num_images, num_to_plot=10, title="Visualizing Stolen Data Vector (d_m)", save_path="stolen_data_dm_visualization.png", labels=None):
    """
    仅使用 stolen_data_dm 向量及其形状信息来重塑和绘制图片。
    
    Args:
        stolen_data_dm_tensor (torch.Tensor): 展平、中心化后的窃取向量 (d_m)。
        H (int): 单张图片的高度。
        W (int): 单张图片的宽度。
        num_images (int): 目标图片总数（通常是客户端总数，例如 100）。
        num_to_plot (int): 要绘制的图片数量。
        title (str): 图表的总标题。
    """
    
    # 1. 移到 CPU 并转换为 NumPy
    dm_np = stolen_data_dm_tensor.cpu().numpy()
    
    pix_per_image = H * W
    target_len = num_images * pix_per_image
    
    # 2. 截断/确保长度匹配
    if dm_np.size < target_len:
        print(f"[Plot Warning] Vector size ({dm_np.size}) is less than required ({target_len}). Plotting available data.")
        dm_np = dm_np[0 : (dm_np.size // pix_per_image) * pix_per_image]
        N_actual = dm_np.size // pix_per_image
    else:
        dm_np = dm_np[:target_len]
        N_actual = num_images

    # 3. 重塑为图片格式 (N_actual, H, W)
    dm_images = dm_np.reshape(N_actual, H, W)
    actual_plot_count = min(num_to_plot, N_actual)
    if actual_plot_count == 0:
        print("No images to plot.")
        return
    
    # 4. 归一化/缩放以适配可视化 (因为它是中心化后的数据，值可能在负数范围)
    # 我们将其线性映射到 [0, 1] 范围进行可视化，以展现其相对强度。
    
    # 计算当前 dm_images 的全局最小值和最大值
    min_val = dm_images.min()
    max_val = dm_images.max()
    
    if max_val == min_val:
        # 避免除以零，如果所有值都相等，则显示为中性灰色
        visual_images = np.zeros_like(dm_images) + 0.5 
    else:
        # 线性缩放至 [0, 1] 范围，以便 Matplotlib 正确显示灰度
        visual_images = (dm_images - min_val) / (max_val - min_val)

    # 5. 绘图
    rows = 2
    cols = int(np.ceil(actual_plot_count / rows))
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(2.5 * cols, 2.5 * rows))
    fig.suptitle(title, fontsize=14, y=1.02)
    
    axes = axes.flatten()
    
    for i in range(actual_plot_count):
        ax = axes[i]
        img = visual_images[i]
        
        # 此时数据在 [0, 1]，使用 'gray' 色图
        ax.imshow(img, cmap='gray') 
        label = labels[i] if labels is not None and i < len(labels) else f"Client {i}"
        ax.set_title(label, fontsize=10)
        ax.axis('off')
        
    for j in range(actual_plot_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path)
