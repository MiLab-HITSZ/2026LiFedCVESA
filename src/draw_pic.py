base_path = "./save/results/" # 如果文件在当前文件夹，保持不变

import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设置实验参数（请确保与你的文件名一致）
dataset = "mnist"
model = "mlp"
num_steal = 5
gammas = [0.0, 0.05, 0.2, 0.5, 1.0]  # 你拥有的 Gama 列表
base_path = "./save/results/"  # npy文件存放路径
MAPE_START_ROUND = 51

# 设置学术风格
for style_name in ('seaborn-v0_8-muted', 'seaborn-muted'):
    try:
        plt.style.use(style_name)
        break
    except OSError:
        continue
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def plot_metric(metric_type, title, ylabel, filename):
    """
    通用绘图函数
    metric_type: "acc" 或 "mape"
    """
    plt.figure(figsize=(10, 6), dpi=100)
    
    for gama in gammas:
        # 匹配你的文件名格式
        file_name = f"{dataset}_{model}_200_C[0.1]_iid[1]_E[10]_B[10]_Gama[{gama}]_numSteal[{num_steal}]_{metric_type}.npy"
        file_path = os.path.join(base_path, file_name)
        
        try:
            data = np.load(file_path)
            # 绘图线段标注
            label_text = f"$\gamma = {gama}$"
            if metric_type == "mape":
                rounds = np.arange(MAPE_START_ROUND, MAPE_START_ROUND + len(data))
                plt.plot(rounds, data, label=label_text, linewidth=2)
            else:
                plt.plot(data, label=label_text, linewidth=2)
        except FileNotFoundError:
            print(f"跳过：未找到 {file_name}")

    plt.title(title, fontsize=15, pad=15)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', frameon=True, fontsize=10)
    
    # 根据指标类型调整纵轴范围
    if metric_type == "acc":
        plt.ylim(0, 1.05) # 准确率最高 100%
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"已保存图表至: {filename}")

# 2. 执行绘图
# 绘制准确率曲线图
plot_metric("acc", 
            title=f"Global Model Accuracy on {dataset.upper()}", 
            ylabel="Accuracy (0-1)", 
            filename=f"{dataset}_accuracy_trend.png")

# 绘制 MAPE 曲线图
plot_metric("mape", 
            title=f"Reconstruction Error (MAPE) on {dataset.upper()}", 
            ylabel="MAPE Value", 
            filename=f"{dataset}_mape_trend.png")
