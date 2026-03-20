import numpy as np
import os

# 1. 设置实验参数
dataset = "fmnist"
model = "cnn"
num_steal = 5
# 确保 gama 列表中的数值与你文件名中的字符串匹配
gammas = [0.0, 0.05, 0.2, 0.5, 1.0] 
target_rounds = [10, 50, 100, 150, 200]

base_path = "./save/results/"

print(f"{'='*65}")
print(f" 数据集: {dataset.upper()} | 关键轮次性能指标提取报告")
print(f"{'='*65}")

for gama in gammas:
    # 构建文件名
    # 注意：如果你的文件名里 0 写作 Gama[0]，而 0.5 写作 Gama[0.5]，此处会自动处理
    file_acc = f"{dataset}_{model}_200_C[0.1]_iid[1]_E[10]_B[10]_Gama[{gama}]_numSteal[{num_steal}]_acc.npy"
    file_mape = f"{dataset}_{model}_200_C[0.1]_iid[1]_E[10]_B[10]_Gama[{gama}]_numSteal[{num_steal}]_mape.npy"
    
    print(f"\n[ 实验组: Gamma = {gama} ]")
    
    # 根据 gamma 是否为 0 调整打印表头
    if gama == 0:
        print(f"{'Round':<10} | {'Accuracy (%)':<18} | {'备注':<15}")
        print("-" * 50)
    else:
        print(f"{'Round':<10} | {'Accuracy (%)':<18} | {'MAPE Value':<15}")
        print("-" * 50)
    
    try:
        # 加载准确率数据 (所有组都需要)
        data_acc = np.load(os.path.join(base_path, file_acc))
        
        # 仅当 gamma != 0 时加载 MAPE 数据
        data_mape = None
        if gama != 0:
            data_mape = np.load(os.path.join(base_path, file_mape))
        
        for r in target_rounds:
            idx = r - 1 # 索引偏移
            
            if idx < len(data_acc):
                acc_val = data_acc[idx] * 100
                
                if gama == 0:
                    # Gamma 为 0 时的输出格式
                    print(f"{r:<10} | {acc_val:>16.2f}% | {'基准(无攻击)':<15}")
                else:
                    # 其他攻击组的输出格式
                    mape_val = data_mape[idx]
                    print(f"{r:<10} | {acc_val:>16.2f}% | {mape_val:>15.4f}")
            else:
                print(f"{r:<10} | {'数据未达此轮次':>16} | {'-':^15}")
                
    except FileNotFoundError as e:
        print(f"跳过：未找到对应文件 (Gama={gama})")

print(f"\n{'='*65}")
