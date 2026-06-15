#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFashion_Enhanced(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Enhanced, self).__init__()
        
        # --- 核心修改 1：输入通道从 3 改为 1 (FashionMNIST 是黑白图) ---
        # 保持 bias=False 以利于梯度泄露攻击 (DLG/iDLG)
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 80) 
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) 

        # 第二层：保持 128 通道
        self.conv2 = nn.Conv2d(80, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 128)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        
        # --- 核心修改 2：重新计算特征图尺寸 ---
        # 输入: 1x28x28
        # Conv1(5x5, p2) -> 80x28x28 -> Pool1(2x2) -> 80x14x14
        # Conv2(3x3, p1) -> 128x14x14 -> Pool2(2x2) -> 128x7x7
        # 计算: 128 * 7 * 7 = 6272
        self.flat_features = 6272 
        
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout = nn.Dropout(0.3) 
        self.act3 = nn.ReLU() 
        
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        # 注意：FashionMNIST 只有 1 个通道 [batch, 1, 28, 28]
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        x = self.pool2(self.act2(self.gn2(self.conv2(x))))
        
        # 展平层
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(self.act3(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar_new(nn.Module):
    def __init__(self, args):
        super(CNNCifar_new, self).__init__()
        
        # 1. 第一个卷积层
        # 输入: 24x24x3 -> 输出: 24x24x64
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=64, 
                               kernel_size=5, 
                               padding=2)
        # LayerNorm 对于卷积层，需要指定 [C, H, W]
        self.ln1 = nn.LayerNorm([64, 24, 24]) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 输出: 12x12x64

        # 2. 第二个卷积层
        # 输入: 12x12x64 -> 输出: 12x12x64
        self.conv2 = nn.Conv2d(in_channels=64, 
                               out_channels=64, 
                               kernel_size=5, 
                               padding=2)
        # LayerNorm: 此时特征图尺寸已减半
        self.ln2 = nn.LayerNorm([64, 12, 12])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 输出: 6x6x64
        
        flat_features = 6 * 6 * 64
        
        # 3. 第一个全连接层
        self.fc1 = nn.Linear(flat_features, 384) 
        self.ln_fc1 = nn.LayerNorm(384) # 全连接层 LN 只需指定特征数
        
        # 4. 第二个全连接层
        self.fc2 = nn.Linear(384, 192)
        self.ln_fc2 = nn.LayerNorm(192)
        
        # 5. 线性变换层
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        # Conv 1 -> LN -> ReLU -> Pool
        # 注意：LayerNorm 放在 ReLU 之前是更常用的做法
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv 2 -> LN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1) 
        
        # FC 1 -> LN -> ReLU
        x = self.fc1(x)
        x = self.ln_fc1(x)
        x = F.relu(x)
        
        # FC 2 -> LN -> ReLU
        x = self.fc2(x)
        x = self.ln_fc2(x)
        x = F.relu(x)
        
        # FC 3 (Logits)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

class CNNCifar_Enhanced(nn.Module):
    def __init__(self, args):
        super(CNNCifar_Enhanced, self).__init__()
        
        # 第一层：保持 80 通道，5x5 卷积，无 Bias 以利于攻击
        self.conv1 = nn.Conv2d(3, 80, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 80) 
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) 

        # 第二层：增加到 128 通道提升表达能力
        self.conv2 = nn.Conv2d(80, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 128)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        
        # --- 修复逻辑 ---
        # 根据报错信息，这里的输入应该是 4608
        # 计算公式：通道数(128) * 特征图宽(6) * 特征图高(6) = 4608
        self.flat_features = 4608 
        
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout = nn.Dropout(0.3) 
        self.act3 = nn.ReLU() 
        
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        x = self.pool2(self.act2(self.gn2(self.conv2(x))))
        
        # 展平层
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(self.act3(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifar_Enhanced_V2(nn.Module):
    def __init__(self, args):
        super(CNNCifar_Enhanced_V2, self).__init__()
        
        # --- 攻击保留层 (第一层) ---
        # 保持 80 通道和 5x5 卷积核，不加 Bias。
        # 这确保了梯度中包含足够的像素映射信息，维持原有的攻击效果。
        self.conv1 = nn.Conv2d(3, 80, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 80) 
        self.act1 = nn.GELU() # 换成平滑的 GELU
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 32->16

        # --- 增强提取层 (第二层 & 第三层) ---
        self.conv2 = nn.Conv2d(80, 192, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(12, 192)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 256)
        self.act3 = nn.GELU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 16->8
        
        # --- 分类头部 ---
        # 使用自适应池化将特征图固定为 4x4，无论输入尺寸如何都不会报错
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat_features = 256 * 4 * 4 # = 4096
        
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout = nn.Dropout(0.3) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        # 卷积阶段
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        x = self.act2(self.gn2(self.conv2(x)))
        x = self.pool2(self.act3(self.gn3(self.conv3(x))))
        
        # 降维与展平
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        
        # 全连接阶段
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class ResidualBlock(nn.Module):
    """一个轻量级的残差块，用于提升分类准确率"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x):
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x) # 跳跃连接
        out = self.act(out)
        return out

class CNNCifar_Enhanced_V3(nn.Module):
    def __init__(self, args):
        super(CNNCifar_Enhanced_V3, self).__init__()
        
        # --- 1. 攻击层 (严禁修改，确保梯度泄露精度) ---
        self.conv1 = nn.Conv2d(3, 80, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 80)
        self.act1 = nn.SiLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 32 -> 16

        # --- 2. 增强骨干层 (使用残差块提升准确率) ---
        self.res1 = ResidualBlock(80, 160)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 16 -> 8
        
        self.res2 = ResidualBlock(160, 320)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # 8 -> 4
        
        # --- 3. 高性能分类头 ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, args.num_classes)
        )

    def forward(self, x):
        # 第一层保持原有逻辑
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        
        # 进入残差增强阶段
        x = self.pool2(self.res1(x))
        x = self.pool3(self.res2(x))
        
        # 展平分类
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class BasicBlock(nn.Module):
    """标准残差块，用于增加深度和参数量"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, planes) # 联邦学习中推荐使用 GroupNorm 代替 BatchNorm
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, self.expansion * planes)
            )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out

class CNNCifar_ResNet_V1(nn.Module):
    def __init__(self, args):
        super(CNNCifar_ResNet_V1, self).__init__()
        
        # --- 1. 攻击层 (严格保留原始结构，确保 CVEA 梯度泄露逻辑一致) ---
        # 这里的 80 通道提供了足够的梯度信息维度
        self.conv1 = nn.Conv2d(3, 80, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 80)
        self.act1 = nn.SiLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        # --- 2. 扩展骨干层 (模仿 ResNet 架构大幅度增加参数) ---
        # 增加通道数：80 -> 160 -> 320 -> 640
        self.layer1 = self._make_layer(BasicBlock, 80, 160, num_blocks=3, stride=2)  # 16x16 -> 8x8
        self.layer2 = self._make_layer(BasicBlock, 160, 320, num_blocks=4, stride=2) # 8x8 -> 4x4
        self.layer3 = self._make_layer(BasicBlock, 320, 640, num_blocks=3, stride=2) # 4x4 -> 2x2
        
        # --- 3. 高性能全连接头 ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(640, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, args.num_classes)
        )

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(in_planes, planes, s))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始攻击特征提取
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        
        # 深度残差特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 分类
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_ResNet18(nn.Module):
    def __init__(self, args):
        super(CNNFashion_ResNet18, self).__init__()
        self.in_planes = 160

        # Fashion-MNIST is 1x28x28. The 160-channel 5x5 stem keeps enough
        # early parameters for the default 5 * 28 * 28 CVEA payload.
        self.conv1 = nn.Conv2d(1, 160, kernel_size=5, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(8, 160)
        self.act1 = nn.SiLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        self.layer1 = self._make_layer(160, num_blocks=2, stride=1)  # 14x14 -> 14x14
        self.layer2 = self._make_layer(320, num_blocks=2, stride=2)  # 14x14 -> 7x7
        self.layer3 = self._make_layer(640, num_blocks=2, stride=2)  # 7x7 -> 4x4
        self.layer4 = self._make_layer(640, num_blocks=2, stride=2)  # 4x4 -> 2x2

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(640, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, args.num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.act1(self.gn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


# ==================== ResNet18 for CIFAR (24x24) ====================

class ResNet18Cifar(nn.Module):
    """
    标准 ResNet18 适配 CIFAR 小图 (24x24 或 32x32)
    - 复用已有的 BasicBlock（2 层 3x3 conv，GroupNorm + SiLU）
    - 第一层使用 3x3 卷积（而非 ImageNet 的 7x7），不进行初始 MaxPool
    - 层配置: [2, 2, 2, 2]（标准 ResNet18）
    - 通道: 64 → 128 → 256 → 512（~11.2M 参数）
    """
    def __init__(self, args, num_blocks=[2, 2, 2, 2]):
        super(ResNet18Cifar, self).__init__()
        self.in_planes = 64

        # --- 初始卷积层 (适配小图，不做过度降采样) ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 64)
        self.act1 = nn.SiLU()
        # 注意：不使用 MaxPool，保留空间分辨率（24x24 -> 24x24）

        # --- 4 个残差阶段 (复用 BasicBlock) ---
        self.layer1 = self._make_layer(64,  num_blocks[0], stride=1)  # 24x24 -> 24x24, 通道 64
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)  # 24x24 -> 12x12, 通道 128
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)  # 12x12 -> 6x6,   通道 256
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)  # 6x6   -> 3x3,   通道 512

        # --- 分类头 ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming 初始化，加速收敛"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        x = self.act1(self.gn1(self.conv1(x)))

        # 4 个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化 + 分类
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(8, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.use_shortcut_conv = stride != 1 or in_planes != planes
        self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False) \
            if self.use_shortcut_conv else nn.Identity()

    def forward(self, x):
        out = F.silu(self.gn1(x))
        shortcut = self.shortcut(out if self.use_shortcut_conv else x)
        out = self.conv1(out)
        out = self.dropout(F.silu(self.gn2(out)))
        out = self.conv2(out)
        return out + shortcut


class WideResNetCifar(nn.Module):
    """
    WideResNet for CIFAR FedAvg baselines.

    depth=28 and widen_factor=2/4 are practical next steps after ResNet18:
    stronger than the current compact CIFAR models while still cheaper than ViT.
    """
    def __init__(self, args, depth=28, widen_factor=2, dropout_rate=0.0):
        super(WideResNetCifar, self).__init__()
        assert (depth - 4) % 6 == 0, 'WideResNet depth should be 6n + 4'
        num_blocks = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(channels[1], num_blocks, dropout_rate, stride=1)
        self.layer2 = self._make_layer(channels[2], num_blocks, dropout_rate, stride=2)
        self.layer3 = self._make_layer(channels[3], num_blocks, dropout_rate, stride=2)
        self.gn = nn.GroupNorm(8, channels[3])
        self.fc = nn.Linear(channels[3], args.num_classes)
        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, dropout_rate, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.silu(self.gn(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
