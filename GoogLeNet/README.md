# GoogLeNet — 经典卷积神经网络实现

使用 PyTorch 实现的 GoogLeNet（Inception v1）卷积神经网络，在 CIFAR-10 数据集上进行训练和测试。

## 项目简介

GoogLeNet 是 Google 团队（Szegedy 等人）于 2014 年提出的深度 CNN 架构，在 ILSVRC-2014 竞赛中取得了分类任务冠军。其核心创新是引入了 Inception 模块，通过在同一层并行使用 1×1、3×3、5×5 卷积和 3×3 最大池化，并将结果在通道维度上拼接，从而在不同尺度上提取特征。同时，大量 1×1 卷积的降维操作有效控制了参数量和计算成本。本项目使用 CIFAR-10（物体分类）数据集来训练和评估 GoogLeNet 模型。

### Inception 模块

Inception 模块包含四条并行的路径，最后在通道维度拼接：

| 路径 | 结构 | 作用 |
|------|------|------|
| 路径1 | 1×1 卷积 | 跨通道信息交互与降维 |
| 路径2 | 1×1 卷积 → 3×3 卷积 | 先降维再进行中等尺度特征提取 |
| 路径3 | 1×1 卷积 → 5×5 卷积 | 先降维再进行大尺度特征提取 |
| 路径4 | 3×3 最大池化 → 1×1 卷积 | 池化后降维，保留空间信息 |

### 网络结构

| 层 | 类型 | 输入通道 | 输出通道 / 配置 |
|---|---|---|---|
| Block 1 | Conv 7×7 (s=2) + ReLU | 3 | 64 |
| | MaxPool 3×3 (s=2) | — | (112→56) |
| Block 2 | Conv 1×1 + ReLU | 64 | 64 |
| | Conv 3×3 + ReLU | 64 | 192 |
| | MaxPool 3×3 (s=2) | — | (56→28) |
| Block 3 | Inception (3a) | 192 | 64 + 128 + 32 + 32 = 256 |
| | Inception (3b) | 256 | 128 + 192 + 96 + 64 = 480 |
| | MaxPool 3×3 (s=2) | — | (28→14) |
| Block 4 | Inception (4a) | 480 | 192 + 208 + 48 + 64 = 512 |
| | Inception (4b) | 512 | 160 + 224 + 64 + 64 = 512 |
| | Inception (4c) | 512 | 128 + 256 + 64 + 64 = 512 |
| | Inception (4d) | 512 | 112 + 288 + 64 + 64 = 528 |
| | Inception (4e) | 528 | 256 + 320 + 128 + 128 = 832 |
| | MaxPool 3×3 (s=2) | — | (14→7) |
| Block 5 | Inception (5a) | 832 | 256 + 320 + 128 + 128 = 832 |
| | Inception (5b) | 832 | 384 + 384 + 128 + 128 = 1024 |
| 输出层 | AdaptiveAvgPool2d (1×1) | — | 1024 |
| | Flatten | — | 1024 |
| | Dropout (0.4) | — | — |
| | Linear | 1024 | 10 |

激活函数使用 ReLU，权重初始化使用 Kaiming 正态分布初始化。输入图像尺寸为 224×224 彩色图。整个网络共包含 9 个 Inception 模块，参数量仅约 600 万。

## 项目结构

```
Classic-CNN-PyTorch/
├── requirements.txt  # 项目依赖（上级目录）
└── GoogLeNet/
    ├── model.py          # GoogLeNet 网络结构定义（含 Inception 模块）
    ├── model_train.py    # 训练脚本（含训练/验证曲线绘制）
    ├── model_test.py     # 测试脚本（批量测试 & 单张预测）
    ├── get_dataset.py    # 数据集加载与可视化
    ├── best_model.pth    # 训练保存的最优模型权重
    ├── README.md         # 本文件
    └── data/             # CIFAR-10 数据集（自动下载）
```

## 环境依赖

- torch==1.10.1
- torchsummary, numpy, pandas, matplotlib

安装依赖：

```bash
pip install -r ../requirements.txt
```

## 使用方法

### 训练模型

```bash
python model_train.py
```

训练过程会自动：
- 下载 CIFAR-10 数据集到 `./data/` 目录
- 按 8:2 划分训练集和验证集
- 训练 20 个 epoch，使用 Adam 优化器（lr=0.001）
- 保存最优模型权重到 `best_model.pth`
- 绘制训练/验证的 Loss 和 Accuracy 曲线

### 测试模型

```bash
python model_test.py
```

支持两种测试模式：
- `test_model_process`：批量测试，输出整体准确率
- `test_model_process_v2`：逐张预测，打印每张图片的预测结果与真实标签

### 可视化数据集

```bash
python get_dataset.py
```

展示 CIFAR-10 训练集的一个 batch 图像（Resize 至 224×224）。

## 数据集

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 包含 10 类物体的 32×32 彩色图像：

| 标签 | 类别 |
|------|------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

> 注意：CIFAR-10 原生图像尺寸为 32×32，通过 `transforms.Resize` 放大至 224×224 以适配 GoogLeNet 输入层，放大过程会引入信息损失。

## 参考

- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going Deeper with Convolutions.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
