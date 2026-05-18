# AlexNet — 经典卷积神经网络实现

使用 PyTorch 实现的 AlexNet 卷积神经网络，在 CIFAR-10 数据集上进行训练和测试。

## 项目简介

AlexNet 是 Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 于 2012 年提出的深度 CNN 架构，在 ImageNet LSVRC-2012 竞赛中大幅领先传统方法夺冠，标志着深度学习时代的开启。本项目使用 CIFAR-10（物体分类）数据集来训练和评估 AlexNet 模型。

### 网络结构

| 层 | 类型 | 参数 |
|---|---|---|
| C1 | 卷积层 + ReLU | in=3, out=96, kernel=11×11, stride=4 |
| P2 | 最大池化 | kernel=3×3, stride=2 |
| C3 | 卷积层 + ReLU | in=96, out=256, kernel=5×5, padding=2 |
| P4 | 最大池化 | kernel=3×3, stride=2 |
| C5 | 卷积层 + ReLU | in=256, out=384, kernel=3×3, padding=1 |
| C6 | 卷积层 + ReLU | in=384, out=384, kernel=3×3, padding=1 |
| C7 | 卷积层 + ReLU | in=384, out=256, kernel=3×3, padding=1 |
| P8 | 最大池化 | kernel=3×3, stride=2 |
| F1 | 全连接 + ReLU + Dropout(0.5) | 9216 → 4096 |
| F2 | 全连接 + ReLU + Dropout(0.5) | 4096 → 4096 |
| F3 | 全连接 | 4096 → 10 |

激活函数使用 ReLU，输入图像尺寸为 227×227 彩色图。

## 项目结构

```
Classic-CNN-PyTorch/
├── requirements.txt  # 项目依赖（上级目录）
└── AlexNet/
    ├── model.py          # AlexNet 网络结构定义
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
- 训练 20 个 epoch，使用 Adam 优化器
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

展示 CIFAR-10 训练集的一个 batch 图像（Resize 至 227×227）。

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

> 注意：CIFAR-10 原生图像尺寸为 32×32，通过 `transforms.Resize` 放大至 227×227 以适配 AlexNet 输入层，放大过程会引入信息损失。

## 参考

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
