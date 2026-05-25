# ResNet18 — 经典卷积神经网络实现

使用 PyTorch 实现的 ResNet18 残差卷积神经网络，在 CIFAR-10 数据集上进行训练和测试。

## 项目简介

ResNet（Residual Network）是微软研究院何恺明等人于 2015 年提出的深度 CNN 架构，在 ILSVRC-2015 竞赛中取得分类任务冠军。其核心创新是引入了**残差连接（Skip Connection）**，将输入直接跨层传递到输出端，与卷积结果相加后再激活。这种设计有效解决了深层网络的退化问题（Degradation Problem），使得训练上百层甚至上千层的网络成为可能。本项目使用 CIFAR-10（物体分类）数据集来训练和评估 ResNet18 模型。

### 残差模块（Residual Block）

每个残差模块包含两条路径：

| 路径 | 结构 | 作用 |
|------|------|------|
| 主路径 | 3×3 卷积 → BN → ReLU → 3×3 卷积 → BN | 学习残差映射 F(x) |
| 跳跃连接 | 恒等映射（输入输出维度相同时） | 保留原始输入 x |
| 跳跃连接 | 1×1 卷积（输入输出维度不同时） | 调整维度使 x 与 F(x) 匹配 |
| 合并 | F(x) + x → ReLU | 残差融合 |

### 网络结构

ResNet18 共包含 17 个卷积层和 1 个全连接层，总计 18 个带权重的层。

| 层 | 类型 | 输出尺寸 | 参数 |
|---|---|---|---|
| C1 | 卷积层 + BN + ReLU | 112×112 | in=3, out=64, kernel=7×7, stride=2, padding=3 |
| P1 | 最大池化 | 56×56 | kernel=3×3, stride=2, padding=1 |
| R1 | 残差模块 ×2 | 56×56 | in=64, out=64, stride=1 |
| R2 | 残差模块（下采样） | 28×28 | in=64, out=128, stride=2 (1×1 shortcut) |
| R3 | 残差模块 | 28×28 | in=128, out=128, stride=1 |
| R4 | 残差模块（下采样） | 14×14 | in=128, out=256, stride=2 (1×1 shortcut) |
| R5 | 残差模块 | 14×14 | in=256, out=256, stride=1 |
| R6 | 残差模块（下采样） | 7×7 | in=256, out=512, stride=2 (1×1 shortcut) |
| R7 | 残差模块 | 7×7 | in=512, out=512, stride=1 |
| AvgPool | 自适应平均池化 | 1×1 | output=(1, 1) |
| FC | 全连接 | 10 | 512 → 10 |

激活函数使用 ReLU，在每个卷积层后使用 Batch Normalization 加速收敛。输入图像尺寸为 224×224 彩色图。

## 项目结构

```
Classic-CNN-PyTorch/
├── requirements.txt  # 项目依赖（上级目录）
└── ResNet18/
    ├── model.py          # ResNet18 网络结构定义（含 Residual 残差模块）
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

> 注意：CIFAR-10 原生图像尺寸为 32×32，通过 `transforms.Resize` 放大至 224×224 以适配 ResNet18 输入层，放大过程会引入信息损失。

## 参考

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
