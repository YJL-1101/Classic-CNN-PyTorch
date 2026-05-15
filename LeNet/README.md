# LeNet — 经典卷积神经网络实现

使用 PyTorch 实现的 LeNet-5 卷积神经网络，在 FashionMNIST 数据集上进行训练和测试。

## 项目简介

LeNet-5 是 Yann LeCun 于 1998 年提出的经典 CNN 架构，最初用于手写数字识别（MNIST）。本项目使用 FashionMNIST（服饰分类）数据集来训练和评估 LeNet 模型。

### 网络结构

| 层 | 类型 | 参数 |
|---|---|---|
| C1 | 卷积层 | in=1, out=6, kernel=5×5, padding=2 |
| S2 | 平均池化 | kernel=2×2, stride=2 |
| C3 | 卷积层 | in=6, out=16, kernel=5×5 |
| S4 | 平均池化 | kernel=2×2, stride=2 |
| F5 | 全连接 | 400 → 120 |
| F6 | 全连接 | 120 → 84 |
| F7 | 全连接 | 84 → 10 |

激活函数使用 Sigmoid，输入图像尺寸为 28×28 灰度图。

## 项目结构

```
Classic-CNN-PyTorch/
├── requirements.txt  # 项目依赖（上级目录）
└── LeNet/
    ├── model.py          # LeNet 网络结构定义ss
    ├── model_train.py    # 训练脚本（含训练/验证曲线绘制）
    ├── model_test.py     # 测试脚本（批量测试 & 单张预测）
    ├── get_dataset.py    # 数据集加载与可视化
    ├── best_model.pth    # 训练保存的最优模型权重
    ├── README.md         # 本文件
    └── data/             # FashionMNIST 数据集（自动下载）
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
- 下载 FashionMNIST 数据集到 `./data/` 目录
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

展示 FashionMNIST 训练集的一个 batch 图像。

## 数据集

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) — 包含 10 类服饰的 28×28 灰度图像：

| 标签 | 类别 |
|------|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 参考

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
- [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)