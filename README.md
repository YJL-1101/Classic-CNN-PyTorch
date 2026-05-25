# Classic-CNN-PyTorch

使用 PyTorch 实现的经典卷积神经网络合集，涵盖深度学习发展史上 5 个里程碑式的 CNN 架构。

## 模型列表

| 模型 | 论文年份 | 核心创新 | 数据集 | 输入尺寸 |
|------|----------|----------|--------|----------|
| [LeNet](LeNet/) | 1998 | 奠定 CNN 基础架构（卷积→池化→全连接） | FashionMNIST | 28×28 灰度图 |
| [AlexNet](AlexNet/) | 2012 | 引入 ReLU、Dropout，GPU 训练，深度学习时代开端 | CIFAR-10 | 227×227 RGB |
| [VGG16](VGG16/) | 2014 | 使用小卷积核（3×3）堆叠加深网络 | CIFAR-10 | 224×224 RGB |
| [GoogLeNet](GoogLeNet/) | 2014 | Inception 模块：多尺度并行卷积 + 1×1 降维 | CIFAR-10 | 224×224 RGB |
| [ResNet18](ResNet18/) | 2015 | 残差连接（Skip Connection），解决深层网络退化问题 | CIFAR-10 | 224×224 RGB |

## 项目结构

```
Classic-CNN-PyTorch/
├── requirements.txt          # 项目依赖
├── README.md                 # 本文件
├── LeNet/                    # LeNet-5 实现
│   ├── model.py              # 网络结构定义
│   ├── model_train.py        # 训练脚本
│   ├── model_test.py         # 测试脚本
│   ├── get_dataset.py        # 数据集加载与可视化
│   └── README.md             # LeNet 详细说明
├── AlexNet/                  # AlexNet 实现
│   └── ...                   # 同上结构
├── VGG16/                    # VGG16 实现
│   └── ...                   # 同上结构
├── GoogLeNet/                # GoogLeNet (Inception v1) 实现
│   └── ...                   # 同上结构
└── ResNet18/                 # ResNet18 实现
    └── ...                   # 同上结构
```

每个子目录均包含独立的 `README.md`，详细介绍了对应模型的网络结构、训练方法和使用说明。

## 环境依赖

- Python 3.7+
- PyTorch 1.10.1
- torchsummary 1.5.1
- NumPy 1.23.2
- Matplotlib 3.5.0
- Pandas 1.3.4

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

每个模型均可独立运行。以 LeNet 为例：

```bash
# 进入模型目录
cd LeNet

# 查看数据集样本
python get_dataset.py

# 训练模型
python model_train.py

# 测试模型
python model_test.py
```

其他模型的使用方式完全一致，只需进入对应的模型目录即可。

## 训练流程

所有模型的训练脚本遵循统一的流程：

1. **数据加载** — 自动下载对应数据集，按 8:2 划分训练集和验证集
2. **模型训练** — 使用 Adam 优化器和交叉熵损失函数
3. **模型保存** — 自动保存在验证集上准确率最高的模型权重
4. **可视化** — 训练完成后自动绘制 Loss 和 Accuracy 曲线

## 数据集

| 数据集 | 类别数 | 图像尺寸 | 训练集 | 测试集 | 使用模型 |
|--------|--------|----------|--------|--------|----------|
| FashionMNIST | 10 | 28×28 灰度 | 60,000 | 10,000 | LeNet |
| CIFAR-10 | 10 | 32×32 RGB | 50,000 | 10,000 | AlexNet / VGG16 / GoogLeNet / ResNet18 |

## 参考

- [LeNet-5](http://yann.lecun.com/exdb/lenet/) — Yann LeCun et al., 1998
- [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — Krizhevsky et al., 2012
- [VGGNet](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, 2014
- [GoogLeNet](https://arxiv.org/abs/1409.4842) — Szegedy et al., 2014
- [ResNet](https://arxiv.org/abs/1512.03385) — He et al., 2015
