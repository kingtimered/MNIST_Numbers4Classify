# MNIST 手写数字 CNN 分类

基于卷积神经网络 (CNN) 的 MNIST 手写数字识别项目，包含数据准备、模型训练、评估可视化全流程。

**注意目前未上传数据集 MNIST 文件夹，需要运行 mnist_save.py 下载**

## 项目来源

- **数据集**：[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 张 28×28 灰度手写数字图像，10 分类
- **参考代码**：[zyf-xtu/DL/classification/zyf_mnist](https://github.com/zyf-xtu/DL/tree/master/classification/zyf_mnist) — 数据下载与整理的 `mnist_save.py` 参考了该仓库
- **参考文章**：[知乎 - zhangyunfei](https://zhuanlan.zhihu.com/p/503280155)

## 项目结构

```
MNIST_Numbers/
├── mnist_save.py          # 数据集下载与图像保存脚本
├── cnn_train.py           # CNN 模型定义、训练、评估脚本
├── generate_report.py     # HTML 报告生成脚本
├── README.md              # 项目说明文档
├── mnist/                 # 数据集目录
│   ├── MNIST/raw/         # 原始 .gz 压缩文件 (torchvision 下载)
│   └── train/             # 训练集 JPG 图片 (按 0-9 分类)
│   └── test/              # 测试集 JPG 图片 (按 0-9 分类)
└── results/               # 输出结果
    ├── mnist_cnn.pth      # 训练好的模型权重 (3.4 MB)
    ├── training_log.json  # 训练指标完整记录 (JSON)
    ├── training_curves.png# Loss / Accuracy / LR 曲线图
    ├── predictions.png    # 100 张测试样本预测可视化
    └── mnist_report.html  # 全过程 HTML 报告
```

## 文件说明

### `mnist_save.py`

通过 `torchvision.datasets.MNIST` 下载 MNIST 数据集，并将每张图像按标签分类保存为 JPG 文件。

- **训练集**：60,000 张图片，保存至 `mnist/train/{0-9}/`
- **测试集**：10,000 张图片，保存至 `mnist/test/{0-9}/`
- 使用 matplotlib 展示部分样本图像

参考了 [zyf-xtu/DL](https://github.com/zyf-xtu/DL/tree/master/classification/zyf_mnist) 仓库的代码实现。

### `cnn_train.py`

CNN 网络定义与训练脚本，包含完整的训练流程和指标追踪。

**网络架构**：双卷积块 + 全连接分类头

| 组件 | 配置 |
|------|------|
| Conv Block 1 | 2×(Conv2d(3×3)+BN+ReLU), MaxPool(2×2), Dropout2d(0.25) |
| Conv Block 2 | 2×(Conv2d(3×3)+BN+ReLU), MaxPool(2×2), Dropout2d(0.25) |
| Classifier | Flatten → Linear(3136→256)+ReLU → Dropout(0.5) → Linear(256→10) |

**训练配置**：

| 参数 | 值 |
|------|-----|
| 批次大小 | 128 |
| 训练轮次 | 15 |
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | StepLR (step=5, gamma=0.5) |
| 损失函数 | CrossEntropyLoss |
| 数据增强 | RandomRotation(±10°) |
| 归一化 | mean=0.1307, std=0.3081 |

**输出文件**：
- `results/mnist_cnn.pth` — 模型权重
- `results/training_log.json` — 每轮 loss/accuracy/lr/time 记录
- `results/training_curves.png` — 训练曲线图
- `results/predictions.png` — 预测样本网格图

### `generate_report.py`

读取训练日志和图片，生成包含全部过程记录的独立 HTML 报告 (`results/mnist_report.html`)。

报告内容包括：
1. 项目概述与流程
2. 数据准备与统计
3. CNN 算法原理详解（卷积、BN、ReLU、池化、Dropout）
4. 网络架构 HTML 可视化结构图
5. 训练配置与超参数
6. 每轮训练指标表格与曲线图
7. 预测样本可视化
8. 结论分析
9. 训练环境硬件信息

## 使用方法

### 环境要求

```bash
pip install torch torchvision numpy pillow matplotlib
```

### 运行步骤

```bash
# 1. 下载 MNIST 并保存为 JPG（可选，训练时也会自动下载）
KMP_DUPLICATE_LIB_OK=TRUE python mnist_save.py

# 2. 训练 CNN 模型
KMP_DUPLICATE_LIB_OK=TRUE python -u cnn_train.py

# 3. 生成 HTML 报告
KMP_DUPLICATE_LIB_OK=TRUE python generate_report.py
```

> **注意**：Anaconda 环境下的 OpenMP 库冲突需要设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

### 训练结果

| 指标 | 值 |
|------|-----|
| 最佳测试准确率 | 99.65% |
| 总参数量 | 871,018 |
| 训练时间 (CPU) | ~34 分钟 |

## 环境信息

- **Python**: 3.13
- **PyTorch**: 2.12.0 (CPU)
- **CPU**: Intel Core i5-10200H @ 2.40GHz
- **RAM**: 15.8 GB
- **OS**: Windows 10

## License

MIT
