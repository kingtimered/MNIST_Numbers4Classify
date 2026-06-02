# MNIST 手写数字 CNN 分类

基于卷积神经网络 (CNN) 的 MNIST 手写数字识别项目，包含数据准备、CPU/GPU 双模式训练、评估可视化全流程。

**注意目前未上传数据集 MNIST 文件夹，需要运行 mnist_save.py 下载**

## 项目来源

- **数据集**：[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 张 28×28 灰度手写数字图像，10 分类
- **参考代码**：[zyf-xtu/DL/classification/zyf_mnist](https://github.com/zyf-xtu/DL/tree/master/classification/zyf_mnist) — 数据下载与整理的 `mnist_save.py` 参考了该仓库
- **参考文章**：[知乎 - zhangyunfei](https://zhuanlan.zhihu.com/p/503280155)

## 项目结构

```
MNIST_Numbers/
├── mnist_save.py            # 数据集下载与图像保存脚本
├── cnn_train.py             # CNN 模型训练脚本 (CPU)
├── cnn_train_gpu.py         # CNN 模型训练脚本 (GPU 加速版)
├── generate_report.py       # CPU 训练报告生成脚本
├── generate_report_gpu.py   # GPU 训练报告生成脚本
├── mnist_gui.py             # 交互式手写数字识别 GUI
├── README.md                # 项目说明文档
├── mnist/                   # 数据集目录
│   ├── MNIST/raw/           # 原始 .gz 压缩文件 (torchvision 下载)
│   └── train/               # 训练集 JPG 图片 (按 0-9 分类)
│   └── test/                # 测试集 JPG 图片 (按 0-9 分类)
└── results/                 # 输出结果
    ├── mnist_cnn.pth        # CPU 训练的模型权重 (3.4 MB)
    ├── mnist_cnn_gpu.pth    # GPU 训练的模型权重 (3.4 MB)
    ├── training_log.json    # CPU 训练指标记录 (JSON)
    ├── training_log_gpu.json# GPU 训练指标记录 (JSON)
    ├── training_curves.png  # CPU 训练 Loss/Accuracy/LR 曲线图
    ├── training_curves_gpu.png # GPU 训练曲线图
    ├── predictions.png      # CPU 模型预测样本可视化
    ├── predictions_gpu.png  # GPU 模型预测样本可视化
    ├── mnist_report.html    # CPU 训练全过程 HTML 报告
    └── mnist_report_gpu.html # GPU 训练全过程 HTML 报告
```

## 文件说明

### `mnist_save.py`

通过 `torchvision.datasets.MNIST` 下载 MNIST 数据集，并将每张图像按标签分类保存为 JPG 文件。

- **训练集**：60,000 张图片，保存至 `mnist/train/{0-9}/`
- **测试集**：10,000 张图片，保存至 `mnist/test/{0-9}/`
- 使用 matplotlib 展示部分样本图像

参考了 [zyf-xtu/DL](https://github.com/zyf-xtu/DL/tree/master/classification/zyf_mnist) 仓库的代码实现。

### `cnn_train.py`

CNN 网络定义与 **CPU 训练**脚本，包含完整的训练流程和指标追踪。

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
| 计算设备 | CPU |

### `cnn_train_gpu.py`

CNN 网络定义与 **GPU 加速训练**脚本。自动检测可用设备（CUDA > MPS > CPU），在 NVIDIA GPU 上启用混合精度 (AMP) 训练。

**GPU 专属优化**：

| 特性 | CPU 方案 | GPU 方案 |
|------|---------|---------|
| Batch Size | 128 | 512 |
| 混合精度 | 无 | FP16 (AMP) |
| DataLoader 进程 | 0 | 2 |
| 内存传输 | 默认 | pin_memory + non_blocking |
| 梯度清零 | zero_grad() | zero_grad(set_to_none=True) |

### `generate_report.py` / `generate_report_gpu.py`

分别读取 CPU/GPU 训练日志和模型，生成各自的全过程 HTML 报告。

### `mnist_gui.py`

交互式手写数字识别 GUI 程序，可直接测试训练好的模型。

**功能**：
- **单个数字模式**：鼠标在画布上写一个数字，实时识别并显示 Top-10 概率分布
- **数字串模式**：在宽画布上写一串数字（如电话号码），程序自动切割每个数字并逐一识别
- 自动检测 CUDA/CPU 设备
- 鼠标松开即自动识别，支持回车键快捷识别

**运行**：
```bash
KMP_DUPLICATE_LIB_OK=TRUE python mnist_gui.py
```

### CPU vs GPU 训练对比

在 Intel Core i5-10200H + NVIDIA RTX 2070 Max-Q 环境下实测：

| 指标 | CPU (i5-10200H) | GPU (RTX 2070) | 提升 |
|------|----------------|-----------------|------|
| 单 Epoch 平均耗时 | ~137s | **~17s** | **8.1×** |
| 总训练时间 | 2,052s (34 min) | **340s (5.7 min)** | **6.0×** |
| 训练吞吐量 | ~440 samples/s | **~3,500 samples/s** | **8.0×** |
| Epoch 数 | 15 | **20** | +33% |
| Batch Size | 128 | **512** | 4× |
| 混合精度 | 无 | **FP16 AMP** | — |
| 最佳测试准确率 | **99.65%** | 99.56% | -0.09% |
| 模型参数量 | 871,018 | 871,018 | 相同 |

> GPU 以约 **6 倍** 的时间完成了更多轮次的训练（20 vs 15），准确率仅差 0.09 个百分点，处于正常统计波动范围。
> 将 GPU batch size 调整至与 CPU 一致 (128) 可进一步缩小准确率差距。

## 使用方法

### 环境要求

**CPU 训练**（无需 GPU）：
```bash
pip install torch torchvision numpy pillow matplotlib
```

**GPU 训练**（需要 NVIDIA GPU + CUDA）：
```bash
# 安装 CUDA 版 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow matplotlib
```

### 运行步骤

```bash
# 0. (可选) 下载 MNIST 并保存为 JPG
KMP_DUPLICATE_LIB_OK=TRUE python mnist_save.py

# 1. GPU 训练 (推荐)
KMP_DUPLICATE_LIB_OK=TRUE python -u cnn_train_gpu.py

# 1. 或 CPU 训练
KMP_DUPLICATE_LIB_OK=TRUE python -u cnn_train.py

# 2. 生成对应报告
KMP_DUPLICATE_LIB_OK=TRUE python generate_report_gpu.py   # GPU 报告
KMP_DUPLICATE_LIB_OK=TRUE python generate_report.py       # CPU 报告

# 3. 启动交互式识别界面
KMP_DUPLICATE_LIB_OK=TRUE python mnist_gui.py
```

> **注意**：Anaconda 环境下的 OpenMP 库冲突需要设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

### 训练结果

| 指标 | CPU | GPU |
|------|-----|-----|
| 最佳测试准确率 | 99.65% | 99.56% |
| 总参数量 | 871,018 | 871,018 |
| 训练时间 | ~34 分钟 | **~5.7 分钟** |

## 环境信息

- **Python**: 3.13
- **PyTorch**: 2.6.0+cu124 (GPU) / 2.12.0 (CPU)
- **CPU**: Intel Core i5-10200H @ 2.40GHz
- **GPU**: NVIDIA GeForce RTX 2070 with Max-Q Design (8 GB)
- **RAM**: 15.8 GB
- **OS**: Windows 10

## License

MIT
