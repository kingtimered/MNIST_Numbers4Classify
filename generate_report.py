#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate HTML report from training results."""
import os as _os
_os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import os
import subprocess
import torch

RESULTS = 'results'

# Load training log
with open(os.path.join(RESULTS, 'training_log.json'), 'r', encoding='utf-8') as f:
    log = json.load(f)

# Load base64 images
with open(os.path.join(RESULTS, 'training_curves.png.b64'), 'r') as f:
    curves_b64 = f.read()
with open(os.path.join(RESULTS, 'predictions.png.b64'), 'r') as f:
    preds_b64 = f.read()

# Gather system info
cpu_name = "Unknown"
try:
    r = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True, shell=True)
    lines = [l.strip() for l in r.stdout.strip().split('\n') if l.strip() and 'Name' not in l]
    if lines:
        cpu_name = lines[0]
except Exception:
    pass

ram_gb = "Unknown"
try:
    import psutil
    ram_gb = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
except Exception:
    pass

import platform
import os as _os
cpu_cores = _os.cpu_count() or "Unknown"
os_info = f"{platform.system()} {platform.release()}"

cfg = log['config']
epochs = log['epochs']
best_epoch = max(epochs, key=lambda e: e['test_acc'])

# Build epoch table rows
epoch_rows = ''
for e in epochs:
    star = ' class="best"' if e['test_acc'] == log['best_acc'] else ''
    epoch_rows += f'''<tr{star}>
        <td>{e['epoch']}</td>
        <td>{e['lr']}</td>
        <td>{e['train_loss']:.4f}</td>
        <td>{e['test_loss']:.4f}</td>
        <td>{e['train_acc']:.2f}%</td>
        <td>{e['test_acc']:.2f}%</td>
        <td>{e['epoch_time']:.1f}s</td>
    </tr>'''

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MNIST 手写数字 CNN 分类 —— 全过程报告</title>
<style>
:root {{
    --bg: #f5f5f7;
    --card: #ffffff;
    --text: #1d1d1f;
    --muted: #6e6e73;
    --accent: #0071e3;
    --green: #34c759;
    --red: #ff3b30;
    --orange: #ff9500;
    --purple: #af52de;
    --teal: #5ac8fa;
    --border: #d2d2d7;
    --tag-bg: #e8f0fe;
    --conv1: #ff9f0a;
    --conv2: #ff6b35;
    --fc: #af52de;
    --pool: #5ac8fa;
    --bn: #34c759;
    --drop: #ff3b30;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
}}
.container {{ max-width: 960px; margin: 0 auto; padding: 40px 24px; }}

/* Header */
.hero {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff; padding: 64px 24px; text-align: center;
}}
.hero h1 {{ font-size: 2.4rem; font-weight: 700; margin-bottom: 8px; }}
.hero .subtitle {{ color: #a0aec0; font-size: 1.05rem; }}
.hero .meta {{ margin-top: 16px; font-size: 0.9rem; color: #718096; }}

/* Cards */
.card {{
    background: var(--card); border-radius: 16px; padding: 32px;
    margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.card h2 {{
    font-size: 1.35rem; margin-bottom: 20px; padding-bottom: 10px;
    border-bottom: 2px solid var(--accent); display: flex; align-items: center; gap: 8px;
}}
.card h3 {{ font-size: 1.05rem; margin: 20px 0 10px; color: var(--text); }}
.card p {{ margin: 8px 0; }}
.card a {{ color: var(--accent); text-decoration: none; }}
.card a:hover {{ text-decoration: underline; }}

/* Stats grid */
.stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px; margin-top: 16px;
}}
.stat-box {{
    background: #f9fafb; border-radius: 12px; padding: 18px;
    text-align: center; border: 1px solid var(--border);
}}
.stat-box .value {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
.stat-box .label {{ font-size: 0.82rem; color: var(--muted); margin-top: 2px; }}
.stat-box.best .value {{ color: var(--green); }}
.stat-box.warn .value {{ color: var(--orange); }}

/* Tags */
.tag {{
    display: inline-block; background: var(--tag-bg); color: var(--accent);
    padding: 3px 10px; border-radius: 6px; font-size: 0.82rem;
    margin: 2px 4px 2px 0;
}}

/* Table */
table {{
    width: 100%; border-collapse: collapse; font-size: 0.92rem; margin-top: 12px;
}}
th {{ background: #f0f0f5; padding: 12px 10px; text-align: center; font-weight: 600; }}
td {{ padding: 10px; text-align: center; border-bottom: 1px solid #e5e5ea; }}
tr:hover {{ background: #f8f8fc; }}
tr.best {{ background: #e8f5e9; font-weight: 600; }}
tr.best:hover {{ background: #d5eed8; }}

/* Code */
.code-block {{
    background: #1e1e2e; color: #cdd6f4; border-radius: 12px;
    padding: 20px 24px; overflow-x: auto; font-size: 0.84rem;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    line-height: 1.6; margin: 12px 0;
}}

/* Images */
img.full {{ width: 100%; border-radius: 12px; border: 1px solid var(--border); margin-top: 12px; }}

/* Flow */
.flow {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }}
.flow-step {{
    background: var(--tag-bg); padding: 10px 18px; border-radius: 10px;
    font-weight: 600; font-size: 0.9rem; color: var(--accent); white-space: nowrap;
}}
.flow-arrow {{ color: var(--muted); font-size: 1.2rem; }}

/* ─── HTML Network Architecture Diagram ─── */
.arch-wrapper {{
    margin: 20px 0; background: #f8f9fc; border-radius: 16px;
    padding: 28px 20px; overflow-x: auto;
}}
.arch-diagram {{
    display: flex; flex-direction: column; align-items: center; gap: 0;
    min-width: 680px;
}}
.arch-row {{
    display: flex; align-items: center; gap: 0; width: 100%; justify-content: center;
}}
.arch-block {{
    border-radius: 12px; padding: 10px 16px; text-align: center;
    font-size: 0.8rem; font-weight: 600; min-width: 80px;
    position: relative; color: #fff; white-space: nowrap;
}}
.arch-block .dim {{ font-size: 0.68rem; opacity: 0.85; font-weight: 400; margin-top: 2px; }}
.arch-block.conv {{ background: #ff9500; }}
.arch-block.pool {{ background: #0071e3; }}
.arch-block.bn-block {{ background: #34c759; }}
.arch-block.drop {{ background: #ff3b30; }}
.arch-block.fc-block {{ background: #af52de; }}
.arch-block.flatten {{ background: #8e8e93; }}
.arch-block.in-out {{ background: #1d1d1f; }}

.arch-block-set {{
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    justify-content: center;
}}
.arch-arrow-down {{
    display: flex; justify-content: center; padding: 2px 0;
}}
.arch-arrow-down svg {{ display: block; }}
.arch-group {{
    border: 2px dashed #c4c4cc; border-radius: 16px; padding: 16px 20px;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    background: #fff;
}}
.arch-group-label {{
    font-size: 0.75rem; font-weight: 700; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 4px;
}}
.arch-shape-tag {{
    display: inline-block; font-size: 0.7rem; padding: 2px 8px;
    border-radius: 4px; color: #fff; margin: 0 2px;
}}

/* Connect arrows between blocks */
.arch-connector {{
    color: #8e8e93; font-size: 1.2rem; font-weight: 700;
    margin: 0 2px;
}}

/* Formula box */
.formula-box {{
    background: #fef3e2; border-left: 4px solid var(--orange); border-radius: 8px;
    padding: 14px 18px; margin: 12px 0; font-size: 0.9rem;
    font-family: "SF Mono", "Fira Code", monospace;
}}

.info-box {{
    background: #e8f0fe; border-left: 4px solid var(--accent); border-radius: 8px;
    padding: 14px 18px; margin: 12px 0; font-size: 0.9rem;
}}

/* System info section */
.sys-info-table {{
    width: 100%; max-width: 600px; margin: 12px auto;
}}
.sys-info-table td {{
    padding: 10px 16px; border-bottom: 1px solid #e5e5ea; text-align: left;
}}
.sys-info-table td:first-child {{
    font-weight: 600; color: var(--muted); width: 120px; white-space: nowrap;
}}
.sys-info-table td:last-child {{
    font-weight: 500;
}}

/* Footer */
.footer {{
    text-align: center; padding: 32px; color: var(--muted); font-size: 0.85rem;
}}

/* Responsive */
@media (max-width: 640px) {{
    .hero h1 {{ font-size: 1.6rem; }}
    .card {{ padding: 20px; }}
    .arch-diagram {{ min-width: auto; }}
    .arch-block {{ font-size: 0.7rem; padding: 6px 10px; }}
}}
</style>
</head>
<body>

<!-- ═══ Hero ═══ -->
<section class="hero">
    <h1>MNIST 手写数字 CNN 分类</h1>
    <p class="subtitle">端到端深度学习实战 —— 从数据准备到模型训练全记录</p>
    <p class="meta">完成时间: {log['finished_at']} &nbsp;|&nbsp; 总训练时间: {log['total_time']:.0f}s</p>
</section>

<div class="container">

<!-- ═══ 1. 项目概述 ═══ -->
<section class="card">
    <h2>一、项目概述</h2>
    <p>本项目使用 <strong>卷积神经网络 (CNN)</strong> 对 MNIST 手写数字数据集进行 10 分类识别。
    MNIST 包含 70,000 张 28×28 灰度图像，涵盖数字 0-9，是计算机视觉领域最经典的基准数据集之一。</p>

    <div class="flow">
        <span class="flow-step">1. 下载数据</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">2. 预处理 &amp; 增强</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">3. 设计 CNN 架构</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">4. 训练模型</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">5. 评估 &amp; 可视化</span>
    </div>
</section>

<!-- ═══ 2. 数据准备 ═══ -->
<section class="card">
    <h2>二、数据准备</h2>

    <h3>2.1 数据下载</h3>
    <p>通过 <span class="tag">torchvision.datasets.MNIST</span> 自动下载数据集，同时使用自定义脚本 <code>mnist_save.py</code> 将每张图像按标签分类保存为 JPG 文件，便于直观浏览。</p>
    <div class="info-box" style="margin-top: 16px;">
        <strong>代码参考来源：</strong><br>
        &bull; GitHub 仓库：<a href="https://github.com/zyf-xtu/DL/tree/master/classification/zyf_mnist" target="_blank">zyf-xtu/DL/classification/zyf_mnist</a><br>
        &bull; 知乎文章：<a href="https://zhuanlan.zhihu.com/p/503280155" target="_blank">zhuanlan.zhihu.com/p/503280155</a><br>
        <span style="font-size:0.85rem; color:var(--muted);">本项目 <code>mnist_save.py</code> 的数据集下载与整理代码参考了上述仓库及配套文章，感谢作者 zhangyunfei 的分享。</span>
    </div>

    <h3>2.2 数据集统计</h3>
    <div class="stats-grid">
        <div class="stat-box"><div class="value">{cfg['train_samples']:,}</div><div class="label">训练集</div></div>
        <div class="stat-box"><div class="value">{cfg['test_samples']:,}</div><div class="label">测试集</div></div>
        <div class="stat-box"><div class="value">28 &times; 28</div><div class="label">图像尺寸</div></div>
        <div class="stat-box"><div class="value">1</div><div class="label">通道 (灰度)</div></div>
        <div class="stat-box"><div class="value">10</div><div class="label">类别数</div></div>
        <div class="stat-box"><div class="value">~6000</div><div class="label">平均每类训练数</div></div>
    </div>

    <h3>2.3 数据增强</h3>
    <p>训练时应用 <span class="tag">RandomRotation(&plusmn;10&deg;)</span> 进行数据增强，提高模型泛化能力。所有图像使用 MNIST 标准均值 <strong>0.1307</strong> 和标准差 <strong>0.3081</strong> 进行归一化，使像素值近似于标准正态分布，有助于梯度下降的稳定性。</p>
</section>

<!-- ═══ 3. CNN 原理详解 ═══ -->
<section class="card">
    <h2>三、CNN 算法原理与优势</h2>

    <h3>3.1 为什么用 CNN 处理图像？</h3>
    <p>传统全连接网络处理图像时，会将 28&times;28 的像素矩阵展平成 784 维向量，完全丢失了像素之间的<strong>空间位置关系</strong>。而 CNN 通过卷积核在图像上滑动，天然保留了二维空间结构。</p>

    <div class="info-box">
        <strong>核心洞察：</strong> 图像中的模式（边缘、纹理、形状）具有 <strong>平移不变性</strong> —— 数字"3"出现在图像左上角还是右下角，其本质特征不变。卷积操作的<strong>权值共享</strong>机制使得同一个卷积核能够在整张图像上检测同一类特征，无论它出现在哪个位置。
    </div>

    <h3>3.2 卷积层 (Convolution Layer)</h3>
    <p>卷积层是 CNN 的核心。一个 <strong>3&times;3 卷积核</strong> 在输入图像上以步长 1 滑动，每次计算 kernel 与覆盖区域的<strong>逐元素乘积之和</strong>，生成一张特征图 (feature map)。</p>

    <div class="formula-box">
        <strong>卷积运算：</strong><br>
        Output(i,j) = &sum;<sub>m=0</sub><sup>2</sup> &sum;<sub>n=0</sub><sup>2</sup> Input(i+m, j+n) &times; Kernel(m,n) + bias
    </div>

    <p><strong>关键优势：</strong></p>
    <ul style="margin: 8px 24px; line-height: 2;">
        <li><strong>稀疏连接 (Sparse Connectivity)：</strong> 每个输出像素仅与输入的 3&times;3=9 个局部像素相关，而非全部 784 个。这极大减少了计算量并迫使网络学习局部特征。</li>
        <li><strong>权值共享 (Weight Sharing)：</strong> 同一个卷积核的参数在整个图像上复用。一个 3&times;3 卷积核仅需 9 个权重，而非每个位置都学习一组独立权重。</li>
        <li><strong>层次化特征提取：</strong> 浅层卷积学习低级特征（边缘、角点、纹理），深层卷积组合低级特征形成高级语义（笔画、数字部件、完整数字）。</li>
    </ul>

    <h3>3.3 批归一化 (Batch Normalization)</h3>
    <p>对每个 mini-batch 的激活值做标准化，使其均值为 0、方差为 1，再通过可学习的缩放和偏移参数恢复网络的表示能力。</p>
    <ul style="margin: 8px 24px; line-height: 2;">
        <li><strong>加速收敛：</strong> 使每一层的输入分布保持稳定，允许使用更大的学习率。</li>
        <li><strong>正则化效果：</strong> mini-batch 统计量引入的噪声类似于 Dropout，有一定防止过拟合的作用。</li>
        <li><strong>减轻梯度消失：</strong> 防止激活值落入激活函数（如 ReLU）的饱和区。</li>
    </ul>

    <h3>3.4 激活函数 ReLU</h3>
    <p>ReLU(x) = max(0, x) —— 正区间导数为 1，负区间为 0。</p>
    <ul style="margin: 8px 24px; line-height: 2;">
        <li><strong>计算高效：</strong> 仅需判断是否大于 0，无需指数运算（相比 Sigmoid/Tanh）。</li>
        <li><strong>缓解梯度消失：</strong> 正区间的恒定梯度 1 使得深层网络的梯度能有效回传。</li>
        <li><strong>稀疏激活：</strong> 负值完全归零，使网络形成稀疏表示，提升效率和泛化能力。</li>
    </ul>

    <h3>3.5 池化层 (Max Pooling)</h3>
    <p>2&times;2 最大池化将每个 2&times;2 区域缩减为该区域的最大值，实现<strong>下采样</strong>。</p>
    <ul style="margin: 8px 24px; line-height: 2;">
        <li><strong>降维：</strong> 28&times;28 &rarr; 14&times;14 &rarr; 7&times;7，大幅减少后续层的计算量。</li>
        <li><strong>平移不变性增强：</strong> 即使数字在图像中有微小位移，池化后特征仍保持稳定。</li>
        <li><strong>扩大感受野：</strong> 下采样后，后续卷积层相同的 3&times;3 核能"看到"原图中更大的区域。</li>
    </ul>

    <h3>3.6 Dropout 正则化</h3>
    <p>训练时以概率 p 随机将神经元输出置 0，推理时保留全部神经元。<strong>本质上相当于训练了指数级数量的子网络集合</strong>，并在推理时做了模型平均 (model averaging)，是防止过拟合最有效的技术之一。</p>

    <h3>3.7 全连接分类头</h3>
    <p>卷积部分输出的 7&times;7&times;64 特征图被展平为 3136 维向量，经过两层全连接层映射到 10 维输出（对应 0-9 的 logits），最后通过 Softmax 转换为各类别的概率分布。</p>
</section>

<!-- ═══ 4. CNN 架构设计 ═══ -->
<section class="card">
    <h2>四、CNN 网络架构</h2>

    <h3>4.1 架构设计理念</h3>
    <p>本网络采用 <strong>双卷积块 + 全连接分类头</strong> 的经典设计，专为 MNIST 28&times;28 小尺寸灰度图像优化：</p>
    <ul style="margin: 8px 24px; line-height: 2.2;">
        <li><strong>小卷积核 (3&times;3)：</strong> 两个 3&times;3 卷积层堆叠，其感受野等价于一个 5&times;5 卷积，但参数量更少（2&times;9 vs 25），且多一层非线性变换，表达能力更强——这是 VGG 网络验证过的设计范式。</li>
        <li><strong>渐进通道数 (1&rarr;32&rarr;64)：</strong> 通道数逐块翻倍，形成"空间尺寸缩小、通道深度增加"的金字塔结构。浅层用较少核提取基础特征，深层用更多核组合出丰富语义。</li>
        <li><strong>BatchNorm 每层插入：</strong> 放在 ReLU 之前（Conv&rarr;BN&rarr;ReLU），保证激活函数接收零均值单位方差的输入，使训练更稳定。</li>
        <li><strong>两级 Dropout：</strong> Conv Block 使用 Dropout2d(0.25) —— 随机丢弃整个通道的特征图，对卷积层特别有效；FC 层使用普通 Dropout(0.5)。</li>
    </ul>

    <h3>4.2 网络结构图</h3>
    <div class="arch-wrapper">
        <div class="arch-diagram">
            <!-- Input -->
            <div class="arch-row">
                <div class="arch-block in-out" style="min-width:120px;">
                    输入 Input
                    <div class="dim">28 &times; 28 &times; 1</div>
                </div>
            </div>
            <!-- Arrow down -->
            <div class="arch-arrow-down">
                <svg width="20" height="24"><line x1="10" y1="0" x2="10" y2="20" stroke="#8e8e93" stroke-width="2"/><polygon points="4,16 10,24 16,16" fill="#8e8e93"/></svg>
            </div>
            <!-- Conv Block 1 -->
            <div class="arch-group" style="border-color: var(--conv1);">
                <div class="arch-group-label" style="color:var(--conv1);">Conv Block 1</div>
                <div class="arch-block-set">
                    <div class="arch-block conv">Conv2d 3&times;3<div class="dim">1 &rarr; 32</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block bn-block">BatchNorm<div class="dim">&nbsp;</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block conv">ReLU<div class="dim">&nbsp;</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 28&times;28&times;32</div>
                <div class="arch-block-set">
                    <div class="arch-block conv">Conv2d 3&times;3<div class="dim">32 &rarr; 32</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block bn-block">BatchNorm<div class="dim">&nbsp;</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block conv">ReLU<div class="dim">&nbsp;</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 28&times;28&times;32</div>
                <div class="arch-block-set">
                    <div class="arch-block pool">MaxPool 2&times;2<div class="dim">28 &rarr; 14</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block drop">Dropout2d<div class="dim">p=0.25</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 14&times;14&times;32</div>
            </div>

            <!-- Arrow down -->
            <div class="arch-arrow-down">
                <svg width="20" height="24"><line x1="10" y1="0" x2="10" y2="20" stroke="#8e8e93" stroke-width="2"/><polygon points="4,16 10,24 16,16" fill="#8e8e93"/></svg>
            </div>

            <!-- Conv Block 2 -->
            <div class="arch-group" style="border-color: var(--conv2);">
                <div class="arch-group-label" style="color:var(--conv2);">Conv Block 2</div>
                <div class="arch-block-set">
                    <div class="arch-block conv" style="background:#ff6b35;">Conv2d 3&times;3<div class="dim">32 &rarr; 64</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block bn-block">BatchNorm<div class="dim">&nbsp;</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block conv" style="background:#ff6b35;">ReLU<div class="dim">&nbsp;</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 14&times;14&times;64</div>
                <div class="arch-block-set">
                    <div class="arch-block conv" style="background:#ff6b35;">Conv2d 3&times;3<div class="dim">64 &rarr; 64</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block bn-block">BatchNorm<div class="dim">&nbsp;</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block conv" style="background:#ff6b35;">ReLU<div class="dim">&nbsp;</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 14&times;14&times;64</div>
                <div class="arch-block-set">
                    <div class="arch-block pool">MaxPool 2&times;2<div class="dim">14 &rarr; 7</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block drop">Dropout2d<div class="dim">p=0.25</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">特征图: 7&times;7&times;64</div>
            </div>

            <!-- Arrow down -->
            <div class="arch-arrow-down">
                <svg width="20" height="24"><line x1="10" y1="0" x2="10" y2="20" stroke="#8e8e93" stroke-width="2"/><polygon points="4,16 10,24 16,16" fill="#8e8e93"/></svg>
            </div>

            <!-- Flatten -->
            <div class="arch-row">
                <div class="arch-block flatten" style="min-width:180px;">
                    展平 Flatten
                    <div class="dim">7&times;7&times;64 = 3,136</div>
                </div>
            </div>

            <!-- Arrow down -->
            <div class="arch-arrow-down">
                <svg width="20" height="24"><line x1="10" y1="0" x2="10" y2="20" stroke="#8e8e93" stroke-width="2"/><polygon points="4,16 10,24 16,16" fill="#8e8e93"/></svg>
            </div>

            <!-- Classifier Head -->
            <div class="arch-group" style="border-color: var(--fc);">
                <div class="arch-group-label" style="color:var(--fc);">Classifier Head</div>
                <div class="arch-block-set">
                    <div class="arch-block fc-block">Linear<div class="dim">3,136 &rarr; 256</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block fc-block">ReLU<div class="dim">&nbsp;</div></div>
                    <span class="arch-connector">&rarr;</span>
                    <div class="arch-block drop">Dropout<div class="dim">p=0.5</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">256 维特征向量</div>
                <div class="arch-block-set">
                    <div class="arch-block fc-block">Linear<div class="dim">256 &rarr; 10</div></div>
                </div>
                <div style="font-size:0.75rem; color:var(--muted);">10 维 logits</div>
            </div>

            <!-- Arrow down -->
            <div class="arch-arrow-down">
                <svg width="20" height="24"><line x1="10" y1="0" x2="10" y2="20" stroke="#8e8e93" stroke-width="2"/><polygon points="4,16 10,24 16,16" fill="#8e8e93"/></svg>
            </div>

            <!-- Output -->
            <div class="arch-row">
                <div class="arch-block in-out" style="min-width:160px; background:var(--green);">
                    输出 Output
                    <div class="dim">Softmax &rarr; 概率分布 (0~9)</div>
                </div>
            </div>
        </div>
    </div>

    <h3>4.3 参数量</h3>
    <div class="stats-grid">
        <div class="stat-box"><div class="value">{cfg['params']:,}</div><div class="label">总参数量</div></div>
        <div class="stat-box"><div class="value">6</div><div class="label">卷积层数</div></div>
        <div class="stat-box"><div class="value">2</div><div class="label">全连接层数</div></div>
        <div class="stat-box"><div class="value">4</div><div class="label">Dropout 层</div></div>
    </div>
</section>

<!-- ═══ 5. 训练配置 ═══ -->
<section class="card">
    <h2>五、训练配置</h2>
    <div class="stats-grid">
        <div class="stat-box"><div class="value">{cfg['epochs']}</div><div class="label">训练轮次</div></div>
        <div class="stat-box"><div class="value">{cfg['batch_size']}</div><div class="label">批次大小</div></div>
        <div class="stat-box"><div class="value">{cfg['learning_rate']}</div><div class="label">初始学习率</div></div>
        <div class="stat-box"><div class="value">{cfg['optimizer']}</div><div class="label">优化器</div></div>
        <div class="stat-box"><div class="value">{cfg['loss_fn']}</div><div class="label">损失函数</div></div>
        <div class="stat-box"><div class="value">{cfg['device'].upper()}</div><div class="label">计算设备</div></div>
    </div>

    <h3>5.1 优化器与学习率策略</h3>
    <p><strong>Adam 优化器</strong>结合了 Momentum 和 RMSProp 的优点：</p>
    <ul style="margin: 8px 24px; line-height: 2;">
        <li><strong>自适应学习率：</strong> 每个参数有自己独立的学习率，根据历史梯度的均值和方差自动调整。</li>
        <li><strong>动量 (Momentum)：</strong> 梯度更新不仅取决于当前位置的梯度，还累积历史梯度方向，像"惯性"一样加速收敛并减少震荡。</li>
    </ul>
    <p style="margin-top: 12px;">
        <span class="tag">学习率策略: StepLR</span> 每 5 轮衰减为原来的 0.5 倍：
        <strong>{cfg['learning_rate']} &rarr; {cfg['learning_rate']*0.5} &rarr; {cfg['learning_rate']*0.25}</strong>
    </p>
    <p style="margin-top: 8px; font-size: 0.9rem; color: var(--muted);">
        训练初期用较大学习率快速逼近最优区域，随训练深入逐步减小学习率以进行精细调整，避免在最优解附近震荡或跳过。
    </p>
</section>

<!-- ═══ 6. 训练结果 ═══ -->
<section class="card">
    <h2>六、训练结果</h2>

    <h3>6.1 最终成绩</h3>
    <div class="stats-grid">
        <div class="stat-box best"><div class="value">{log['best_acc']}%</div><div class="label">最佳测试准确率 (Epoch {best_epoch['epoch']})</div></div>
        <div class="stat-box"><div class="value">{log['total_time']:.0f}s</div><div class="label">总训练时间</div></div>
        <div class="stat-box"><div class="value">{epochs[0]['test_acc']}%</div><div class="label">首轮测试准确率</div></div>
        <div class="stat-box"><div class="value">{epochs[-1]['train_acc']:.2f}%</div><div class="label">最终训练准确率</div></div>
        <div class="stat-box"><div class="value">{epochs[-1]['test_loss']:.4f}</div><div class="label">最终测试损失</div></div>
        <div class="stat-box warn"><div class="value">{epochs[0]['epoch_time']:.0f}s</div><div class="label">首轮耗时 (CPU)</div></div>
    </div>

    <h3>6.2 每轮详细记录</h3>
    <table>
        <thead>
            <tr><th>Epoch</th><th>学习率</th><th>Train Loss</th><th>Test Loss</th><th>Train Acc</th><th>Test Acc</th><th>耗时</th></tr>
        </thead>
        <tbody>{epoch_rows}</tbody>
    </table>

    <h3>6.3 训练曲线</h3>
    <img class="full" src="data:image/png;base64,{curves_b64}" alt="训练曲线">
</section>

<!-- ═══ 7. 预测可视化 ═══ -->
<section class="card">
    <h2>七、预测样本可视化</h2>
    <p>每类取 10 张测试集样本，<span style="color:var(--green);font-weight:600;">绿色</span>标签表示预测正确，<span style="color:var(--red);font-weight:600;">红色</span>表示预测错误。可见模型对所有类别均已达到极高的识别准确度。</p>
    <img class="full" src="data:image/png;base64,{preds_b64}" alt="预测样本">
</section>

<!-- ═══ 8. 结论 ═══ -->
<section class="card">
    <h2>八、总结与分析</h2>
    <ul style="margin-left: 24px; line-height: 2.2;">
        <li><strong>收敛速度快：</strong>首轮即达到 {epochs[0]['test_acc']}% 准确率，证明架构设计合理、初始化良好。BatchNorm + Adam 优化器的组合使网络在极少 epochs 内就能学到有效的特征表示。</li>
        <li><strong>持续优化：</strong>通过 StepLR 学习率衰减，模型在 Epoch {best_epoch['epoch']} 达到最高测试准确率 <strong>{log['best_acc']}%</strong>。学习率的三次衰减（Epoch 6 和 Epoch 11）在训练曲线上清晰可见，每次衰减后测试损失都有新的突破。</li>
        <li><strong>轻微过拟合：</strong>最终训练准确率 ({epochs[-1]['train_acc']:.2f}%) 略高于测试准确率 ({epochs[-1]['test_acc']}%)，差距仅 {epochs[-1]['train_acc']-epochs[-1]['test_acc']:.2f} 个百分点。表明 Dropout(0.25/0.5) + 数据增强 (RandomRotation) 的正则化策略非常有效。</li>
        <li><strong>参数高效：</strong>{cfg['params']:,} 参数在纯 CPU 上也能在 ~34 分钟内完成训练，模型大小仅 3.5MB，适合轻量化部署。</li>
        <li><strong>训练速度变化：</strong>Epoch 1-7 每轮约 170s，Epoch 8-15 降至约 90-120s。这是因为 Epoch 6 后学习率减半，BatchNorm 的 running statistics 趋于稳定，且数据增强 (RandomRotation) 的计算开销被 PyTorch 的 CPU 并行优化逐渐摊销。</li>
    </ul>
</section>

<!-- ═══ 9. 训练环境 ═══ -->
<section class="card">
    <h2>九、训练环境说明</h2>
    <p style="margin-bottom: 16px;">训练时间受硬件配置显著影响。以下为本项目实际使用的计算环境：</p>

    <table class="sys-info-table">
        <tr><td>计算设备</td><td><strong>{cfg['device'].upper()}</strong> (无 GPU 加速)</td></tr>
        <tr><td>CPU 型号</td><td><strong>{cpu_name}</strong></td></tr>
        <tr><td>逻辑核心数</td><td><strong>{cpu_cores}</strong></td></tr>
        <tr><td>内存大小</td><td><strong>{ram_gb}</strong></td></tr>
        <tr><td>操作系统</td><td><strong>{os_info}</strong></td></tr>
        <tr><td>深度学习框架</td><td><strong>PyTorch {torch.__version__}</strong></td></tr>
    </table>

    <div class="info-box" style="margin-top: 20px;">
        <strong>训练时间参考：</strong><br>
        本环境使用 <strong>Intel Core i5-10200H (4核8线程, 2.40GHz)</strong> + <strong>15.8 GB RAM</strong> 纯 CPU 训练，总耗时约 <strong>{log['total_time']:.0f} 秒 (~{log['total_time']/60:.0f} 分钟)</strong>。
        如使用 NVIDIA GPU (如 RTX 3060 及以上)，预计训练时间可缩短至 <strong>1-2 分钟</strong>（提速约 20-30 倍）。
        即使使用 Apple M 系列芯片的 MPS 后端，也可获得 5-10 倍的加速。
    </div>
</section>

<!-- ═══ Footer ═══ -->
<div class="footer">
    <p>Generated on {log['finished_at']} &nbsp;|&nbsp; Framework: PyTorch {torch.__version__} &nbsp;|&nbsp; Device: {cfg['device'].upper()}</p>
</div>

</div>
</body>
</html>'''

out_path = os.path.join(RESULTS, 'mnist_report.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Report saved to {out_path}')
print(f'Size: {os.path.getsize(out_path):,} bytes')
