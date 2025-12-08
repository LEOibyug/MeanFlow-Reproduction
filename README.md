# MeanFlow 复现

本项目提供了一个简洁的代码结构，用于复现 MeanFlow 的核心逻辑。MeanFlow 是一种基于 Diffusion Transformer (DiT) 架构，并利用潜在空间进行训练的模型。

## 项目结构

```
MeanFlow-Reproduction/
├── data/                   # 数据存储
│   ├── imagenette2/        # 原始数据集 (需要自行解压)
│   └── latents/            # 预处理后的 VAE 潜在向量 (.pt 文件)
├── models/
│   ├── __init__.py
│   └── dit.py              # DiT 模型定义 (修改后支持 t 和 r 输入)
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # 潜在数据集加载器
│   └── vae_prep.py         # 预处理脚本：图片 -> 潜在向量
├── scripts/
│   ├── run_exp_baseline.sh # 实验 A: 传统 Flow Matching
│   ├── run_exp_meanflow.sh # 实验 B: MeanFlow (核心)
│   ├── run_exp_cfg.sh      # 实验 C: MeanFlow + CFG
│   └── run_exp_fail.sh     # 实验 D: 消融实验 (破坏 JVP)
├── requirements.txt        # 项目依赖
├── train.py                # 主训练循环 (包含 JVP 和 Loss 计算核心)
└── sample.py               # 推理与可视化脚本 (1 步生成)
```

## 依赖关系

所需依赖项列在 `requirements.txt` 中：

- `torch>=2.0.0`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `tqdm`
- `numpy`
- `Pillow`
- `einops`

## 代码实现亮点

- **`models/dit.py`**: 包含一个轻量级的 DiT 实现。一个关键的修改是其 `TimestepEmbedder`，它通过将 $(t, r)$ 两个时间变量的嵌入 $(t, t-r)$ 相加作为条件输入来处理，正如论文所建议的。

- **`utils/dataset.py`**: 负责加载用于训练的预处理潜在数据集。

- **`utils/vae_prep.py`**: 此脚本用于初始数据准备，将原始图像转换为 VAE 潜在向量，然后用于训练。

- **`train.py`**: MeanFlow 模型训练的主脚本。它包含了 JVP (Jacobian-Vector Product) 和损失计算的核心逻辑。它支持不同的实验标志，如 `baseline`、`use_cfg` 和 `ablation_no_jvp`。

- **`sample.py`**: 此脚本用于执行 1 步推理并可视化生成的图像。它加载训练好的 MeanFlow 模型和 VAE 解码器，将潜在向量转换回图像。

## 操作流程

1.  **数据准备**: 首先，运行 `python utils/vae_prep.py` 将图像数据预处理为 VAE 潜在向量。此步骤可能需要大约 30 分钟。

2.  **启动实验**: 
    *   打开终端 1: 运行 `bash scripts/run_exp_baseline.sh` (使用 GPU 0)
    *   打开终端 2: 运行 `bash scripts/run_exp_meanflow.sh` (使用 GPU 1)

3.  **等待**: 训练过程可能需要大约 12 小时才能完成。

4.  **验证**: 修改 `sample.py` 中的 `exp_name` 并运行它以生成和查看结果图像。

尽管代码库简洁，但它严格遵循 MeanFlow 论文中的数学定义，并能够在双 A100 GPU 上复现核心 MeanFlow 效果。
