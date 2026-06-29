# SPORT - SIRIUS Project One-Run Toolkit

本文件夹包含运行 `main.py` 所需的所有代码和数据集，**完全自包含**，无需依赖外部文件即可运行。

---

## 📦 需要的 Python 包

```bash
pip install torch numpy scipy scikit-learn matplotlib tqdm pillow kmeans_gpu
```

推荐环境：
- Python ≥ 3.9
- PyTorch ≥ 1.12（支持 CUDA 11.8+）
- NumPy < 2.0（避免与 matplotlib 兼容性问题）

---

## 📁 文件夹架构

```
SPORT/
├── main.py                 # 主程序入口（训练流程、损失函数、Cayley prototype learner）
├── load_data.py            # 数据加载 + 缺失视图索引构造
├── Nmetrics.py             # 聚类评估指标（ACC、ARI、F-score、NMI、Purity）
├── network.py              # 多视图编码器模型定义
├── datasets.py             # PyTorch Dataset / Sampler 定义
├── utils.py                # 工具函数（get_Similarity、clustering）
├── README.md               # 本说明文档（本文件）
└── Dataset/                # 数据集目录（见下方说明）
    ├── animal.mat
    ├── ALOI_100_7.mat
    ├── Digit4k.mat
    ├── Reuters_21578.mat
    ├── 100leaves.mat
    ├── VGGFace2-50.mat
    └── ...（共 48 个 .mat 文件）
```

---

## 📊 Dataset 放置方式

### 目录结构

所有数据集文件（`.mat` 格式）必须放在 `SPORT/Dataset/` 目录下：

```
SPORT/Dataset/
├── animal.mat
├── ALOI_100_7.mat
├── Digit4k.mat
├── Reuters_21578.mat
├── 100leaves.mat
├── VGGFace2-50.mat
├── Caltech101_7.mat
├── LandUse_21_v73.mat
├── Scene_15.mat
└── ...（共 48 个数据集）
```

### 支持的数据集

当前支持的数据集及其对应的 `--i_d` 参数：

| `--i_d` | 数据集文件名              | 类别数 K | 视图数 V |
|---------|---------------------------|----------|----------|
| 0       | Caltech101_7.mat          | 7        | 6        |
| 3       | ALOI_100_7.mat            | 100      | 4        |
| 8       | 100leaves.mat             | 100      | 3        |
| 12      | animal.mat                | 50       | 6        |
| 20      | Digit4k.mat               | 10       | 2        |
| 34      | Reuters_21578.mat         | 10       | 5        |
| 39      | VGGFace2-50.mat           | 50       | 3        |
| ...     | ...                       | ...      | ...      |

完整映射见 `main.py` 中的 `DATASET_ID_TO_NAME` 字典。

### 数据集文件格式要求

- 格式：`.mat`（MATLAB 格式）
- 必须包含的字段：
  - `X`: 视图特征列表（cell array 或 struct array）
  - `Y`: 标签向量
- 缺失视图在运行时由 `load_data.py` 根据 `--missrate` 参数动态构造

---

## 🚀 快速启动

### 1. 激活虚拟环境（推荐）

```bash
cd /root/imvc_new/SPORT
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy scikit-learn matplotlib tqdm pillow kmeans_gpu
```

### 2. 运行示例

```bash
# Animal 数据集，缺失率 50%，随机种子 42
python main.py --i_d 12 --missrate 0.5 --seed 42 \
    --para_loss 1.6e-05 5 5 10 0.3 0.7

# Digit4k 数据集，缺失率 30%
python main.py --i_d 20 --missrate 0.3 --seed 1 \
    --pretrain_epochs 200 --FineTuning_epochs 100 \
    --lr_pre 0.0005 --lr_finetuning 0.00001
```

### 3. 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--i_d` | 12 | 数据集 ID（见上方表格） |
| `--missrate` | 0.4 | 缺失率（0.0 ~ 1.0） |
| `--seed` | 42 | 随机种子 |
| `--pretrain_epochs` | 200 | Pretraining 阶段 epoch 数 |
| `--FineTuning_epochs` | 100 | Fine-Tuning 阶段 epoch 数 |
| `--para_loss` | [1.6e-05, 5, 5, 10, 0.3, 0.7] | 6 个损失权重参数 |
| `--lr_pre` | 0.0005 | Pretraining 学习率 |
| `--lr_finetuning` | 0.00001 | Fine-Tuning 学习率 |

---

## 📝 输出说明

运行后会在当前目录生成：

- `models/`：保存的模型文件
- `logs/`：训练日志
- 控制台输出：每个 epoch 的损失和最终的聚类指标（ACC、ARI、F-score）

---

## ⚠️ 注意事项

1. **Dataset 必须放在 `SPORT/Dataset/`**，不要放在其他位置
2. **参数 `--para_loss` 必须提供 6 个值**，顺序为 `[alpha, beta, eta, lambda, gamma, w_neigh]`
3. **如果 GPU 显存不足**，程序会自动 fallback 到 CPU
4. **首次运行可能需要下载数据集**（如果 Dataset 目录为空）

---

*整理日期：2026-06-29*
*版本：v1.0*