# MPFF-IS
A Molecular–Protein Fusion Framework for Discovery of KRAS G12D Inhibitors in Pancreatic Ductal Adenocarcinoma


## 环境设置与依赖安装

请确保以下依赖版本一致：

- **DGL (Deep Graph Library)**：2.4.0（需 CUDA 12.1 支持）  
- **DGL-LifeSci**：0.3.2  
- **scikit-learn**：1.6.1  

建议使用 Conda 创建独立环境：

```bash
pip install dgl-cu121==2.4.0
pip install dgllife==0.3.2
pip install scikit-learn==1.6.1
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Frozen-Zephyr/MPFF-IS.git
cd MPFF-IS
```

### 2. 下载模型文件

从huggingface上下载esm2_t33_650M_UR50D模型文件到MPFF-IS文件夹：
https://huggingface.co/facebook/esm2_t33_650M_UR50D

或使用wget下载：

```bash
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
```

### 3. 运行代码

直接进入环境，运行MPFF-IS.py：
```bash
python MPFF-IS.py
```

## 使用方法（Usage）

以下为 `MPFF-IS` 的命令行参数说明及示例用法。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-f` | str  | 无 | 输入单个 CSV 文件路径 |
| `-F` | str  | 无 | 输入包含多个 CSV 文件的文件夹路径（不可与 `-f` 同时使用） |
| `-m` | str  | best_model.pth | 指定推理使用的模型文件 |
| `-s` | int  | 42 | 随机种子 |
| `-bs` | int | 64 | Batch size |
| `-c` | int | 0 | 数据可用性检查：`0=检查`，`1=跳过检查` |

> 注意：`-f` 与 `-F` **不能同时使用**。

---

### 基础使用示例

#### **① 使用单个 CSV 文件进行预测**
```bash
python MPFF-IS.py \
    -f data/sample.csv \
    -m best_model.pth \
    -bs 64 \
    -s 42 \
    -c 0
```

#### **② 使用包含多个 CSV 文件的文件夹进行预测**
```bash
python MPFF-IS.py \
    -F data/folder/ \
    -m best_model.pth \
    -bs 64 \
    -s 42 \
    -c 0
