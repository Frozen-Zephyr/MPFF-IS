# MPFF-IS
A Molecular–Protein Fusion Framework for Discovery of KRAS G12D Inhibitors in Pancreatic Ductal Adenocarcinoma


## 环境设置与依赖安装

请确保以下依赖版本一致：

- **DGL (Deep Graph Library)**：2.4.0（需 CUDA 12.1 支持）  
- **DGL-LifeSci**：0.3.2  
- **scikit-learn**：1.6.1  

建议使用 Conda 创建独立环境：

```bash
conda create -n mpff-is python=3.9
conda activate mpff-is
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

从huggingface上下载esm2_t33_650M_UR50D 模型文件到本地：
https://huggingface.co/facebook/esm2_t33_650M_UR50D




