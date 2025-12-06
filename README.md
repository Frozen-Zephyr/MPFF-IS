# MPFF-IS
A Molecular–Protein Fusion Framework for Discovery of KRAS G12D Inhibitors in Pancreatic Ductal Adenocarcinoma


## Environment setup and dependency installation

请确保以下依赖版本一致：

- **DGL (Deep Graph Library)**：2.4.0（需 CUDA 12.1 支持）  
- **DGL-LifeSci**：0.3.2  
- **scikit-learn**：1.6.1  

It is recommended to use Conda to create a standalone environment:

```bash
pip install dgl-cu121==2.4.0
pip install dgllife==0.3.2
pip install scikit-learn==1.6.1
```

## Quick Start

### 1. Cloning repository

```bash
git clone https://github.com/Frozen-Zephyr/MPFF-IS.git
cd MPFF-IS
```

### 2. Download model file

从Hugging Face上下载 esm2_t33_650M_UR50D.pt 模型文件到MPFF-IS文件夹：
https://huggingface.co/facebook/esm2_t33_650M_UR50D



### 3. Run code

Enter the environment directly and run MPFF-IS.py:
```bash
python MPFF-IS.py -f data/sample.csv  -m best_model.pth 
```

## Usage

The following is an explanation of the command-line parameters and example usage of `MPFF-IS`.

### CSV input file instructions
The application must have SMILES and Protein_sequence columns; other columns are optional.
Multiple files can be entered, in the form of folders.

### Parameter Description

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `-f` | str  |  | Input single CSV file path |
| `-F` | str  |  | Input folder path containing multiple CSV files (cannot be used with `-f`) |
| `-m` | str  | best_model.pth | Specify the model file used for inference |
| `-s` | int  | 42 | Random seed |
| `-bs` | int | 64 | Batch size |
| `-c` | int | 0 | Data availability check: `0=check`, `1=skip check` |

> Note: `-f` and `-F` **cannot be used simultaneously**.

---

### Basic usage example

#### **① Predict using a single CSV file**
```bash
python MPFF-IS.py \
    -f data/sample.csv \
    -m best_model.pth \
    -bs 64 \
    -s 42 \
    -c 0
```

#### **② Predict using a folder containing multiple CSV files**
```bash
python MPFF-IS.py \
    -F data/folder/ \
    -m best_model.pth \
    -bs 64 \
    -s 42 \
    -c 0
```

## Result Description

The results will be output to the MPFF-IS/results folder, named after the input file.



