import requests
import pandas as pd
import os
from tqdm import tqdm
from time import sleep

# BindingDB 数据下载 API（可调整）
BINDINGDB_URL = "https://www.bindingdb.org/bind/BindingDB_All.tsv"

# 保存的 CSV 文件
OUTPUT_FILE = "bindingdb_data.csv"

# 需要的列名
COLUMNS = [
    "Target Name",  # 蛋白靶标名称
    "Target Sequence",  # 蛋白靶标氨基酸序列
    "Ligand Name",  # 化合物配体名称
    "SMILES",  # 化合物配体 SMILES
    "IC50 relation",  # IC50 关系（<=, >=, =）
    "IC50 (nM)"  # IC50 值
]


def download_bindingdb(output_file):
    """下载 BindingDB 数据并保存到 CSV 文件"""
    # 先检查是否已有下载
    if os.path.exists(output_file):
        print(f"文件 {output_file} 已存在，跳过下载。")
        return

    print("正在下载 BindingDB 数据...")
    response = requests.get(BINDINGDB_URL, stream=True)

    if response.status_code != 200:
        print("下载失败，请检查 URL 或网络连接。")
        return

    # 计算总大小
    total_size = int(response.headers.get('Content-Length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open("BindingDB_All.tsv", "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            sleep(0.01)
    progress_bar.close()

    print("下载完成！")


def process_bindingdb(input_file, output_file):
    """解析 TSV 文件，提取关键字段并保存 CSV"""
    df = pd.read_csv(input_file, sep='\t', low_memory=False, on_bad_lines='skip')

    # 过滤所需字段
    df_filtered = df[COLUMNS].dropna()

    df_filtered = df_filtered.rename(columns={
        "Target Name": "Target_Name",
        "Target Sequence": "Target_Sequence",
        "Ligand Name": "Ligand_Name",
        "SMILES": "SMILES",
        "IC50 relation": "IC50_Relation",
        "IC50 (nM)": "IC50_Value"
    })

    df_filtered.to_csv(output_file, index=False)
    print(f"数据处理完成，已保存到 {output_file}")



                # 下载数据
download_bindingdb("BindingDB_All.tsv")

                # 处理数据
process_bindingdb("", OUTPUT_FILE)
