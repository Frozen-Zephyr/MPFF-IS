import requests
import csv
import time
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

input_csv = "csv/positive_ligands_nm.csv"
output_csv = "csv/protein_secquence.csv"

# UniProt API 查询 URL
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/search?query={}&format=json"

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

def get_protein_sequence(protein_name):
    """从 UniProt 获取蛋白质序列"""
    url = UNIPROT_API_URL.format(protein_name)
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["sequence"]["value"]  # 提取序列
        else:
            return "未找到序列"
    else:
        return "请求失败"

# 读取 CSV 获取蛋白质名称列表
with open(input_csv, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    protein_names = [row['Protein Name'].strip() for row in reader if row]
    protein_name_unre=[]
    for i in protein_names:
        if i not in protein_name_unre:
            protein_name_unre.append(i)
        else:
            continue

# 打开 CSV 文件写入结果
with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["pro_name", "pro_seq"])  # 写入标题
    

    for protein_name in tqdm(protein_name_unre[238:], desc="查询蛋白质序列", unit="项"):
        sequence = get_protein_sequence(protein_name)
        writer.writerow([protein_name, sequence])
        time.sleep(0.5)  # 避免 API 访问过快

print(f"查询完成，结果已保存至 {output_csv}。")
