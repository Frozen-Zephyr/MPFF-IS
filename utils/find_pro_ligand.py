from tqdm import tqdm
import requests 
import pandas as pd
import os
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 设置 API 网址
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
# 设置请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# 创建 Session 并添加重试机制
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_targets(web_limit,total,out_path,target_web_limit,total_visit,ic50_standard_value,min_ic50_lig):
    limit=web_limit
    total_visit=total_visit
    n=0
    flag=''
    for offset in range(0, total_visit, limit):
        target_url = f"{BASE_URL}/target.json?limit={limit}&offset={offset}"
        # 处理请求错误
        try:
            response = session.get(target_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            targets = response.json().get("targets", [])
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            time.sleep(5)  # 失败时等待 5 秒后重试
            continue

     # 遍历所有蛋白靶点
        for target in tqdm(targets, desc="Processing targets", unit="target"):

            ligands = []
            target_id = target["target_chembl_id"]
            protein_name = target.get("pref_name", "Unknown Protein")

          # 查询该蛋白的活性化合物数量
            activity_url = f"{BASE_URL}/activity.json?target_chembl_id={target_id}&limit={target_web_limit}"

            try:
                response = session.get(activity_url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                activities = response.json().get("activities", [])
            except requests.exceptions.RequestException as e:
                print(f"请求失败: {e}")
                time.sleep(5)
                continue

            if  not activities:
                break
            for activity in activities:
                if activity.get("standard_type") == "IC50" and activity.get("standard_value") is not None:
                    value = activity.get("standard_value")
                    if value is not None:
                        ic50_value = float(value)
                        if ic50_value <= ic50_standard_value:
                            ligands.append({"Protein Name": protein_name,
                                            'Protein ChEMBL ID': target_id,
                                            "Molecule ChEMBL ID": activity.get("molecule_chembl_id"),
                                            "Standard Type": activity.get("standard_type"),
                                            "Standard Value": activity.get("standard_value"),
                                            "Standard Units": activity.get("standard_units"),
                                            'SMILES': activity.get("canonical_smiles")
                                            })  # 添加到列表中

            if 224>=len(ligands) >= min_ic50_lig:
                to_csv(ligands, out_path)
                n += len(ligands)
                time.sleep(0.5)
                print('已找到{}组蛋白-配体'.format(n))
    # 达到 224 个配体时停止
            elif len(ligands)>224:
                to_csv(ligands[:224], out_path)
                n+=224
                time.sleep(0.5)
                print('已找到{}组蛋白-配体'.format(n))
            else:
                continue

            if n>=total:
                print('查找完毕：找到{}组蛋白-配体'.format(n))
                flag='end'
                break
        if flag=='end':
            break


def to_csv(data, output_csv):
    columns= ["Protein Name",'Protein ChEMBL ID', "Molecule ChEMBL ID", "Standard Type", "Standard Value", "Standard Units", "SMILES"]
    df_new = pd.DataFrame(data, columns=columns)
    # 读取已有文件，或创建新文件
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
       try:
        df_old = pd.read_csv(output_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
       except pd.errors.EmptyDataError:
        df = df_new
    else:
        df=df_new

    df.to_csv(output_csv, index=False)


out_path="target6.1.csv"
get_targets(web_limit=1000,
            total=120000,
            out_path=out_path,
            target_web_limit=1000,
            total_visit=99999999999,
            ic50_standard_value=6.1,
            min_ic50_lig=10
            )




