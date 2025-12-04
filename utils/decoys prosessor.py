from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import os
from functools import partial
import csv
from tqdm import tqdm  # 进度条


MAX_ROWS = 1000000  # CSV 文件的最大行数

def get_new_filename(base_name):
    """自动寻找新的文件名（编号递增）"""
    i = 1
    new_filename = f"{base_name}_{i}.csv"
    while os.path.exists(new_filename):
        df_existing = pd.read_csv(new_filename)
        if len(df_existing) < MAX_ROWS:
            return new_filename  # 发现文件未超限，继续使用
        i += 1
        new_filename = f"{base_name}_{i}.csv"
    return new_filename  # 找到一个新的未使用的文件名



# 计算理化性质
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "TPSA": round(Descriptors.TPSA(mol), 2)
        }
    return None


def tanimoto(smiles1, smiles2):
    # 将SMILES转换为RDKit分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string")

    # 计算两分子的指纹（默认是RDKit指纹）
    fp1 = RDKFingerprint(mol1)
    fp2 = RDKFingerprint(mol2)

    # 计算Tanimoto系数
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity


def to_csv(pro_id,pro_name,pro_seq,std_typ,std_vlu,std_unt,smiles,target,decoys_lst,output_csv,lig_id,BDB):
    output_csv = get_new_filename(output_csv)

    activate_data = pd.DataFrame([{
        'Protein ChEMBL ID': pro_id,
        'Protein_name': pro_name,
        'Protein Sequence': pro_seq,
        'ligand ChEMBL ID': lig_id,
        'SMILES': smiles,
        'Standard Type': std_typ,
        'Standard Value': std_vlu,
        'Standard Units': std_unt,
        'BindingDB Entry DOI': BDB,
        'Label': target,

    }])

    # 构造 Decoys 数据（虚假配体）
    decoy_data = pd.DataFrame([{
        'Protein ChEMBL ID': pro_id,
        'Protein_name': pro_name,
        'Protein Sequence': pro_seq,
        'ligand ChEMBL ID':'N/A',
        'SMILES': decoy,
        'Standard Type': 'N/A',
        'Standard Value': 'N/A',
        'Standard Units': 'N/A',
        'BindingDB Entry DOI': 'N/A',
        'Label': 0,
    } for decoy in decoys_lst])

    total_data = pd.concat([activate_data, decoy_data], ignore_index=True)

    # 追加数据到 CSV，首次写入文件时加表头，否则不加
    total_data.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))

    # # 读取已有文件，或创建新文件
    # if os.path.exists(output_csv):
    #     df = pd.read_csv(output_csv)
    #     df = pd.concat([df, activate_data, decoy_data], ignore_index=True)
    # else:
    #     df = pd.concat([activate_data, decoy_data], ignore_index=True)
    #
    # # 保存 CSV
    # df.to_csv(output_csv, index=False)



import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_one(row, ibs_data):
    # 这里保持不变
    lig_properties = compute_properties(row["SMILES"])
    if lig_properties is None:
        return None
    n, decoys_lst = 0, []
    for line in ibs_data:
        try:
            mw_value = float(line["MW"])
            logp_value = float(line["LogP"])
            hba_value = float(line["HBA"])
            hbd_value = float(line["HBD"])
            tpsa_value = float(line["TPSA"])
        except ValueError:
            continue
        if (abs(mw_value - lig_properties["MW"]) < 0.5 and
            abs(logp_value - lig_properties["LogP"]) < 0.5 and
            hba_value == lig_properties["HBA"] and
            hbd_value == lig_properties["HBD"] and
            abs(tpsa_value - lig_properties["TPSA"]) < 10 and
            tanimoto(line['SMILES'], row["SMILES"]) < 0.75):
            n += 1
            decoys_lst.append(line["SMILES"])
        if n > 224:
            break
    if n >= 0:
        return [
            row["Protein ChEMBL ID"],
            row["Protein Name"],
            row["Protein_sequence"],
            row["Standard Type"],
            row["Standard Value"],
            row["Standard Units"],
            row["SMILES"],
            1,
            ";".join(decoys_lst),
            row["Molecule ChEMBL ID"],
            row["BindingDB Entry DOI"]
        ]
    return None


def find_decoys(ibs_bank,
                active_csv,
                output_csv
                ):
    with open(active_csv, "r",encoding="utf-8-sig") as f_lig:
        reader1 = list(csv.DictReader(f_lig) ) # 以字典形式读取，自动识别表头
        for row in tqdm(reader1):
            pro_id = (row["Protein ChEMBL ID"])
            pro_name = (row["Protein Name"])
            pro_seq = (row["Protein_sequence"])
            lig_id = (row["Molecule ChEMBL ID"])
            std_typ = (row["Standard Type"])
            std_vlu = (row["Standard Value"])
            std_unt = (row["Standard Units"])
            smiles = (row["SMILES"])
            BDB=(row["BindingDB Entry DOI"])
            target = 1
            lig_properties=compute_properties(smiles)
            if lig_properties is None:
                continue
            n = 0
            decoys_lst = []

            for file in ibs_bank:
                with open(file, "r",encoding="utf-8-sig") as f_ibs:
                    reader2 = list(csv.DictReader(f_ibs))
                    for line in reader2:
                        try:
                            mw_value = float(line["MW"])
                            logp_value = float(line["LogP"])
                            hba_value = float(line["HBA"])
                            hbd_value = float(line["HBD"])
                            tpsa_value = float(line["TPSA"])
                        except ValueError:
                            continue
                        if (abs(mw_value - lig_properties["MW"]) < 0.5 and  # 分子量相近
                            abs(logp_value - lig_properties["LogP"])< 0.5 and  # LogP 相近
                            hba_value == lig_properties["HBA"]  and  # 氢键受体相同
                            hbd_value == lig_properties["HBD"]  and  # 氢键供体相同
                            abs(tpsa_value - lig_properties["TPSA"]) < 10  and  # 极性表面积相近
                            tanimoto(line['SMILES'],smiles) < 0.75):
                            n += 1
                            decoys_lst.append(line["SMILES"])
                        if n>224:
                            break
                    if n>=0:
                        to_csv(pro_id,pro_name,pro_seq,std_typ,std_vlu,std_unt,smiles,target,decoys_lst,output_csv,lig_id,BDB)
                    else:
                        continue

# 你的活性分子 CSV 文件
active_csv = "/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/kras data/single/single_positive.csv"
# 你的 ZINC Decoy 库 SMI 文件
ibs_bank = ["/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/IBS/ibs_all.csv"]
# 输出最终匹配的 Decoy 分子
output_csv = "/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/kras data/single/single_decoys"


find_decoys(ibs_bank,active_csv,output_csv)


