import numpy as np
from matplotlib import pyplot as plt
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from difflib import SequenceMatcher
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors, Lipinski
from sklearn.model_selection import StratifiedKFold

def compute_properties(smiles):
    """计算分子理化性质"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MW": round(Descriptors.MolWt(mol), 2),  # 分子量
            "LogP": round(Descriptors.MolLogP(mol), 2),  # LogP
            "HBA": Descriptors.NumHAcceptors(mol),  # 氢键受体数
            "HBD": Descriptors.NumHDonors(mol),  # 氢键供体数
            "TPSA": round(Descriptors.TPSA(mol), 2)  # 极性表面积
        }
    return None


def dealing_csv(active_df,output_csv):
    '''处理虚假配体库'''
    # 处理 CSV 文件
    results = []
    for smiles in tqdm(active_df['SMILES']):
        properties = compute_properties(smiles)
        if properties:
            results.append([smiles, properties["MW"], properties["LogP"], properties["HBA"], properties["HBD"], properties["TPSA"]])
        else:
            results.append([smiles, "Invalid", "Invalid", "Invalid", "Invalid", "Invalid"])

    columns = ["SMILES", "MW", "LogP", "HBA", "HBD", "TPSA"]
    pd.DataFrame(results, columns=columns).to_csv(output_csv, index=False)

    print(f"理化性质计算完成，结果已保存至 {output_csv}")


def DUDE(input_file,output_file):
    '''删除少于五个正样本的蛋白（DUDE）'''
    df = pd.read_csv(input_file,low_memory=False)
    positive_counts = df[df["Label"] == 1].groupby("Protein Name").size()
    proteins_to_remove = positive_counts[positive_counts < 5].index
    filtered_df = df[~df["Protein Name"].isin(proteins_to_remove)]
    filtered_df.to_csv(output_file, index=False)
    print(f"筛选后数据已保存至 {output_file}")


def count_labels(*csv_files):
    '''计算正负样本数量'''
    count_0=0
    count_1=0
    count_total=0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file,low_memory=False)
        if 'Label' not in df.columns:
            raise ValueError("CSV 文件中没有 'Label' 列")
        count_0 += (df['Label'] == 0).sum()
        count_1 += (df['Label'] == 1).sum()
    count_total += count_0 + count_1

    ratio = count_0 / count_1 if count_1 > 0 else float('inf')

    print('一共数据有：{}'.format(count_total))
    print(f"负样本有: {count_0}")
    print(f"正样本有: {count_1}")
    print(f"负样本数是正样本的 {ratio:.2f} 倍")


def count_proteins(*file_path):
    '''计算蛋白数量'''
    unique_proteins=0
    for csv_file in file_path:
        df = pd.read_csv(csv_file,low_memory=False)
        if 'Protein Name' not in df.columns:
            raise ValueError("CSV文件中未找到 'Protein Name' 列")
        unique_proteins += df['Protein Name'].nunique()
    print('有{}个蛋白'.format(unique_proteins))


def is_similar(seq1, seq2, threshold=0.8):
    """判断两个蛋白质序列是否相似，使用相似度阈值"""
    return SequenceMatcher(None, seq1, seq2).ratio() >= threshold


def remove_duplicates(file1, file2, output_file):
    '''正样本间查重'''
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    if 'SMILES' not in df1.columns or 'Protein Sequence' not in df1.columns:
        raise ValueError("CSV文件1中缺少 'SMILES' 或 'Protein Sequence' 列")
    if 'SMILES' not in df2.columns or 'Protein_sequence' not in df2.columns:
        raise ValueError("CSV文件2中缺少 'SMILES' 或 'Protein Sequence' 列")

    duplicates = set()

    for i, row1 in tqdm(df1.iterrows(), total=len(df1), desc="Processing File 1"):
        for j, row2 in df2.iterrows():
            if row1['SMILES'] == row2['SMILES'] and is_similar(row1['Protein Sequence'], row2['Protein Sequence']):
                duplicates.add(j)  # 记录df2中重复的行索引

    df2_filtered = df2.drop(index=list(duplicates))
    merged_df = pd.concat([df1, df2_filtered], ignore_index=True)

    merged_df.to_csv(output_file, index=False)
    print(f'去重后的数据已保存至 {output_file}')


def save_large_csv(data, base_filename, max_rows=1_000_000):
    """当 CSV 超过 max_rows 时，拆分多个文件"""
    num_parts = (len(data) // max_rows) + (1 if len(data) % max_rows != 0 else 0)
    for i in range(num_parts):
        start = i * max_rows
        end = start + max_rows
        filename = f"{base_filename}_{i + 1}.csv" if num_parts > 1 else f"{base_filename}.csv"
        data.iloc[start:end].to_csv(filename, index=False)


def split_csv(*file_paths, label_col='Label', train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    '''多个csv文件数据集分割'''
    total_data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path,low_memory=False)
        total_data.append(df)

    df = pd.concat(total_data, ignore_index=True)  # 合并数据

    train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), stratify=df[label_col], random_state=42)
    valid_size = valid_ratio / (valid_ratio + test_ratio)
    valid_data, test_data = train_test_split(temp_data, test_size=(1 - valid_size), stratify=temp_data[label_col],
                                             random_state=42)

    save_large_csv(train_data, "/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/kras data/decoys/decoys+负样本/krasdecoys_train.csv",)
    save_large_csv(valid_data, "/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/kras data/decoys/decoys+负样本/krasdecoys_valid.csv")
    save_large_csv(test_data, "/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/kras data/decoys/decoys+负样本/krasdecoys_test.csv")

    print("数据集拆分完成！")


def Tanimoto(input_path, output_path):
    df = pd.read_csv(input_path)
    smiles_list = df["SMILES"]

    # 两个参考化合物的 SMILES
    MRTX133 = Chem.MolFromSmiles("C#Cc1c(ccc2c1c(cc(c2)O)c3c(c4c(cn3)c(nc(n4)OC[C@@]56CCCN5C[C@@H](C6)F)N7C[C@H]8CC[C@@H](C7)N8)F)F")  # 替换为你的参考化合物1
    BI_2852 = Chem.MolFromSmiles("c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@@](=O)(CP(=O)(O)O)O)O)O)N=C(NC2=O)N")  # 替换为你的参考化合物2

    # 生成参考化合物的指纹
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp_ref1 = AllChem.GetMorganFingerprintAsBitVect(MRTX133, radius=2, nBits=2048)
    fp_ref2 = AllChem.GetMorganFingerprintAsBitVect(BI_2852, radius=2, nBits=2048)

    # 计算每个化合物与参考化合物的相似度
    similarities_1 = []
    similarities_2 = []

    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            sim1 = DataStructs.TanimotoSimilarity(fp, fp_ref1)
            sim2 = DataStructs.TanimotoSimilarity(fp, fp_ref2)
        else:
            sim1, sim2 = None, None
        similarities_1.append(round(sim1,5))
        similarities_2.append(round(sim2,5))

    df["similarity_to_MRTX1133"] = similarities_1
    df["similarity_to_BI2852"] = similarities_2
    df.to_csv(output_path, index=False)


def fix_pdb(input_pdb, output_pdb, ph=7.0):
    '''修补蛋白质氨基酸原子'''
    fixer = PDBFixer(filename=input_pdb)

    print("Finding missing residues and atoms...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    print("Adding hydrogens...")
    fixer.addMissingHydrogens(pH=ph)

    print(f"Saving fixed PDB to: {output_pdb}")
    with open(output_pdb, 'w') as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, out)


def smile_to_pdb(smiles,out):
    smiles = smiles
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, out)


def fold(input_csv,output_dir):
    n_splits = 5

    df = pd.read_csv(input_csv)
    if "Label" not in df.columns:
        raise ValueError("CSV 文件必须包含 'Label' 列")

    X = df.drop(columns=["Label"])
    y = df["Label"]

    os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

        print(f"Fold {i}: train {len(train_df)}, val {len(val_df)} saved to {fold_dir}")


def lipinski(input_csv,output_csv):
    df = pd.read_csv(input_csv)

    def calc_props(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # 如果SMILES无法解析，返回NaN
            return [None] * 4
        mw = Descriptors.MolWt(mol)  # 分子量
        logp = Descriptors.MolLogP(mol)  # LogP
        hbd = Lipinski.NumHDonors(mol)  # 氢键供体
        hba = Lipinski.NumHAcceptors(mol)  # 氢键受体
        return [mw, logp, hbd, hba]

    df[['MolWt', 'LogP', 'HBD', 'HBA']] = df['SMILES'].apply(
        lambda x: pd.Series(calc_props(x))
    )

    df.to_csv(output_csv, index=False)
    print(f"计算完成，结果已保存到 {output_csv}")


def IQR(input_csv):
    df = pd.read_csv(input_csv)

    cols = ["MolWt", "LogP", "HBD", "HBA"]

    print("中位数 ± 1.5×IQR 计算结果：")
    for col in cols:
        median = df[col].median()
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = median - 1.5 * IQR
        upper = median + 1.5 * IQR
        print(f"{col}: 中位数 = {median:.3f}, 下界 = {lower:.3f}, 上界 = {upper:.3f}")


RDLogger.DisableLog('rdApp.*')
def read_csv_ignore_encoding(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")  # 备用编码

def process_csv(file_path, output_dir, min_MolWt, max_MolWt, min_LogP, max_LogP,
                min_HBD, max_HBD, min_HBA, max_HBA):
    df = read_csv_ignore_encoding(file_path)  # 自定义函数处理编码问题
    filtered_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(file_path), leave=True):
        smi = row["smile"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        MolWt = Descriptors.MolWt(mol)
        LogP = Descriptors.MolLogP(mol)
        HBD = Descriptors.NumHDonors(mol)
        HBA = Descriptors.NumHAcceptors(mol)

        if (min_MolWt <= MolWt <= max_MolWt and
            min_LogP <= LogP <= max_LogP and
            min_HBD <= HBD <= max_HBD and
            min_HBA <= HBA <= max_HBA):
            filtered_rows.append(row)

    if filtered_rows:  # 只有非空才保存
        out_df = pd.DataFrame(filtered_rows)
        out_file = os.path.join(output_dir, os.path.basename(file_path))
        out_df.to_csv(out_file, index=False)
        print(f"{os.path.basename(file_path)} saved with {len(filtered_rows)} rows")
    else:
        print(f"{os.path.basename(file_path)} skipped, 0 rows kept")

    return os.path.basename(file_path), len(filtered_rows)


def filter_folder(input_dir, output_dir, batch_size=50):
    os.makedirs(output_dir, exist_ok=True)

    # 自定义筛选范围
    min_MolWt, max_MolWt = 550, 1500
    min_LogP, max_LogP = 4, 8
    min_HBD, max_HBD = 1, 4
    min_HBA, max_HBA = 7, 13

    # 按文件名排序
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")])
    num_workers = multiprocessing.cpu_count()

    results = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_csv, f, output_dir,
                                       min_MolWt, max_MolWt,
                                       min_LogP, max_LogP,
                                       min_HBD, max_HBD,
                                       min_HBA, max_HBA)
                       for f in batch_files]
            for f in futures:
                results.append(f.result())

    print("处理完成。")
    for fname, n in results:
        print(f"{fname}: {n} rows kept")

def similarity(positive_csv,experimental_csv,output_csv):
    def calculate_fingerprints(mol, fp_type='morgan', radius=2, n_bits=2048):
        """计算分子指纹"""
        if mol is None:
            return None

        try:
            if fp_type == 'morgan':
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == 'maccs':
                return AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == 'rdkit':
                return Chem.RDKFingerprint(mol)
            else:
                raise ValueError(f"不支持的指纹类型: {fp_type}")
        except Exception as e:
            print(f"计算指纹时出错: {e}")
            return None

    def load_samples(csv_file, smiles_col='SMILES', id_col='ID'):
        """加载样本数据并计算三种指纹"""
        print(f"正在加载数据: {csv_file}")
        df = pd.read_csv(csv_file)

        molecules = []
        fingerprints = {
            'morgan': [],
            'maccs': [],
            'rdkit': []
        }
        ids = []
        valid_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="加载分子"):
            smiles = row[smiles_col]
            mol_id = row[id_col]

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                molecules.append(mol)
                ids.append(mol_id)

                # 计算三种指纹
                for fp_type in fingerprints.keys():
                    fp = calculate_fingerprints(mol, fp_type)
                    fingerprints[fp_type].append(fp)

                valid_count += 1
            else:
                print(f"警告: 无法解析SMILES: {smiles} (ID: {mol_id})")

        print(f"成功加载 {valid_count}/{len(df)} 个分子")
        return molecules, fingerprints, ids

    def calculate_similarities(exp_fingerprints, pos_fingerprints, exp_ids, pos_ids, similarity_threshold=0.7):
        """计算三种指纹类型的相似度"""
        results = {
            'morgan': {'similar_ids': [], 'counts': [], 'max_similarities': []},
            'maccs': {'similar_ids': [], 'counts': [], 'max_similarities': []},
            'rdkit': {'similar_ids': [], 'counts': [], 'max_similarities': []}
        }

        print("正在计算三种指纹类型的相似度...")

        # 对每种指纹类型分别计算
        for fp_type in exp_fingerprints.keys():
            print(f"计算 {fp_type.upper()} 指纹相似度...")

            for i, exp_fp in enumerate(tqdm(exp_fingerprints[fp_type], desc=f"{fp_type}指纹")):
                if exp_fp is None:
                    # 如果指纹计算失败，添加空结果
                    results[fp_type]['similar_ids'].append("")
                    results[fp_type]['counts'].append(0)
                    results[fp_type]['max_similarities'].append(0)
                    continue

                similar_ids = []
                max_similarity = 0

                # 与所有正样本比较
                for j, pos_fp in enumerate(pos_fingerprints[fp_type]):
                    if pos_fp is None:
                        continue

                    similarity = DataStructs.TanimotoSimilarity(exp_fp, pos_fp)

                    if similarity > max_similarity:
                        max_similarity = similarity

                    if similarity >= similarity_threshold:
                        similar_ids.append(f"{pos_ids[j]}({similarity:.3f})")

                results[fp_type]['similar_ids'].append("; ".join(similar_ids))
                results[fp_type]['counts'].append(len(similar_ids))
                results[fp_type]['max_similarities'].append(max_similarity)

        return results

    def add_results_to_dataframe(df, results):
        """将结果添加到DataFrame"""
        for fp_type in results.keys():
            df[f'Similar_Positive_IDs_{fp_type.upper()}'] = results[fp_type]['similar_ids']
            df[f'Similarity_Count_{fp_type.upper()}'] = results[fp_type]['counts']
            df[f'Max_Similarity_{fp_type.upper()}'] = results[fp_type]['max_similarities']

        return df

    def generate_statistics(df):
        """生成统计信息"""
        print("\n" + "=" * 60)
        print("三种指纹类型相似度分析统计")
        print("=" * 60)

        total_compounds = len(df)

        for fp_type in ['morgan', 'maccs', 'rdkit']:
            count_col = f'Similarity_Count_{fp_type.upper()}'
            max_sim_col = f'Max_Similarity_{fp_type.upper()}'

            compounds_with_similar = len(df[df[count_col] > 0])
            total_similarities = df[count_col].sum()

            print(f"\n{fp_type.upper()}指纹:")
            print(
                f"  具有相似正样本的化合物数: {compounds_with_similar}/{total_compounds} ({compounds_with_similar / total_compounds * 100:.1f}%)")
            print(f"  总相似对数量: {total_similarities}")
            print(f"  平均每个化合物的相似正样本数: {total_similarities / total_compounds:.2f}")

            # 相似度分布
            high_sim_count = len(df[df[max_sim_col] >= 0.95])
            med_sim_count = len(df[(df[max_sim_col] >= 0.9) & (df[max_sim_col] < 0.95)])

            print(f"  高相似度(≥0.95): {high_sim_count} 个化合物")
            print(f"  中等相似度(0.9-0.95): {med_sim_count} 个化合物")

        # 显示最相似的化合物（按Morgan指纹）
        top_similar = df.nlargest(5, 'Max_Similarity_MORGAN')[['ID', 'Max_Similarity_MORGAN', 'Similarity_Count_MORGAN']]
        print(f"\n最相似的前5个化合物 (按Morgan指纹):")
        for _, row in top_similar.iterrows():
            print(
                f"  ID: {row['ID']}, 最大相似度: {row['Max_Similarity_MORGAN']:.3f}, 相似正样本数: {row['Similarity_Count_MORGAN']}")

    def similar(positive_csv,experimental_csv,output_csv):

        try:
            # 1. 加载正样本并计算三种指纹
            pos_molecules, pos_fingerprints, pos_ids = load_samples(positive_csv)

            # 2. 加载实验样本并计算三种指纹
            exp_molecules, exp_fingerprints, exp_ids = load_samples(experimental_csv)

            # 3. 计算相似度
            results = calculate_similarities(exp_fingerprints, pos_fingerprints, exp_ids, pos_ids, similarity_threshold=0.9)

            # 4. 加载原始实验样本数据框
            df_experimental = pd.read_csv(experimental_csv)

            # 5. 添加结果到数据框
            df_experimental = add_results_to_dataframe(df_experimental, results)

            # 6. 生成统计信息
            generate_statistics(df_experimental)

            # 7. 保存结果
            df_experimental.to_csv(output_csv, index=False)
            print(f"\n结果已保存到: {output_csv}")

            # 8. 显示高度相似的分子对
            print(f"\n高度相似分子对 (相似度 ≥ 0.95):")
            for fp_type in ['morgan', 'maccs', 'rdkit']:
                count_col = f'Similarity_Count_{fp_type.upper()}'
                high_sim_count = len(df_experimental[df_experimental[count_col] > 0])
                print(f"  {fp_type.upper()}指纹: {high_sim_count} 个化合物有高度相似正样本")

        except Exception as e:
            print(f"发生错误: {e}")

    similar(positive_csv,experimental_csv,output_csv)

def distance(positive_csv, experimental_csv, output_csv):

    def calculate_fingerprint(mol, fp_type='morgan', radius=2, n_bits=2048):
        """计算单个分子指纹"""
        if mol is None:
            return None

        try:
            if fp_type == 'morgan':
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == 'maccs':
                return AllChem.GetMACCSKeysFingerprint(mol)
            elif fp_type == 'rdkit':
                return Chem.RDKFingerprint(mol)
            else:
                raise ValueError(f"不支持的指纹类型: {fp_type}")
        except Exception as e:
            print(f"计算指纹时出错: {e}")
            return None

    def load_and_process_samples(csv_file, smiles_col='SMILES', id_col='ID'):
        """加载样本数据并计算三种指纹"""
        print(f"正在加载数据: {csv_file}")
        df = pd.read_csv(csv_file)

        results = {
            'ids': [],
            'molecules': [],
            'fingerprints': {
                'morgan': [],
                'maccs': [],
                'rdkit': []
            }
        }

        valid_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="加载分子"):
            smiles = row[smiles_col]
            mol_id = row[id_col]

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                results['ids'].append(mol_id)
                results['molecules'].append(mol)

                # 计算三种指纹
                for fp_type in results['fingerprints'].keys():
                    fp = calculate_fingerprint(mol, fp_type)
                    results['fingerprints'][fp_type].append(fp)

                valid_count += 1
            else:
                print(f"警告: 无法解析SMILES: {smiles} (ID: {mol_id})")

        print(f"成功加载 {valid_count}/{len(df)} 个分子")
        return results, df

    def calculate_similarity_coordinates(exp_data, pos_data):
        """计算每个样本的三维相似度坐标"""
        print("正在计算三维相似度坐标...")

        # 为实验样本计算坐标
        exp_coordinates = []

        for i, exp_mol in enumerate(tqdm(exp_data['molecules'], desc="计算实验样本坐标")):
            if exp_mol is None:
                exp_coordinates.append([0, 0, 0])
                continue

            # 计算三种指纹
            exp_fp_morgan = calculate_fingerprint(exp_mol, 'morgan')
            exp_fp_maccs = calculate_fingerprint(exp_mol, 'maccs')
            exp_fp_rdkit = calculate_fingerprint(exp_mol, 'rdkit')

            # 计算与正样本的平均相似度作为坐标
            morgan_sims = []
            maccs_sims = []
            rdkit_sims = []

            for j, pos_mol in enumerate(pos_data['molecules']):
                if pos_mol is None:
                    continue

                # 计算三种指纹相似度
                pos_fp_morgan = pos_data['fingerprints']['morgan'][j]
                pos_fp_maccs = pos_data['fingerprints']['maccs'][j]
                pos_fp_rdkit = pos_data['fingerprints']['rdkit'][j]

                if exp_fp_morgan is not None and pos_fp_morgan is not None:
                    morgan_sim = DataStructs.TanimotoSimilarity(exp_fp_morgan, pos_fp_morgan)
                    morgan_sims.append(morgan_sim)

                if exp_fp_maccs is not None and pos_fp_maccs is not None:
                    maccs_sim = DataStructs.TanimotoSimilarity(exp_fp_maccs, pos_fp_maccs)
                    maccs_sims.append(maccs_sim)

                if exp_fp_rdkit is not None and pos_fp_rdkit is not None:
                    rdkit_sim = DataStructs.TanimotoSimilarity(exp_fp_rdkit, pos_fp_rdkit)
                    rdkit_sims.append(rdkit_sim)

            # 计算平均相似度作为坐标
            morgan_coord = np.mean(morgan_sims) if morgan_sims else 0
            maccs_coord = np.mean(maccs_sims) if maccs_sims else 0
            rdkit_coord = np.mean(rdkit_sims) if rdkit_sims else 0

            exp_coordinates.append([morgan_coord, maccs_coord, rdkit_coord])

        # 为正样本计算坐标（相对于自身）
        pos_coordinates = []

        for i, pos_mol in enumerate(tqdm(pos_data['molecules'], desc="计算正样本坐标")):
            if pos_mol is None:
                pos_coordinates.append([0, 0, 0])
                continue

            # 计算三种指纹
            pos_fp_morgan = pos_data['fingerprints']['morgan'][i]
            pos_fp_maccs = pos_data['fingerprints']['maccs'][i]
            pos_fp_rdkit = pos_data['fingerprints']['rdkit'][i]

            # 计算与正样本的平均相似度作为坐标
            morgan_sims = []
            maccs_sims = []
            rdkit_sims = []

            for j, other_pos_mol in enumerate(pos_data['molecules']):
                if i == j or other_pos_mol is None:
                    continue

                # 计算三种指纹相似度
                other_fp_morgan = pos_data['fingerprints']['morgan'][j]
                other_fp_maccs = pos_data['fingerprints']['maccs'][j]
                other_fp_rdkit = pos_data['fingerprints']['rdkit'][j]

                if pos_fp_morgan is not None and other_fp_morgan is not None:
                    morgan_sim = DataStructs.TanimotoSimilarity(pos_fp_morgan, other_fp_morgan)
                    morgan_sims.append(morgan_sim)

                if pos_fp_maccs is not None and other_fp_maccs is not None:
                    maccs_sim = DataStructs.TanimotoSimilarity(pos_fp_maccs, other_fp_maccs)
                    maccs_sims.append(maccs_sim)

                if pos_fp_rdkit is not None and other_fp_rdkit is not None:
                    rdkit_sim = DataStructs.TanimotoSimilarity(pos_fp_rdkit, other_fp_rdkit)
                    rdkit_sims.append(rdkit_sim)

            # 计算平均相似度作为坐标
            morgan_coord = np.mean(morgan_sims) if morgan_sims else 0
            maccs_coord = np.mean(maccs_sims) if maccs_sims else 0
            rdkit_coord = np.mean(rdkit_sims) if rdkit_sims else 0

            pos_coordinates.append([morgan_coord, maccs_coord, rdkit_coord])

        return np.array(exp_coordinates), np.array(pos_coordinates)

    def calculate_distances_to_positive_cloud(exp_coordinates, pos_coordinates):
        """计算每个实验样本到正样本坐标云的距离"""
        print("正在计算到正样本坐标云的距离...")

        # 计算正样本坐标云的中心
        pos_center = np.mean(pos_coordinates, axis=0)

        # 计算每个实验样本到正样本中心的距离
        distances_to_center = np.linalg.norm(exp_coordinates - pos_center, axis=1)

        # 计算每个实验样本到最近正样本的距离
        min_distances = []
        for exp_coord in tqdm(exp_coordinates, desc="计算最小距离"):
            distances = np.linalg.norm(pos_coordinates - exp_coord, axis=1)
            min_distances.append(np.min(distances))

        # 计算每个实验样本到正样本云的平均距离
        avg_distances = []
        for exp_coord in tqdm(exp_coordinates, desc="计算平均距离"):
            distances = np.linalg.norm(pos_coordinates - exp_coord, axis=1)
            avg_distances.append(np.mean(distances))

        return distances_to_center, np.array(min_distances), np.array(avg_distances)

    def visualize_coordinates(exp_coordinates, pos_coordinates, exp_ids, pos_ids, top_n=20):
        """可视化三维坐标"""
        # 随机选择部分样本进行可视化，避免过于拥挤
        if len(exp_coordinates) > top_n:
            indices = np.random.choice(len(exp_coordinates), top_n, replace=False)
            exp_coordinates_vis = exp_coordinates[indices]
            exp_ids_vis = [exp_ids[i] for i in indices]
        else:
            exp_coordinates_vis = exp_coordinates
            exp_ids_vis = exp_ids

        if len(pos_coordinates) > top_n:
            indices = np.random.choice(len(pos_coordinates), top_n, replace=False)
            pos_coordinates_vis = pos_coordinates[indices]
            pos_ids_vis = [pos_ids[i] for i in indices]
        else:
            pos_coordinates_vis = pos_coordinates
            pos_ids_vis = pos_ids

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制正样本点
        ax.scatter(pos_coordinates_vis[:, 0], pos_coordinates_vis[:, 1], pos_coordinates_vis[:, 2],
                   c='blue', marker='o', s=50, alpha=0.7, label='正样本')

        # 绘制实验样本点
        ax.scatter(exp_coordinates_vis[:, 0], exp_coordinates_vis[:, 1], exp_coordinates_vis[:, 2],
                   c='red', marker='^', s=50, alpha=0.7, label='实验样本')

        # 添加标签
        for i, (x, y, z) in enumerate(pos_coordinates_vis):
            ax.text(x, y, z, pos_ids_vis[i], fontsize=8)

        for i, (x, y, z) in enumerate(exp_coordinates_vis):
            ax.text(x, y, z, exp_ids_vis[i], fontsize=8)

        ax.set_xlabel('Morgan 相似度')
        ax.set_ylabel('MACCS 相似度')
        ax.set_zlabel('RDKit 相似度')
        ax.set_title('分子相似度三维坐标空间')
        ax.legend()

        plt.tight_layout()
        plt.savefig('molecular_coordinates_3d.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_statistics(df, distances_to_center, min_distances, avg_distances):
        """生成统计信息"""
        print("\n" + "=" * 60)
        print("分子相似度坐标距离分析统计")
        print("=" * 60)

        total_compounds = len(df)

        # 距离统计
        print(f"\n距离统计:")
        print(f"到正样本中心平均距离: {np.mean(distances_to_center):.4f}")
        print(f"到正样本中心距离标准差: {np.std(distances_to_center):.4f}")
        print(f"最小距离: {np.min(distances_to_center):.4f}")
        print(f"最大距离: {np.max(distances_to_center):.4f}")

        # 最近距离统计
        print(f"\n到最近正样本距离统计:")
        print(f"平均最近距离: {np.mean(min_distances):.4f}")
        print(f"最近距离标准差: {np.std(min_distances):.4f}")

        # 平均距离统计
        print(f"\n到正样本平均距离统计:")
        print(f"平均距离: {np.mean(avg_distances):.4f}")
        print(f"平均距离标准差: {np.std(avg_distances):.4f}")

        # 距离分布
        distance_ranges = [
            (0, 0.1, "0.0-0.1"),
            (0.1, 0.2, "0.1-0.2"),
            (0.2, 0.3, "0.2-0.3"),
            (0.3, 0.4, "0.3-0.4"),
            (0.4, 1.0, "0.4+")
        ]

        print(f"\n到正样本中心距离分布:")
        for low, high, label in distance_ranges:
            count = len([d for d in distances_to_center if low <= d < high])
            print(f"  {label}: {count} 个化合物 ({count / total_compounds * 100:.1f}%)")

        # 显示距离最小的化合物
        closest_indices = np.argsort(distances_to_center)[:5]
        print(f"\n距离正样本中心最近的5个化合物:")
        for idx in closest_indices:
            print(f"  ID: {df.iloc[idx]['ID']}, 距离: {distances_to_center[idx]:.4f}")

    def Distance(positive_csv,experimental_csv,output_csv):
        """主函数"""


        try:
            # 1. 加载正样本
            pos_data, pos_df = load_and_process_samples(positive_csv)

            # 2. 加载实验样本
            exp_data, exp_df = load_and_process_samples(experimental_csv)

            # 3. 计算三维坐标
            exp_coordinates, pos_coordinates = calculate_similarity_coordinates(exp_data, pos_data)

            # 4. 计算距离
            distances_to_center, min_distances, avg_distances = calculate_distances_to_positive_cloud(
                exp_coordinates, pos_coordinates)

            # 5. 添加结果到数据框
            exp_df['Distance_To_Center'] = distances_to_center
            exp_df['Min_Distance_To_Positive'] = min_distances
            exp_df['Avg_Distance_To_Positive'] = avg_distances

            # 6. 添加坐标值
            exp_df['Morgan_Coordinate'] = exp_coordinates[:, 0]
            exp_df['MACCS_Coordinate'] = exp_coordinates[:, 1]
            exp_df['RDKit_Coordinate'] = exp_coordinates[:, 2]

            # 7. 生成统计信息
            generate_statistics(exp_df, distances_to_center, min_distances, avg_distances)

            # 8. 可视化
            # visualize_coordinates(exp_coordinates, pos_coordinates, exp_data['ids'], pos_data['ids'])

            # 9. 保存结果
            exp_df.to_csv(output_csv, index=False)
            print(f"\n结果已保存到: {output_csv}")

        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

    Distance(positive_csv, experimental_csv, output_csv)

def smile23d(csv_path):
    # 输出文件夹
    output_dir = os.path.join(os.path.dirname(csv_path), "pdb_output")
    os.makedirs(output_dir, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 检查列名
    if not {"ID", "SMILES"}.issubset(df.columns):
        raise ValueError("CSV 文件必须包含列名 'ID' 和 'SMILES'")

    # 转换为 3D PDB
    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        mol_id = str(row["ID"])
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[跳过] {mol_id}: 无效 SMILES")
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            pdb_path = os.path.join(output_dir, f"{mol_id}.pdb")
            Chem.MolToPDBFile(mol, pdb_path)
            print(f"[成功] {mol_id} → {pdb_path}")
        except Exception as e:
            print(f"[失败] {mol_id}: {e}")

    print(f"\n✅ 全部完成，输出路径: {output_dir}")



input_path = ''
output_path = ''

if __name__ == "__main__":
    while True:
        code=input('mission:')

        if code == 'DUDE':
            DUDE(input_path,output_path)
            break

        elif code == 'count':
            '计算正负样本数，蛋白数量'
            count_labels(input_path)
            count_proteins(input_path)
            break

        elif code == 'split':
            '分割为训练、验证、测试集'
            split_csv(input_path)
            break

        elif code == 'tanimoto_trndata':
            Tanimoto(input_path, output_path)
            break

        elif code == 'fix':
            input_pdb = ''
            output_pdb = ''
            fix_pdb(input_pdb, output_pdb)
            break

        elif code == 'to_pdb':
            smiles = 'C#CCN(C[C@@H]1[C@@H](OC(=O)NCC2CCCCC2)[C@@H](OC(=O)NCC2CCCCC2)Cn2c(=O)n(CCCCCC)c(=O)n21)S(=O)(=O)c1ccc(C)cc1'
            out = ''
            smile_to_pdb(smiles, out)
            break

        elif code == '5fold':
            fold(input_path,output_path)
            break

        elif code == 'lipinski':
            lipinski(input_path,output_path)
            break

        elif code == 'IQR':
            IQR(input_path)
            break

        elif code == 'filter':
            input_dir = ""
            output_dir = ""
            filter_folder(input_dir, output_dir)
            break

        elif code == 'similar':
            positive_csv =''
            experimental_csv=''
            output_csv=''
            similarity(positive_csv,experimental_csv,output_csv)
            break

        elif code == 'distance':
            positive_csv =''
            experimental_csv=''
            output_csv=''
            distance(positive_csv,experimental_csv,output_csv)
            break

        elif code == 'smile23d':
            smile23d(input_path)
            break

        else:
            continue

