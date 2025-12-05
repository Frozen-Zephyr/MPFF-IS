import torch
import transformers
import pandas as pd
import time
import dgl
from rdkit import Chem
from torch.utils.data import Sampler
from encoders import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class MolecularDataset(Dataset):
    """自定义 PyTorch Dataset，用于存储化合物和蛋白质特征"""
    def __init__(self, molecule_data):
        self.molecule_data = molecule_data

    def __len__(self):
        return len(self.molecule_data)

    def __getitem__(self, idx):
        return self.molecule_data[idx]

    @staticmethod
    def collate_fn(batch):
        """批量处理 DataLoader 提供的数据"""
        ligand_feats_list = [item['lig_feats'] for item in batch]
        ligand_feats = pad_sequence(ligand_feats_list, batch_first=True)   # [batch_size, 节点特征维度]
        protein_feats = pad_sequence([item['protein_feats'] for item in batch], batch_first=True, padding_value=0)
        labels = torch.stack([torch.tensor([item['label']], dtype=torch.float32) for item in batch], dim=0) # [batch_size]


        return ligand_feats,protein_feats, labels

    @staticmethod
    def loading_data(file_paths):
        smiles_list=[]
        pro_sequences_list=[]
        label_list=[]

        transformers.logging.set_verbosity_error()
        """从csv加载数据"""

        df = pd.read_csv(file_paths,low_memory=False )
        smiles_list.extend(df['SMILES'].tolist())
        pro_sequences_list.extend(df['Protein Sequence'].tolist())
        label_list.extend(df['Label'].tolist())



        molecule_data = []  # 存储所有分子数据
        pro_feats_dic = {}
        protein_feats = None

        print('———————————————— Start loading dataset ————————————————')
        start = time.time()
        lig_feats_dic={}
        for i, smiles in enumerate(smiles_list):
            '''化合物特征提取'''
            encoder=encoders()
            lig_feats=encoder.get_lig_fingerprint_features(smiles=smiles)


            '''蛋白质特征提取'''
            if not pro_feats_dic:
                protein_feats, pro_feats_dic = encoder.get_protein_onehot_features(pro_sequences_list, i,
                                                                                            pro_feats_dic)
            else:
                if pro_sequences_list[i] not in pro_feats_dic:
                    protein_feats, pro_feats_dic = encoder.get_protein_onehot_features(pro_sequences_list, i,
                                                                                                pro_feats_dic)
                else:
                    protein_feats = pro_feats_dic[pro_sequences_list[i]]
            # protein_feats=''


            '''label提取'''
            label = label_list[i]

            # 存储到字典
            molecule_info = {
                'lig_feats': lig_feats,  # 直接存 DGLGraph
                'protein_feats': protein_feats,  # 形状 [1, 640]，之后 `collate_fn` 处理
                'label': label,
            }
            #添加到列表
            molecule_data.append(molecule_info)

            if i % 10000 == 0:
                print(f'———————————— {i} sets of data loaded, time consuming: {time.time() - start:.2f}s ————————————')

        end = time.time()
        print('——————————Loading data completed, total {} sets of data,time consuming：{:.2f}s——————————'.format(
            len(smiles_list), end - start))
        return MolecularDataset(molecule_data)




# device = torch.device("cpu")
# print('using device:{}'.format(device))
# file_path='/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/训练conbine/trainset.csv'
# loading_data=MolecularDataset.loading_data(file_path)

