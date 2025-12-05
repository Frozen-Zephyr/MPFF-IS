import torch
import transformers
import pandas as pd
import dgl
from encoders import *
from torch.utils.data import Sampler
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dgllife.utils import smiles_to_bigraph,CanonicalAtomFeaturizer, CanonicalBondFeaturizer
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
        ligand_feats = pad_sequence(ligand_feats_list, batch_first=True)  # [batch_size, 节点特征维度]
        protein_feats = pad_sequence([item['protein_feats'] for item in batch], batch_first=True, padding_value=0)
        labels = torch.stack([torch.tensor([item['label']], dtype=torch.float32) for item in batch],
                             dim=0)  # [batch_size]

        return ligand_feats, protein_feats, labels

    def get_protein_features(prosequences_list,i,tokenizer,model,device,pro_feats_dic):

        #处理蛋白序列
        protein_sequence = prosequences_list[i]
        protein_tokenizer = tokenizer(protein_sequence, return_tensors="pt").to(device)
        #提取蛋白特征
        with torch.no_grad():
            outputs = model(**protein_tokenizer)
            protein_feats = outputs.last_hidden_state.mean(dim=0)  # 取平均作为蛋白特征
            protein_feats=protein_feats.to(torch.device('cpu'))
            pro_feats_dic[protein_sequence] = protein_feats

        return protein_feats,pro_feats_dic       #（L，C）序列个数，隐藏层维度

    @staticmethod
    def loading_data(file_path,device):
        smiles_list=[]
        pro_sequences_list=[]
        label_list=[]

        transformers.logging.set_verbosity_error()
        """从csv加载数据"""

        df = pd.read_csv(file_path,low_memory=False )
        smiles_list.extend(df['SMILES'].tolist())
        pro_sequences_list.extend(df['Protein Sequence'].tolist())
        label_list.extend(df['Label'].tolist())

        model_path = "esm2_t33_650M_UR50D.pt"
        tokenizer_path = "tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 加载模型配置文件
        config = AutoConfig.from_pretrained(tokenizer_path)
        # 初始化模型架构
        model =AutoModel.from_config(config)
        checkpoint = torch.load(model_path,weights_only=False, map_location=device)  # 读取 .pt 权重文件
        model_weights = checkpoint['model']  # 获取模型的权重
        # 将权重加载到模型
        model.load_state_dict(model_weights, strict=False)
        model = (model.to(device)).eval()



        molecule_data = []  # 存储所有分子数据
        pro_feats_dic = {}
        protein_feats = None

        print('———————————————— Start loading dataset ————————————————')
        start = time.time()
        for i, smiles in enumerate(smiles_list):
            '''化合物特征提取'''
            encoder = encoders()
            lig_feats = encoder.get_lig_fingerprint_features(smiles=smiles)

            '''蛋白质特征提取'''
            if not pro_feats_dic:
                protein_feats, pro_feats_dic = MolecularDataset.get_protein_features(pro_sequences_list, i, tokenizer, model, device,
                                                                     pro_feats_dic)
            else:
                if pro_sequences_list[i] not in pro_feats_dic:
                    protein_feats,pro_feats_dic=MolecularDataset.get_protein_features(pro_sequences_list,i,tokenizer,model,device,
                                                                      pro_feats_dic)
                else:
                    protein_feats=pro_feats_dic[pro_sequences_list[i]]

            '''label提取'''
            label = label_list[i]

            # 存储到字典
            molecule_info = {
                'lig_feats': lig_feats,  # shape:[L,128]
                'protein_feats': protein_feats,  # 形状 [L, 1280]，之后 `collate_fn` 处理
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

class ProportionalBatchSampler(Sampler):
    def __init__(self, generic_len, kras_len, batch_size, kras_ratio=0.2):
        self.generic_len = generic_len
        self.kras_len = kras_len
        self.batch_size = batch_size
        self.kras_ratio = kras_ratio

        self.num_kras = int(batch_size * kras_ratio)
        self.num_generic = batch_size - self.num_kras

        self.generic_indices = list(range(generic_len))
        self.kras_indices = list(range(kras_len))

    def __iter__(self):
        # 每次打乱
        random.shuffle(self.generic_indices)
        random.shuffle(self.kras_indices)

        g_ptr = 0
        k_ptr = 0

        while g_ptr + self.num_generic <= self.generic_len:
            if k_ptr + self.num_kras > self.kras_len:
                # 重复使用 KRAS 索引
                random.shuffle(self.kras_indices)
                k_ptr = 0

            generic_batch = self.generic_indices[g_ptr: g_ptr + self.num_generic]
            kras_batch = self.kras_indices[k_ptr: k_ptr + self.num_kras]

            g_ptr += self.num_generic
            k_ptr += self.num_kras

            # 返回组合 batch 索引，前面是 generic，后面是 KRAS（或混合）
            yield [('generic', i) for i in generic_batch] + [('kras', i) for i in kras_batch]

    def __len__(self):
        return self.generic_len // self.num_generic

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, generic_dataset, kras_dataset):
        self.generic_dataset = generic_dataset
        self.kras_dataset = kras_dataset

    def __getitem__(self, index_tuple):
        kind, idx = index_tuple
        if kind == 'generic':
            return self.generic_dataset[idx]
        else:
            return self.kras_dataset[idx]

    def __len__(self):
        return len(self.generic_dataset) + len(self.kras_dataset)  # 用不到

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('using device:{}'.format(device))
# file_path='testset.csv'
# loading_data=MolecularDataset.loading_data(file_path,device=device)
