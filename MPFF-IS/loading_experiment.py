import torch
import transformers
import pandas as pd
import time
import dgl
from rdkit import Chem
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dgllife.utils import smiles_to_bigraph,CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import gc

class MolecularDataset(Dataset):
    "Custom PyTorch Dataset for storing compound and protein features"
    def __init__(self, molecule_data):
        self.molecule_data = molecule_data

    def __len__(self):
        return len(self.molecule_data)

    def __getitem__(self, idx):
        return self.molecule_data[idx]

    @staticmethod
    def collate_fn(batch):
        """Batch processing of data provided by DataLoader."""
        graphs = [item['graph'] for item in batch]
        batched_graph = dgl.batch(graphs)  # Merge DGLGraph

        node_feats = torch.cat([item['node_feats'] for item in batch], dim=0)
        edge_feats = torch.cat([item['edge_feats'] for item in batch], dim=0)
        protein_feats = pad_sequence([item['protein_feats'] for item in batch], batch_first=True, padding_value=0)

        adj_matrices = []
        max_rows = max([item['adj_matrix'].shape[0] for item in batch])
        max_cols = max([item['adj_matrix'].shape[1] for item in batch])
        # 将最大行列数存储在 max_size
        max_size = (max_rows, max_cols)
        for item in batch:
            adj_matrix = item['adj_matrix']
            rows, cols = adj_matrix.shape

            # Calculate the rows and columns that need to be filled.
            padding_rows = max_size[0] - rows
            padding_cols = max_size[1] - cols

            padded_matrix = F.pad(adj_matrix, (0, padding_cols, 0, padding_rows), value=0)
            adj_matrices.append(padded_matrix)
        adj_matrix=torch.stack(adj_matrices, dim=0)

        return batched_graph, node_feats, edge_feats, protein_feats,adj_matrix

    @staticmethod
    def get_protein_features(prosequences_list,i,tokenizer,model,device,pro_feats_dic):

        #Processing protein sequences
        protein_sequence = prosequences_list[i]
        protein_tokenizer = tokenizer(protein_sequence, return_tensors="pt").to(device)
        #Extracting protein characteristics
        with torch.no_grad():
            outputs = model(**protein_tokenizer)
            protein_feats = outputs.last_hidden_state.mean(dim=0)
            protein_feats=protein_feats.to(torch.device('cpu'))
            pro_feats_dic[protein_sequence] = protein_feats

        return protein_feats,pro_feats_dic

    @staticmethod
    def loading_data(*file_paths,device):
        smiles_list=[]
        pro_sequences_list=[]

        transformers.logging.set_verbosity_error()
        for file_path in file_paths:
            df = pd.read_csv(file_path,low_memory=False )
            smiles_list.extend(df['SMILES'].tolist())
            pro_sequences_list.extend(df['Protein Sequence'].tolist())

        atom_featurizer = CanonicalAtomFeaturizer()     #Molecular feature extraction function physicalization
        bond_featurizer = CanonicalBondFeaturizer()

        'loading esm2'
        model_path = "esm2_t33_650M_UR50D.pt"
        tokenizer_path = "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = AutoConfig.from_pretrained(tokenizer_path)
        model =AutoModel.from_config(config)
        checkpoint = torch.load(model_path,weights_only=False, map_location=device)
        model_weights = checkpoint['model']
        model.load_state_dict(model_weights, strict=False)
        model = (model.to(device)).eval()

        molecule_data = []  # Store all molecular data
        pro_feats_dic = {}
        protein_feats = None

        print('———————————————— Start loading dataset ————————————————')
        start = time.time()
        for i, smiles in enumerate(smiles_list):
            '''Compound feature extraction'''
            graph = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
            node_feats = graph.ndata['h']
            edge_feats = graph.edata['e']
            mol = Chem.MolFromSmiles(smiles)
            # Get the adjacency matrix
            adj_matrix = Chem.GetAdjacencyMatrix(mol)
            adj_matrix=torch.tensor(adj_matrix)

            '''Protein Feature Extraction'''
            if not pro_feats_dic:
                protein_feats, pro_feats_dic = MolecularDataset.get_protein_features(pro_sequences_list, i, tokenizer, model, device,
                                                                     pro_feats_dic)
            else:
                if pro_sequences_list[i] not in pro_feats_dic:
                    protein_feats,pro_feats_dic=MolecularDataset.get_protein_features(pro_sequences_list,i,tokenizer,model,device,
                                                                      pro_feats_dic)
                else:
                    protein_feats=pro_feats_dic[pro_sequences_list[i]]

            # Store in dictionary
            molecule_info = {
                'graph': graph,
                'node_feats': node_feats,
                'edge_feats': edge_feats,
                'protein_feats': protein_feats,
                'adj_matrix': adj_matrix.float()
            }

            molecule_data.append(molecule_info)

            if i % 1000 == 0:
                print(f'———————————— {i} sets of data loaded, time consuming: {time.time() - start:.2f}s ————————————')

        end = time.time()
        print('——————————Loading data completed, total {} sets of data,time consuming：{:.2f}s——————————'.format(
            len(smiles_list), end - start))
        return MolecularDataset(molecule_data)


