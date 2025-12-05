import shutil
import torch
import os
import random
import dgl
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import DataLoader
from mpnn_predictor import MPNNPredictorWithProtein
from loading_experiment import MolecularDataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def append_columns_to_csv(csv_path, list1, list2, col_name1, col_name2):

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    df[col_name1] = pd.Series(list1)
    df[col_name2] = pd.Series(list2)

    df.to_csv(csv_path, index=False)
    print(f"Columns '{col_name1}' and '{col_name2}' have been added to {csv_path}")


def experiment(model,
         dataset,
         batch_size,
         device,
         file,
         seed=None
         ):
    model.eval()
    output_lst=[]
    predicted_lst=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=MolecularDataset.collate_fn)
    counter = 0
    if seed is  None:
        set_seed()
    else:
        set_seed(seed)


    with torch.no_grad():
        for i in dataloader:
            graph, node_feats, edge_feats, protein_feats,adj_matrix= i
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            adj_matrix = adj_matrix.to(device)      # data prepare

            output = torch.sigmoid(model(graph, node_feats, edge_feats, protein_feats,adj_matrix))
            output_lst.extend([round(float(i),4) for i in output.float()])
            predicted_lst.extend([int(i) for i in (output >= 0.5).float()])

            counter += 1
            if counter % 100 == 0:
                print('Calculated {} batches of batch-size data'.format(counter))

    append_columns_to_csv(file, output_lst, predicted_lst, "out", "pre_Label")


def parse_args():
    parser = argparse.ArgumentParser(description="The usage command for MPFF-IS")

    parser.add_argument('-f', type=str, help='CSV file path.', metavar='File path')
    parser.add_argument('-F', type=str, help='CSV files folder path.[Reminder: The -f and -F options cannot be used at the same time.]',
                        metavar='Folder path')
    parser.add_argument('-m', type=str, default='best_model.pth', help='The training model to be used,default=best_model.pth',
                        metavar='Model')
    parser.add_argument('-s', type=int, default=42, help='seed,default=42', metavar='Seed')
    parser.add_argument('-bs', type=int, default=64, help='Batch size,default=64', metavar='Batch size')
    parser.add_argument('-c', type=int, default=0, help='Check data availability,0:check, 1:skip check,default=0', metavar='Check')
    args = parser.parse_args()

    if args.f and args.F:
        print("Error: Cannot use both -f and -F options at the same time!")
        sys.exit(1)

    if args.f:
        file=(args.f,)
    elif args.F:
        file = tuple(f"{os.path.join(args.F, f)}" for f in os.listdir(args.F) if f.endswith('.csv'))

    model=args.m
    bs=args.bs
    seed=args.s
    checkcode=args.c

    return file, model, bs, seed, checkcode


def filter_and_save_csv(file_paths,checkcode, out_dir='Results'):
    base_dir = os.path.dirname(file_paths[0])
    output_dir = os.path.join(base_dir, out_dir)
    os.makedirs(output_dir, exist_ok=True)

    if checkcode == 1:
        for file_path in file_paths:
            out_file = os.path.join(output_dir, os.path.basename(file_path))
            shutil.copy(file_path, out_file)
            print(f"checkcode=1, skipped screening. File copied to {out_file}")
        return output_dir

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    file_count = 1  # New CSV file number

    for file_path in file_paths:
        print(f'Screening {file_count} of {len(file_paths)} files...')
        df = pd.read_csv(file_path, low_memory=False)

        filtered_rows = []

        # tqdm displays the progress of compounds in the current CSV file.
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(file_path), leave=True):
            smiles = row['SMILES']

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                graph = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
                if 'h' not in graph.ndata or 'e' not in graph.edata:
                    continue

                filtered_rows.append(row)
            except Exception:
                continue

        if filtered_rows:
            new_df = pd.DataFrame(filtered_rows)
            out_file_name = os.path.basename(file_path)
            out_file = os.path.join(output_dir, out_file_name)
            new_df.to_csv(out_file, index=False)
            print(f"Screened file saved to {out_file}，total of {len(filtered_rows)} rows")
        else:
            print(f"{file_path} does not have valid data, skipping save.")

        file_count += 1

    return output_dir

