import pandas as pd
import requests
from tqdm import tqdm

csv_file="test.csv"
df=pd.read_csv(csv_file)

#获取蛋白质fasta序列
def get_fasta_sequence(protein_name):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct.title",
                "operator": "contains_phrase",
                "value": protein_name
            }
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry"
    }

    response = requests.post(url, json=query)
    results=response.json().get('result_set',[])
    if results and len(results) > 0:
        pdb_id = results[0].get('identifier', None)
        fasta_url = f"https://files.rcsb.org/download/{pdb_id}.fasta"
        fasta_response = requests.get(fasta_url)
        fasta_sequence = fasta_response.text.split('\n',1)[1].replace('\n','')
        return pdb_id,fasta_sequence
    else:
        print("No results found for protein name:{}".format(protein_name))
        return [],[]

#获取蛋白质结合位点的所有编号
def binding_sites(pdb_id,smiles):
    binding_url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/1"
    response = requests.get(binding_url)
    result=response.json()

    binding_sites=[]

    for entity in result.get("entities",[]):
        for ligand in entity.get("ligands",[]):
            if ligand.get("smiles")==smiles:
                binding_sites.append(ligand.get("residue_number",[]))
    return binding_sites


def to_csv(csv_file,pdb_ids):
    df=pd.read_csv(csv_file)
    df["PDB_id"]=pdb_ids
    df.to_csv(csv_file,index=False)


for index, row in tqdm(df.iterrows(),desc="Processing",unit="row" ):
    binding_seq=[]
    binding_sequence=[] 
    pdb_ids=[]
    fasta_sequence=[]
    protein_name = row["Protein Name"]
    ligand_smiles = row["SMILES"]
    pdb_ids,fasta_sequence=get_fasta_sequence(protein_name)
    if not pdb_ids:
        continue
    if not fasta_sequence:
        continue
    if pdb_ids and fasta_sequence:  
        binding_site = binding_sites(pdb_ids,ligand_smiles)
        if binding_site:
            binding_sequence="".join([fasta_sequence[i-1]for i in binding_site if i<=len(fasta_sequence)])
        else:
            binding_sequence="None"
            binding_seq.append(binding_sequence)
        to_csv(csv_file,pdb_ids)
    else:
        break
