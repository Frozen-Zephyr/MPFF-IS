from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import torch
from rdkit import RDLogger
import gc, torch, time, numpy as np
import torch, gc, time
from collections import OrderedDict


class encoders(nn.Module):
    def __init__(self, max_cache=20000):
        super().__init__()
        self._lig_cache = OrderedDict()
        self._call_count = 0
        self._max_cache = max_cache


    def get_lig_fingerprint_features(self,smiles, radius=2, n_bits=64):
        RDLogger.DisableLog('rdApp.*')

        # 缓存命中
        if smiles in self._lig_cache:
            return self._lig_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        bitInfo = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, bitInfo=bitInfo
        )

        lig_feats = np.zeros((num_atoms, n_bits), dtype=np.float32)
        for bit, atoms in bitInfo.items():
            for atom_idx, _ in atoms:
                lig_feats[atom_idx, bit] = 1.0

        lig_feats = torch.from_numpy(lig_feats).contiguous()

        # 缓存结果
        if len(self._lig_cache) < self._max_cache:
            self._lig_cache[smiles] = lig_feats

        # 定期清理内存
        self._call_count += 1
        if self._call_count % 5000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.strftime("%H:%M:%S")
            print(f"[{t0}] Cache cleaned, size = {len(self._lig_cache)}")

        del mol, bitInfo
        return lig_feats

# e=encoders()
# l=e.get_lig_fingerprint_features('CC(Oc1ccc(cc1C)C(=O)C1=C(O)C(=O)N(C1c1ccc(cc1)O)CCCN1CCOCC1)C')
