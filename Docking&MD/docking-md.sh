#!/bin/bash
conda activate amber
# usage: ./prep_ligand.sh input.pdbqt output_prefix charge
# eg: ./prep_ligand.sh lig.pdbqt chembl520 -4

if [ $# -lt 3 ]; then
  echo "usage: $0 input.pdbqt output_prefix charge"
  echo "eg: $0 lig.pdbqt gtp -4"
  exit 1
fi

input_pdbqt=$1
prefix=$2
manual_charge=$3

echo ">>> Step 1: PDBQT to PDB"
obabel "$input_pdbqt" -O ${prefix}.pdb

echo ">>> Step 3: reduce"
reduce -Trim ${prefix}_noCONECT.pdb > ${prefix}_noH.pdb
reduce -BUILD ${prefix}_noH.pdb > ${prefix}_allH.pdb

echo ">>> Step 4: pdb4amber"
pdb4amber -i ${prefix}_allH.pdb -o ${prefix}_clean.pdb --reduce

echo ">>> Step 5: Regenerate mol2 with manually entered charges"
antechamber -i ${prefix}_clean.pdb -fi pdb -o ${prefix}.mol2 -fo mol2 -c bcc -s 2 -nc $manual_charge

echo ">>> done!"
echo "    ${prefix}_clean.pdb (clean pdb)"
echo "    ${prefix}.mol2      "

