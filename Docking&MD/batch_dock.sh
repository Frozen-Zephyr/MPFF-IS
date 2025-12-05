#!/bin/bash
#
#   ./run_docking.sh <working_directory> <receptor_file> <reference_ligand_file> [output_filename]
#
# eg:
#   ./run_docking.sh /home/yzhu/docking krasg12d.pdb ref_lig.pdb results.txt
#


if [[ $# -lt 3 ]]; then
    echo " $0 <working_directory> <receptor_file> <reference_ligand_file> [output_filename]"
    exit 1
fi

workdir=$1
receptor=$2
ref_lig=$3
output_txt=${4:-docking_results.txt}   

cd "$workdir" || { echo "❌ can't find $workdir"; exit 1; }


> "$output_txt"


for lig in *.pdb; do
    if [[ "$lig" == "$receptor" || "$lig" == "$ref_lig" ]]; then
        continue
    fi

    echo " $lig ..."

    result=$(rundock -p "$receptor" -l "$lig" -F REF -R "$ref_lig" 2>&1)

    energy=$(echo "$result" | grep "DOCKING>" | tail -n 1 | awk '{print $3}')

    echo "$lig : $energy"
    echo "$lig : $energy" >> "$output_txt"
done

echo "save to $output_txt"
