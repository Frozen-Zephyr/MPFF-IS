#!/bin/bash
gmx make_ndx -f md.tpr -o index.ndx <<EOF
q
EOF

# Step 1: PBC展开+中心化
echo -e "1\n0" | gmx trjconv -s md.tpr -f md.xtc -o mdtrj.xtc -pbc mol -center -ur compact -n index.ndx

# Step 2: 对mdtrj进行构象拟合
echo -e "4\n0" | gmx trjconv -s md.tpr -f mdtrj.xtc -o mdfit.xtc -fit rot+trans -n index.ndx

# Step 3: RMSD 计算(对齐骨架计算骨架)
echo -e "4\n4" | gmx rms -s md.tpr -f mdfit.xtc -o rmsd.xvg -tu ns 

# Step 4: RMSF 计算（看蛋白）(residue)
echo -e "1" | gmx rmsf -s md.tpr -f mdfit.xtc -o rmsf.xvg -res

# Step 5: Rg (Gyration radius) 计算（蛋白）
echo 1 | gmx gyrate -s md.tpr -f mdfit.xtc -o gyrate.xvg

# Step 6: SASA 计算
gmx sasa -s md.tpr -f mdfit.xtc -o sasa.xvg -surface 'Protein'


# Step 7: 结果整理
# 将所有结果文件整合到一个文件夹中
mkdir results
mv rmsd.xvg rmsf.xvg gyrate.xvg sasa.xvg results/

echo "计算完成，结果已保存到results文件夹中。"