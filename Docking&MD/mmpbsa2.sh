#!/bin/bash


LIG1_MOL2="$1"
LIG2_MOL2="$2"

if [ -z "$LIG1_MOL2" ]; then
    echo "❌ "
    exit 1
fi

LIG1_BASE="${LIG1_MOL2%.*}"

HAS_LIG2=false
if [ -n "$LIG2_MOL2" ]; then
    HAS_LIG2=true
    LIG2_BASE="${LIG2_MOL2%.*}"
fi

echo "🔍 lig1：$LIG1_MOL2"
if $HAS_LIG2; then
    echo "🔍 lig2：$LIG2_MOL2"
else
    echo "🔍 lig-one"
fi



echo "Step 0: parmchk2 "

parmchk2 -i "$LIG1_MOL2" -f mol2 -o "${LIG1_BASE,,}.frcmod"

if $HAS_LIG2; then
    parmchk2 -i "$LIG2_MOL2" -f mol2 -o "${LIG2_BASE,,}.frcmod"
fi



echo "Step 1: creat index.ndx..."

gmx make_ndx -f md.tpr -o index.ndx <<EOF
1 | 14
name 23 rec
15 | 16
name 24 lig
1|14|15|16
name 25 complex
q
EOF


echo " tleap.in..."

cat > tleap.in <<EOF
source leaprc.gaff

# Load ligand1
loadAmberParams ${LIG1_BASE,,}.frcmod
lig1 = loadMol2 $LIG1_MOL2
EOF

if $HAS_LIG2; then
cat >> tleap.in <<EOF

# Load ligand2
loadAmberParams ${LIG2_BASE,,}.frcmod
lig2 = loadMol2 $LIG2_MOL2

# Combine
lig = combine { lig1 lig2 }
EOF
else
cat >> tleap.in <<EOF

# Single ligand
lig = lig1
EOF
fi

cat >> tleap.in <<EOF

saveMol2 lig lig_combined.mol2 1
savePDB lig lig_combined.pdb

check lig
desc lig
quit
EOF

echo "tleap..."
tleap -f tleap.in

echo "Step 3: Trajectory processing: remove water/ions, unwrap PBC, and align..."

gmx trjconv -s md.tpr -f md.xtc -o no_solv.xtc -pbc mol -center -n index.ndx <<EOF
1
25
EOF

echo "提取初始结构..."
gmx trjconv -s md.tpr -f md.xtc -n index.ndx -o start.gro -dump 0 <<EOF
25
EOF


echo "creat mmpbsa.in..."

cat > mmpbsa.in <<EOF
&general
  startframe=1
  endframe=30000
  interval=200
  verbose=1
  keep_files=0
/

&gb
  igb=5
  saltcon=0.150
/

&pb
  istrng=0.150
/
EOF



echo "🚀 Step 5: gmx_MMPBSA "

if $HAS_LIG2; then
    gmx_MMPBSA -O \
      -i mmpbsa.in \
      -cs md.tpr \
      -ct no_solv.xtc \
      -ci index.ndx \
      -cg 23 24 \
      -lm lig_combined.mol2

else
    gmx_MMPBSA -O \
      -i mmpbsa.in \
      -cs md.tpr \
      -ct no_solv.xtc \
      -ci index.ndx \
      -cg 23 16 \
      -lm "$LIG1_MOL2"

fi

echo "🎉 MMPBSA done！"
