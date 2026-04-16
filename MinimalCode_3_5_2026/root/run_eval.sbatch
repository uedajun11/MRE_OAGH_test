#!/bin/bash
#SBATCH --job-name=mre_eval
#SBATCH --account=gts-jueda3
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# ------------------------------------------------
# DO NOT load anaconda3
# DO NOT use conda activate/deactivate
# ------------------------------------------------

PYTHON=/storage/project/r-jueda3-0/jueda3/conda_envs/mre_eval/bin/python

echo "Python executable:"
$PYTHON --version

echo "Torch check:"
$PYTHON - << 'EOF'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

cd /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root
$PYTHON evaluate_Ueda.py