#!/bin/bash
#SBATCH -J fixed_hom                              # Job name
#SBATCH --account=gts-jueda3                      # charge account
#SBATCH -N1                                       # Number of nodes
#SBATCH --gres=gpu:h100:1                         # GPU type and count
#SBATCH --mem-per-gpu=40GB                        # Memory per GPU
#SBATCH -t0-8:00:00                               # Duration
#SBATCH -qinferno                                 # QOS Name
#SBATCH -oReport-fixed_hom-%j.out                 # Output file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jueda3@gatech.edu
cd "$SLURM_SUBMIT_DIR"

module load pytorch/2.1.0
ROOT_DIR="/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root"
SLURM_DIR="$ROOT_DIR/Slurm_Training"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export APPTAINERENV_PYTHONPATH="$ROOT_DIR"

# Pre-flight check
echo "=== Pre-flight check ==="
echo "ROOT_DIR: $ROOT_DIR"
ls -la "$ROOT_DIR/train_functions.py" 2>&1
ls -la "$ROOT_DIR/architectures/UpdatedNetwork.py" 2>&1
echo "=== Config: FIXED LAMBDA + HOMOGENEOUS (no harmonization) ==="

TRAIN_SCRIPT="$SLURM_DIR/main_train.py"

CMD="apptainer exec --nv \
  --env PYTHONPATH=$ROOT_DIR \
  --bind /storage \
  $PYTORCH_2_1_0_SIF \
  python $TRAIN_SCRIPT \
--epochs 300 \
--batch-size 64 \
--learning-rate 0.001 \
--train-input /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Data/Training/ \
--validation-input /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Data/Validation/pt_files/ \
--model /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Model_Logs/ \
--offsets 8 \
--field-of-view 0.2 \
--arch-type FDTDNet \
--arch-subtype None \
--loss-type residual \
--lambda-data 1.0 \
--lambda-physics 1.0 \
--mse-in-k-space \
--harmonization none"

echo "Running command:"
echo $CMD
srun $CMD
