#!/bin/bash
#SBATCH -J het_oagh_R2c                            # Job name
#SBATCH --account=gts-jueda3                       # charge account
#SBATCH -N1                                        # Number of nodes
#SBATCH --gres=gpu:h100:1                          # GPU type and count
#SBATCH --mem-per-gpu=40GB                         # Memory per GPU
#SBATCH -t0-8:00:00                                # Duration
#SBATCH -qinferno                                  # QOS Name
#SBATCH -oReport-het_oagh_R2_curriculum-%j.out     # Output file
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
ls -la "$SLURM_DIR/main_train_R2.py" 2>&1
echo "=== Config: HETEROGENEOUS + OAGH + R2 Curriculum ==="

# R2 Option 2: Curriculum learning
# Stage 1 (epochs 0-74):   train SNR=30 (clean)   — learn physics from clean gradients
# Stage 2 (epochs 75-149): train SNR=25           — mild noise introduction
# Stage 3 (epochs 150-224):train SNR=20           — moderate noise
# Stage 4 (epochs 225-299):train SNR=15 (noisy)   — full noise adaptation
#
# IMPORTANT FIXES vs previous run (which crashed on a dead import):
#  1) Dead `from losses.mse_loss import MSELoss` removed — that typo was the
#     sole reason the previous curriculum job crashed at startup (11s walltime).
#  2) Validation loader is now built ONCE at a FIXED SNR (see
#     --curriculum-val-snr below). The previous version rebuilt val loader per
#     stage at the stage's training SNR, so best_loss always saved the
#     stage-1 clean-only checkpoint.
#  3) Checkpoint scheme: weights.pth (best by fixed-SNR val) AND
#     weights_final.pth (last-epoch snapshot). Evaluate both on PACE.
#  4) Per-epoch progress printed for every epoch — easier log inspection.
#
# --curriculum-val-snr 15 → worst-case noise (equals min(schedule)). This is
# the most honest "best" target: pick the checkpoint that generalizes best at
# the hardest SNR in the evaluation grid.

TRAIN_SCRIPT="$SLURM_DIR/main_train_R2.py"

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
--heterogeneous \
--harmonization oagh \
--warmup-epochs 50 \
--curriculum \
--curriculum-schedule 30,25,20,15 \
--curriculum-val-snr 15 \
--r2-tag R2_curriculum_30_25_20_15"

echo "Running command:"
echo $CMD
srun $CMD
