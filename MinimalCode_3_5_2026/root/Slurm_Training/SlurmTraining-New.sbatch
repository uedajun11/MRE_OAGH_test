#!/bin/bash
#SBATCH -J SlurmTraining                         # Job name
#SBATCH --account=gts-jueda3                    # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --gres=gpu:h100:1                       # GPU type and number of GPUs
#SBATCH --mem-per-gpu=40GB                      # Memory per CPU core, 8 CPUs/GPU
#SBATCH -t0-8:00:00                            # Duration of the job (Ex: t3600=60 min, or use tD-HH:MM:SS)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=jueda3@gatech.edu         # E-mail address for notifications
cd “$SLURM_SUBMIT_DIR”                          # Change to working directory

module load pytorch/2.1.0                       # Load module dependencies
ROOT_DIR=“/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root”
export PYTHONPATH=“$ROOT_DIR”
export APPTAINERENV_PYTHONPATH=“$ROOT_DIR”  # Pass PYTHONPATH into the apptainer container

# ==================== TRAINING CONFIGURATION ====================

# Basic training parameters
EPOCHS_ARG=“--epochs 300”
BATCH_ARG=“--batch-size 64"
LR_ARG=“--learning-rate 0.001”

# Data paths



TRAIN_INPUT_ARG=“--train-input /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Data/Training/”
VAL_INPUT_ARG=“--validation-input /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Data/Validation/”
MODEL_ARG=“--model /storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Model_Logs/”

# Data parameters
OFFSETS_ARG=“--offsets 8”
FOV_ARG=“--field-of-view 0.2"

# Architecture
ARCH_TYPE=“--arch-type FDTDNet”
ARCH_SUBTYPE_ARG=“--arch-subtype None”

# Loss configuration
LOSS_TYPE=“--loss-type residual”      # Options: mse, residual, ratio

# Loss weights
LAMBDA_DATA_ARG=“--lambda-data 1.0"   # Weight for MSE loss
LAMBDA_PHYSICS_ARG=“--lambda-physics 1.0”  # Weight for physics loss

# Loss-specific flags (uncomment as needed)
HETEROGENEOUS_FLAG=“--heterogeneous”        # For residual: use heterogeneous term
ORTHOGONALIZE_FLAG=“--orthogonalize-het”    # For ratio: orthogonalize T_het
MSE_K_SPACE_FLAG=“--mse-in-k-space”         # Compute MSE in k-space

# Gradient harmonization: none, oagh, or oagh_c
HARMONIZATION_ARG=“--harmonization oagh”  # Options: none, oagh, oagh_c

# Run script with arguments
CMD=“apptainer exec --nv $PYTORCH_2_1_0_SIF python main_train.py \
$EPOCHS_ARG \
$BATCH_ARG \
$LR_ARG \
$TRAIN_INPUT_ARG \
$VAL_INPUT_ARG \
$MODEL_ARG \
$OFFSETS_ARG \
$FOV_ARG \
$ARCH_TYPE \
$ARCH_SUBTYPE_ARG \
$LOSS_TYPE \
$LAMBDA_DATA_ARG \
$LAMBDA_PHYSICS_ARG \
$MSE_K_SPACE_FLAG \
$HARMONIZATION_ARG \
$HETEROGENEOUS_FLAG ”



# Run the command
echo “Running command:”
echo $CMD
srun $CMD