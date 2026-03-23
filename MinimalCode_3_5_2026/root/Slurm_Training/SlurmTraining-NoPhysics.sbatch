#!/bin/bash
#SBATCH -JSlurmTrainingNoPhysics                # Job name
#SBATCH --account=gts-jueda3                    # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --gres=gpu:h100:1                       # GPU type and number of GPUs
#SBATCH --mem-per-gpu=32GB                      # Memory per CPU core, 8 CPUs/GPU
#SBATCH -t0-00:10:00                            # Duration of the job (Ex: t3600=60 min, or use tD-HH:MM:SS)
#SBATCH -qinferno                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=hnieves6@gatech.edu         # E-mail address for notifications
cd "$SLURM_SUBMIT_DIR"                          # Change to working directory

module load pytorch/2.1.0                       # Load module dependencies
#export PYTHONPATH=/storage/home/hcoda1/9/hnieves6/.local/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=""

# Define arguments
FILE_ARG="--file myfile.txt"
EPOCHS_ARG="--epochs 4"
BATCH_ARG="--batch-size 64"
LR_ARG="--learning-rate 0.001"
ROOT_ARG='--root /storage/coda1/p-jueda3/0/hnieves6/MRE_Inversion/'
TRAIN_INPUT_ARG='--train-input /storage/coda1/p-jueda3/0/hnieves6/MRE_Inversion/Training/'
VAL_INPUT_ARG='--validation-input /storage/coda1/p-jueda3/0/hnieves6/MRE_Inversion/Validation/'
MODEL_ARG='--model /storage/coda1/p-jueda3/0/hnieves6/MRE_Inversion/Model_Logs/'
OFFSETS_ARG="--offsets 8"
FOV_ARG="--field-of-view 0.2"
ARCH_TYPE="--arch-type FDTDNet"
ARCH_SUBTYPE_ARG="--arch-subtype None"
LOSS_TYPE="--loss-type mse"     # Choose 'mse' or 'hom'
# DENSITY_ARG="--density 1000.0"
# RESIDUAL_ARG="--residual-type normalized"
# K_FILTER_ARG="--k-filter 1000.0"
# EPSILON_ARG="--epsilon 1e-10"
# VERBOSE_ARG="--verbose False"
# LAMBDA_HOM_ARG="--lambda-hom 1.0"
LAMBDA_DATA_ARG="--lambda-data 1.0"


# Run script with arguments
CMD="apptainer exec --nv $PYTORCH_2_1_0_SIF python main_train.py \
  $FILE_ARG \
  $EPOCHS_ARG \
  $BATCH_ARG \
  $LR_ARG \
  $ROOT_ARG \
  $TRAIN_INPUT_ARG \
  $VAL_INPUT_ARG \
  $MODEL_ARG \
  $OFFSETS_ARG \
  $FOV_ARG \
  $ARCH_TYPE \
  $ARCH_SUBTYPE_ARG \
  $LOSS_TYPE \
  $LAMBDA_DATA_ARG"

#   $RESIDUAL_ARG \
#   $DENSITY_ARG \
#   $K_FILTER_ARG \
#   $EPSILON_ARG \
#   $VERBOSE_ARG \
#   $LAMBDA_HOM_ARG"

# Run the command
echo "Running command:"
echo $CMD
srun $CMD