#!/bin/bash
#SBATCH -J di_sweep                                # Job name
#SBATCH --account=gts-jueda3                       # charge account
#SBATCH -N1                                        # Number of nodes
#SBATCH --gres=gpu:h100:1                          # GPU type and count
#SBATCH --mem-per-gpu=40GB                         # Memory per GPU
#SBATCH -t0-4:00:00                                # Duration (longer for sweep)
#SBATCH -qinferno                                  # QOS Name
#SBATCH -oReport-direct_inv_sweep-%j.out           # Output file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jueda3@gatech.edu
cd "$SLURM_SUBMIT_DIR"

module load pytorch/2.1.0
ROOT_DIR="/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/root"
SLURM_DIR="$ROOT_DIR/Slurm_Training"
DATA_DIR="/storage/home/hcoda1/2/jueda3/r-jueda3-0/MRE_OAGH_test/MinimalCode_3_5_2026/Data"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export APPTAINERENV_PYTHONPATH="$ROOT_DIR"

# Pre-flight check
echo "=== Pre-flight check ==="
echo "ROOT_DIR: $ROOT_DIR"
echo "DATA_DIR: $DATA_DIR"
ls -la "$ROOT_DIR/losses/homogeneous.py" 2>&1
echo "=== Config: DIRECT INVERSION PARAMETER SWEEP ==="

# Output directory for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/storage/project/r-jueda3-0/jueda3/MRE_OAGH_test/MinimalCode_3_5_2026/Model_Logs/DirectInversion_sweep_${TIMESTAMP}"

CMD="apptainer exec --nv \
  --env PYTHONPATH=$ROOT_DIR \
  --bind /storage \
  $PYTORCH_2_1_0_SIF \
  python $SLURM_DIR/run_direct_inversion.py \
--test-dirs \
  $DATA_DIR/Test/pt_files/ \
  $DATA_DIR/Inclusion/R0_3125cm/pt_files/ \
  $DATA_DIR/Inclusion/R0_625cm/pt_files/ \
  $DATA_DIR/Inclusion/R1_25cm/pt_files/ \
  $DATA_DIR/Inclusion/R2_5cm/pt_files/ \
--output-dir $OUTPUT_DIR \
--fov 0.2 \
--offsets 8 \
--snr-levels 30 25 20 15 \
--k-filter-sweep 200 300 500 700 1000 \
--medfilt-sweep 3 5 7"

echo "Running command:"
echo $CMD
echo ""
echo "Sweep: 5 k_filter values x 3 medfilt kernels = 15 configurations"
echo "Each config evaluates 5 datasets x 4 SNR levels = 20 conditions"
echo "Total: 300 evaluation runs"
echo ""
srun $CMD
