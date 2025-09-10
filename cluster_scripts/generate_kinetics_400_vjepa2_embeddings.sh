#!/bin/bash -l
#SBATCH --job-name=vjepa2_k400_embed_vitl64
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/private/home/francoisporcher/vjepa2/cluster_scripts/logs/%x_%j.out
#SBATCH --error=/private/home/francoisporcher/vjepa2/cluster_scripts/logs/%x_%j.err

set -euo pipefail

# --- Paths & metadata ---
SCRIPT_PATH="/private/home/francoisporcher/vjepa2/scripts/generate_kinetics_400_vjepa2_embeddings.py"
PROJECT_ROOT="/private/home/francoisporcher/vjepa2"
LOG_DIR="/private/home/francoisporcher/vjepa2/cluster_scripts/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "==================== JOB START ===================="
echo "Job Name     : $SLURM_JOB_NAME"
echo "Job ID       : $SLURM_JOB_ID"
echo "User         : $USER"
echo "Cluster      : $SLURM_CLUSTER_NAME"
echo "Node list    : $SLURM_NODELIST"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "QOS          : ${SLURM_JOB_QOS:-normal}"
echo "GPUs         : ${SLURM_GPUS:-1}"
echo "CPUs/Task    : ${SLURM_CPUS_PER_TASK:-12}"
echo "Work dir     : $(pwd)"
echo "Script       : $SCRIPT_PATH"
echo "Start time   : $(date)"
echo "==================================================="

# Threading hints
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- GPU/System info ---
echo ">>> nvidia-smi"
nvidia-smi || true
echo ">>> Environment modules (if any)"
module list 2>/dev/null || true

# --- Conda env ---
# Try the standard hook first
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # Fallback to known locations
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

echo ">>> Activating conda env: vjepa2-312"
conda activate vjepa2-312

echo ">>> Python / CUDA / PyTorch versions"
python -V

# --- Run job ---
echo "================= RUNNING SCRIPT =================="
srun python "$SCRIPT_PATH"
EXIT_CODE=$?
echo "==================================================="

echo "End time     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "=================== JOB COMPLETE =================="
exit $EXIT_CODE
