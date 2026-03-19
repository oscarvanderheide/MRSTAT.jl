#!/bin/bash
# ============================================================================
#  单节点多 GPU: node005 或 node008 (A4000 × 4)
#  用法: sbatch scripts/slurm/run_mpi_single_node.sh
# ============================================================================
#SBATCH --job-name=mrstat-mpi
#SBATCH --output=mrstat_mpi_%j.out
#SBATCH --error=mrstat_mpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A4000:4
#SBATCH -C A4000
#SBATCH --time=01:00:00

module load julia/1.10.3
module load cuda12.3/toolkit/12.3
module load openmpi4/4.1.6-cuda

export JULIA_DEPOT_PATH="/var/scratch/iwang3/julia_depot"
export JULIA_CONDAPKG_ENV="/var/scratch/iwang3/condapkg_env"
export MPLBACKEND="agg"

cd /home/iwang3/mrstat_main

echo "============================================="
echo "  MPI MRSTAT — single node, 4 × A4000"
echo "  Node : $(hostname)"
echo "  Tasks: ${SLURM_NTASKS}"
echo "  Job  : ${SLURM_JOB_ID}"
echo "============================================="

mpirun julia --project=. scripts/mpi_mrstat_recon.jl

echo "Job finished at $(date)"
