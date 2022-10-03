#!/bin/bash
#
#SBATCH --job-name=text_gen
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=15GB
#SBATCH --output=results_2/%A_%a.out
#SBATCH --error=results_2/%A_%a.err

module purge
module load python/intel/3.8.6


python generate_ai_text_hpc_2.py $SLURM_ARRAY_TASK_ID

