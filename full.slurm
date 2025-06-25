#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # MPI per node
#SBATCH --partition=ncpum
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=%u@ipgp.fr
#SBATCH --mail-type=END,FAIL


cd /gpfs/scratch/bonaime/git/UAV_Dziani
source .venv/bin/activate


# Analyse du film
python3 DzianiBullage.py $1 -a
# Interpolation
python3 DzianiBullage.py $1 -i
# Largeur du panache
python3 DzianiBullage.py $1 -l


