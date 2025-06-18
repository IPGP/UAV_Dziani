#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # MPI per node
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL


cd /gpfs/scratch/bonaime/git/UAV_Dziani
source .venv/bin/activate
python3 -V

# Trouve le centre des calculs pour l'option $1 qui est le numéro de ligne à traiter

python3 find_center.py -n $1
