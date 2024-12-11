#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # MPI per node
#SBATCH --partition=ncpulong,ncpu,ncpushort
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=benard@ipgp.fr
#SBATCH --mail-type=END,FAIL


cd /gpfs/users/benard/UAV_Dziani
source .venv/bin/activate
python3 -V

#On lance avec l'option $1 qui est le numéro de ligne à traiter
python3 DzianiBullage.py $1 $2
