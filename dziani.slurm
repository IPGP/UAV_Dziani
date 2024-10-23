#!/bin/sh
#SBATCH -J Dziani_Python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128  # MPI per node
#SBATCH --partition=ncpulong,ncpu,ncpushort
#SATCH --mem=120G
#SBATCH --output /gpfs/users/bonaime/logs/slurm-openmp-%j.out
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


cd /gpfs/scratch/bonaime/git/UAV_Dziani

source .venv/bin/activate


#On lance avec l'option $1 qui est le numéro de ligne à traiter
python3 DzianiBullage.py $1
