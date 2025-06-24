#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # MPI per node
#SBATCH --partition=ncpum
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL


cd /gpfs/scratch/bonaime/git/UAV_Dziani
source .venv/bin/activate
python3 -V

#On lance avec l'option $1 qui est le numéro de ligne à traiter et option potentiel -a pour
# le 2eme parametre pour analyser le fichier

python3 DzianiBullage.py $1 $2
date
