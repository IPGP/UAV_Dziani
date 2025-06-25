#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64  # MPI per node
#SBATCH --partition=ncpum,ncpushort,ncpu
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL


scontrol update JobId=$SLURM_JOB_ID jobname="dz_$1"

cd /gpfs/scratch/bonaime/git/UAV_Dziani
source .venv/bin/activate

#On lance avec l'option $1 qui est le numéro de ligne à traiter et option potentiel -a pour
# le 2eme parametre pour analyser le fichier
date
python3 DzianiBullage.py $1 $2
date
