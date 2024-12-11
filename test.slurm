#!/bin/sh
#SBATCH -J Dziani
#SBATCH --partition=ncpulong,ncpu,ncpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output /gpfs/users/%u/logs/slurm-openmp-%j.out
#SBATCH --mail-user=bonaime@ipgp.fr
#SBATCH --mail-type=END,FAIL

echo $HOME
echo $USER


echo "toto"
