#!/bin/bash

liste=("0" "1" "2" "3" "4" "5" "5" "7" "10" "11" "12" "13" "14" "15" "16" "17" "18")

for x in "${liste[@]}"; do
  sbatch full.slurm "$x"
done
