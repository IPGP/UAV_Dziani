# installation
## Linux et Mac

### Environnement virtuel
python3 -m venv .venv #
source .venv/bin/activate

### activation
pip install -r requirements.txt


## Windows
#### Créer environnement sous anaconda :
conda create -n env python=3.12
conda activate env

### Installer les librairies dans l’environnement :
pip install -r requirements.txt

### Utilisation dans Powershell prompt anaconda
conda activate env
spyder

# Utilisation
## Calcul du centre de detection
sbatch find_center.slurm 4

## Calculs principaux
BB
sbatch dziani_interpolation_BB.slurm 4 -a # Analyse la ligne 4 et traite les données
sbatch dziani_interpolation_BB.slurm 4 # Traite les données seulement

SB
sbatch dziani_interpolation_SB.slurm 4 -a # Analyse la ligne 4 et traite les données
sbatch dziani_interpolation_SB.slurm 4 # Traite les données seulement
