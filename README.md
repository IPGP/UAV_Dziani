# Installation
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

# Fichier de configuration
le fichier .env doit être créé à la racine du projet et doit contenir les variables d'environnement suivantes :
```
# Identifiant de la feuille Google Sheets contenant les parametres de traitement
GG_SHEET_ID=XXXXXXXXXXXXXXXXXXX
# Chemin vers le dossier de données
ROOT_DATA_PATH=/Volumes/data/images
# Chemin vers le dossier de résultats
OUTPUT_PATH=./
```


# Utilisation
N est le numéro de la ligne dans la feuille Google Sheets contenant les paramètres de traitement pour une expérience donnée.

## Bash
### Calcul du centre de detection
python3 DzianiBullage.py N -c

ex: python3 DzianiBullage.py 4 -c


### Calculs principaux
* Analyse des videos
python3 DzianiBullage.py N -a
ex: python3 DzianiBullage.py 4 -a

* Calcul des interpolations
python3 DzianiBullage.py N -i
ex: python3 DzianiBullage.py 4 -i

* Calcul des interpolations
python3 DzianiBullage.py N -i
ex: python3 DzianiBullage.py 4 -i

* Calcul Largeur Panache
python3 DzianiBullage.py N -l
ex: python3 DzianiBullage.py 4 -l



