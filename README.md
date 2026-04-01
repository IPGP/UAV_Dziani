# Installation
## Linux et Mac

### Environnement virtuel
python3.12 -m venv .venv


### activation
source .venv/bin/activate

### Installer les librairies dans l’environnement :
pip install -r requirements.txt


## Windows
### Option 1 : avec uv (recommandé)
```powershell
# Installer uv si nécessaire
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Créer l'environnement et installer les dépendances
uv sync

# Activer l'environnement
.venv\Scripts\activate
```

### Option 2 : avec conda
### installer FFMPEG
Installez ffmpeg sur Windows :
Téléchargez la version Windows sur https://ffmpeg.org/download.html
Extrayez le contenu de l'archive téléchargée et ajoutez le chemin du dossier "bin" de ffmpeg à la variable d'environnement PATH de Windows.

### Créer environnement sous anaconda :
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

## Avec uv (recommandé)
### Calcul du centre de detection
```bash
uv run DzianiBullage.py N -c
# Exemple:
uv run DzianiBullage.py 4 -c
```

### Calculs principaux
* Analyse des vidéos
```bash
uv run DzianiBullage.py N -a
# Exemple:
uv run DzianiBullage.py 10 -a
```

* Calcul des interpolations
```bash
uv run DzianiBullage.py N -i
# Exemple:
uv run DzianiBullage.py 4 -i
```

* Calcul Largeur Panache
```bash
uv run DzianiBullage.py N -l
# Exemple:
uv run DzianiBullage.py 4 -l
```

## Avec Python (alternative)
Si vous utilisez l'environnement virtuel traditionnel :
```bash
python3 DzianiBullage.py N -c
python3 DzianiBullage.py N -a
python3 DzianiBullage.py N -i
python3 DzianiBullage.py N -l
```



