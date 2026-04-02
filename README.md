# UAV Dziani - Gas Bubble Detection Analysis

Analysis tool for detecting and measuring gas bubbles in UAV footage of Dziani lake, Mayotte. This project processes drone videos to identify bubble plumes, calculate their positions, and interpolate temporal data.

## Prerequisites

- Python 3.12 or higher
- FFmpeg (for video processing)

# Installation
## Linux et Mac

### Option 1 : avec uv (recommandé)
```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Créer l'environnement et installer les dépendances
uv sync

# Activer l'environnement
source .venv/bin/activate
```

### Option 2 : avec pip
```bash
# Créer l'environnement virtuel
python3.12 -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Installer les librairies
pip install -r requirements.txt
```

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
#### Installer FFMPEG
Installez ffmpeg sur Windows :
1. Téléchargez la version Windows sur https://ffmpeg.org/download.html
2. Extrayez le contenu de l'archive téléchargée
3. Ajoutez le chemin du dossier "bin" de ffmpeg à la variable d'environnement PATH de Windows

#### Créer et activer l'environnement
```powershell
# Créer l'environnement
conda create -n env python=3.12

# Activer l'environnement
conda activate env

# Installer les librairies
pip install -r requirements.txt
```

#### Utilisation avec Spyder
```powershell
conda activate env
spyder
```

# Configuration

Créez un fichier `.env` à la racine du projet avec les variables suivantes :
```bash
# Identifiant de la feuille Google Sheets contenant les paramètres de traitement
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

# Options de traitement

- `-c` : Calcul du centre de détection (requis en premier)
- `-a` : Analyse des vidéos pour détecter les bulles
- `-i` : Calcul des interpolations temporelles
- `-l` : Calcul de la largeur du panache

# Dépannage

**Problème : FFmpeg non trouvé**
- Linux/Mac : `brew install ffmpeg` ou `apt-get install ffmpeg`
- Windows : Vérifiez que le chemin FFmpeg est dans PATH

**Problème : Erreur d'accès Google Sheets**
- Vérifiez que GG_SHEET_ID est correct dans le fichier .env
- Confirmez les permissions d'accès à la feuille

**Problème : Module non trouvé**
- Vérifiez que l'environnement virtuel est activé
- Réinstallez les dépendances : `pip install -r requirements.txt`

# Structure du projet

```
UAV_Dziani/
├── DzianiBullage.py      # Script principal
├── traitement.py          # Fonctions de traitement vidéo
├── utils.py               # Utilitaires
├── find_center.py         # Détection du centre
├── find_zone.py           # Détection de zones
├── cv2_parametres.py      # Paramètres OpenCV
├── requirements.txt       # Dépendances Python
└── pyproject.toml         # Configuration uv
```
