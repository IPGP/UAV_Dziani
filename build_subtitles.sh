#!/bin/bash

# Vérifier si le répertoire est passé en argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Répertoire où se trouvent les fichiers vidéo
DIR="$1"

# Types de fichiers vidéo que nous voulons vérifier
VIDEO_EXTENSIONS=("mp4" "mkv" "avi" "mpeg" "mov")



# Fonction pour vérifier l'existence des sous-titres et les extraire si nécessaire
extract_subtitles() {
  local file="$1"
  local filename=$(basename "$file")
  local base="${file%.*}"  # Retirer l'extension du fichier vidéo
  local srt_file="$base.srt"

  # Vérifier si le fichier .srt existe déjà
  if [ ! -f "$srt_file" ]; then
    echo "Vérification des sous-titres pour: $filename"

    # Trouver le flux de sous-titres dans le fichier vidéo
    subtitle_stream=$(ffmpeg -hide_banner -loglevel error -i "$file" 2>&1 | grep 'Stream.*Subtitle' | grep -oP '#\K[0-9]+:[0-9]+')

    if [ -z "$subtitle_stream" ]; then
      echo "Pas de sous-titres trouvés dans $filename"
    else
      echo "Flux de sous-titres trouvé: $subtitle_stream"
      echo "Extraction des sous-titres pour: $filename"
    subtitle_stream="0:3"

      # Extraire les sous-titres avec ffmpeg
      ffmpeg -hide_banner -loglevel error -i "$file" -map 0:"$subtitle_stream" "$srt_file"

      if [ $? -eq 0 ]; then
        echo "Sous-titres extraits avec succès dans $srt_file"
      else
        echo "Échec de l'extraction des sous-titres pour $filename"
      fi
    fi
  else
    echo "Sous-titres déjà présents pour: $filename"
  fi
}

# Parcourir tous les fichiers dans le répertoire
find "$DIR" -type f | while read -r file; do
  # Obtenir l'extension du fichier et la convertir en minuscules
  extension="${file##*.}"
  extension=$(echo "$extension" | tr '[:upper:]' '[:lower:]')

  # Vérifier si le fichier est une vidéo (selon les extensions)
  for ext in "${VIDEO_EXTENSIONS[@]}"; do
    if [[ "$extension" == "$ext" ]]; then
      extract_subtitles "$file"
      break
    fi
  done
done
