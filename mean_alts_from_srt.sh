#!/bin/bash

dir_name=$1

mean_file(){
filename="$1"
file_n=`basename $filename`
#echo $filename
# Extraire l'extension du fichier
extension="${filename##*.}"
#echo $extension

# Vérifier si l'extension est .srt ou .SRT et effectuer des actions différentes
if [[ "$extension" == "srt" ]]; then
    echo "Action pour un fichier .srt"

    cat $filename | grep HOME | cut -d' ' -f9- | cut -d'S' -f1,2
    # Insère ici l'action que tu veux pour un fichier en .srt
elif [[ "$extension" == "SRT" ]]; then
    #echo "Action pour un fichier .SRT"
    # Insère ici l'action que tu veux pour un fichier en .SRT
    # Rel alti abs_alt
    #cat $fichier | grep rel_alt | cut -d':' -f 12,13 | cut -d']' -f1 | cut -d' ' -f2,4 | awk '{ total_rel += $1;total_abs+=$2} END { print "moyenne rel : "total_rel/NR "\tmoyenne abs : "total_abs/NR}'
    moy=`cat $filename | grep rel_alt | cut -d':' -f 12,13 | cut -d']' -f1 | cut -d' ' -f2,4 | awk '{ total_abs+=$2} END { print total_abs/NR}'`
    echo $file_n $moy

    #rel_alt | cut -d':' -f 12,13 | cut -d']' -f1 | cut -d' ' -f2,4 | awk '{ total_abs+=$2} END { print total_abs/NR}'`

fi

}

export -f mean_file
find $dir_name -iname "*.srt"  -exec bash -c 'mean_file "$0"' {} \;
