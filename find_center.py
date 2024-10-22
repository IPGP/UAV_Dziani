#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import math
from multiprocessing import Pool, freeze_support, RLock
import os
import random
import argparse
import socket
import sys
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import cv2
from matplotlib import colors
import psutil
import requests
from scipy.interpolate import griddata
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter
from tslearn.datasets import CachedDatasets
from utils import get_data_from_google_sheet

#from skimage import color, data, filters, graph, measure, morphology


NB_POINTS: int = 5000

SHOW_IMAGES = False
SAVE_PLOTS = True

rayon_cercle_largeur_ligne = 1



# Paramètres pour la détection de coins Shi-Tomasi et le suivi optique Lucas-Kanade
DETECTION_PARAMETERS = dict(maxCorners=NB_POINTS, qualityLevel=0.1, minDistance=0.5, blockSize=10)
TRACKING_PARAMETERS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def frame_to_BGR2GRAY(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def findIntersection(L1,L2):
    x=(L1[1]-L2[1])/(L2[0]-L1[0])
    y=L1[0]*x+L1[1]
    return(x,y)

def save_trajet(img,name):

    #cv2.imshow('frame', img)

    if SAVE_PLOTS:
        filename = name
        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, img)







def calcul_centre(video_file,seuils_classes_distances,SECONDS_TO_COMPUTE,decalage,nb_total_decalage,numero_decalage):
    #print(f'{SECONDS_TO_COMPUTE=}\t{decalage=}\t{nb_total_decalage=}\t{numero_decalage=}')

    #Définition des résultats des calculs
    distances_totales = {}  # Distances totales parcourues par chaque point
    total_times = {}  # Temps total de suivi pour chaque point
    speed_m_per_sec_par_trajet = {} # Dictionnaire où chaque trajet correspond à une liste qui contient les vitesses prises par chaque point du trajet en m/s
    all_points = [] # Liste pour stocker tous les points de trajectoire
    speeds_m_per_sec = [] # Liste pour stocker les vitesses en m/s pour chaque point
    trajets =  {}

    video_file = cv2.VideoCapture(video_file)
    if not video_file.isOpened():
        print(f"Error while opening video file {video_file}")
        sys.exit()

    # Obtenir le nombre total de frames
    total_frames = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))


    detection_center = frame_width // 2, frame_height // 2
    detection_diameter = int(min(frame_width/2, frame_height/2))
    #print(f'{detection_diameter=}\n{detection_center=}')

    # Obtenir le taux de frames par seconde (fps)
    frames_per_second = round(video_file.get(cv2.CAP_PROP_FPS))
    FRAMES_TO_COMPUTE  = SECONDS_TO_COMPUTE * frames_per_second

    # Calculer la durée en secondes
    movie_length_seconds = round(total_frames / frames_per_second)
    frame_time = 1 / frames_per_second  # Durée d'un frame en secondes

    # Masques pour les zones de détection
    masque_detection = np.zeros((frame_height,frame_width), dtype=np.uint8)  # Crée un masque de la même taille que l'image, mais en niveaux de gris
    # Dessine un cercle plein (rayon 500) sur le masque avec une valeur de 255 (blanc)
    cv2.circle(masque_detection, detection_center, detection_diameter, 255, thickness=-1)

    # Decalage du film pour se mettre au bon endroit pour les calculs
    video_file.set(cv2.CAP_PROP_POS_FRAMES, decalage * frames_per_second)
    frame_available, frame = video_file.read()


    previous_frame_gray = frame_to_BGR2GRAY(frame)
    #image_precedente_grise = frame_to_grey_sum(image_precedente)

    # Utilise le masque full pour tout détecter
    positions_initiales = cv2.goodFeaturesToTrack(previous_frame_gray,mask=masque_detection, **DETECTION_PARAMETERS)
    #positions_initiales = cv2.goodFeaturesToTrack(previous_frame_gray,mask=masque_detection, **DETECTION_PARAMETERS)


    #nb_shift_total =int(movie_length_seconds - window_size_seconds/windows_shift_seconds)
    table_colors = plt.colormaps.get_cmap('plasma').resampled(nb_total_decalage).colors

    # Boucle de traitement pour chaque frame jusqu'à atteindre frames_per_window
    t = trange(FRAMES_TO_COMPUTE, desc=f'{decalage:03} frame analysis ',
               position=numero_decalage,        leave=True,                # the progress bar will be cleared up and the cursor position unchanged when finished

               colour=colors.rgb2hex(table_colors[numero_decalage]))

#    for frame_count in range(FRAMES_TO_COMPUTE):
#        print(f'{decalage:03}  {frame_count:03})')
    for frame_count in t:
        #t.set_postfix(refresh=False)

        #avant_read_frame = time.time()
        frame_available, frame = video_file.read()

        # arret si pb de lecture ==fin de fichier
        if not frame_available:
            break

        #Calcul du flux optique pour suivre les caractéristiques d'une frame à l'autre
        frame_gray = frame_to_BGR2GRAY(frame)
        masque_suivi = np.zeros_like(frame)

        positions_suivies, statuts,err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, positions_initiales, None, **TRACKING_PARAMETERS)

        # Filtrer les bons points suivis dans la nouvelle frame
        points_encore_suivis = positions_suivies[statuts == 1]
        points_initiaux_encore_suivis = positions_initiales[statuts == 1]

        previous_frame_gray = frame_gray.copy()
        positions_initiales = points_encore_suivis.reshape(-1, 1, 2)


        if len(masque_suivi.shape) == 2 or masque_suivi.shape[2] == 1:  # mask est en niveaux de gris
            masque_suivi = cv2.cvtColor(masque_detection, cv2.COLOR_GRAY2BGR)


        for i, (new, old) in enumerate(zip(points_encore_suivis, points_initiaux_encore_suivis)):

            x_new_point, y_new_point = new.ravel()
            x_old_point, y_old_point = old.ravel()

            distance = np.sqrt((x_new_point - x_old_point) ** 2 + (y_new_point - y_old_point) ** 2)

            all_points.append(new)  # Stocker le point

            # Initialisation s'ils n'existent pas déjà
            if i not in distances_totales:
                distances_totales[i] = 0

            # Initialisation s'ils n'existent pas déjà
            if i not in trajets:
                trajets[i] = []
            trajets[i].append(new.ravel())

            # Mise à jour des distances pour chaque point suivi
            distances_totales[i] += distance

                #https://excelatfinance.com/xlf/xlf-colors-1.php

    video_file.release()



    ### Distribution des distances en histogamme
    if False :
        array_distance =[]
        for key in distances_totales.keys():
            #print(key)
            array_distance.append(distances_totales[key])

        cm = plt.colormaps.get_cmap('plasma')

        n, bins, patches = plt.hist(array_distance, color='lightgreen', ec='black', bins=100)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # scale values to interval [0,1]
        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        #plt.show()

    AFFICHAGE_TRAJETS = False

    masque_suivi = np.zeros_like(frame)
    trajets_longs={}
    for key, value in distances_totales.items():
        if value < seuils_classes_distances[0]:
            #Cyan
            color = (0,255,255)
            if AFFICHAGE_TRAJETS :
                for point in trajets[key] :
                    x_new_point, y_new_point = point.ravel()
                    cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), rayon_cercle_largeur_ligne, color , -1)

        if distances_totales[key] >  seuils_classes_distances[0]  and distances_totales[key] <  seuils_classes_distances[1]:
            # Fushia
            color = (255,0,255)
            if AFFICHAGE_TRAJETS :
                for point in trajets[key] :
                    x_new_point, y_new_point = point.ravel()
                    #cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), rayon_cercle_largeur_ligne, color , -1)

        elif distances_totales[key] >  seuils_classes_distances[1]:
            trajets_longs[key]=np.array(trajets[key].copy())
            # Rouge
            color = (0,0,255)

            if AFFICHAGE_TRAJETS :
                for point in trajets[key] :
                    x_new_point, y_new_point = point.ravel()
                    cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), rayon_cercle_largeur_ligne, color , -1)

    # Dessine le cercle de détection sur le masque avec une valeur de 255 (blanc)
    #cv2.circle(masque_suivi, detection_center, detection_diameter, (255,255,255), thickness=20)

    if AFFICHAGE_TRAJETS:
        img = cv2.add(frame, masque_suivi)
        cv2.imshow('frame',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #save_trajet(img, "vitesse.png")


    interpolations = {}
    trajets_coefs_all =[]
    i =0
    #print(f'{decalage:03}  interpolation des trajets long')

    for trajet in trajets_longs.values():
        trajet = np.array(trajet)
        min_trajet_x, max_trajet_x = min(trajet[:,0]), max(trajet[:,0])
        # Create an evenly spaced array that ranges from the minimum to the maximum
        # This will be used as your new x coordinates
        new_trajets_x = np.linspace(min_trajet_x, max_trajet_x, 100)

        # Fit a 1rd degree polynomial to your data
        trajets_coefs = np.polyfit(trajet[:,0],trajet[:,1], 1)
        trajets_coefs_all.append(trajets_coefs)
        interpolation_trajet_x = np.linspace(min_trajet_x, max_trajet_x, 100)
        interpolation_trajet_y = np.polyval(trajets_coefs, interpolation_trajet_x)
        interpolations[i] =[interpolation_trajet_x,interpolation_trajet_y]

        i=i+1

    # find intersections

    # On garde un nb pair de solutions
    if len(trajets_coefs_all) %2 !=0:
        trajets_coefs_all=trajets_coefs_all[:-1]

    solutions=[]
    # random de toutes les interpolations
    np.random.shuffle(trajets_coefs_all)
    #print(f'{decalage:03}  calculs solutions')
    for couple in [(trajets_coefs_all[i],trajets_coefs_all[i+1]) for i in range(0,len(trajets_coefs_all),2)]:
        a =couple[0]
        b = couple[1]
        solutions.append(findIntersection(a,b))


    #plt.imshow(frame)

    # plot interpolations
    #for inter in interpolations:
    #    interpolation_trajet_x = interpolations[inter][0]
    #    interpolation_trajet_y = interpolations[inter][1]
    #    plt.plot(interpolation_trajet_x, interpolation_trajet_y,linewidth=0.5,color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))


    #print(f'{decalage:03}  filtrage solutions')
    filtered_solutions=[]
    for solution in solutions:
        x= solution[0]
        y = solution[1]

        # la solution doit etre dans l'image
        # et  pas etre trop loin du center
        if x>0 and x< frame_width and y > 0 and y < frame_height and math.dist(solution,detection_center)<800:
            filtered_solutions.append(solution)
            #plt.plot(x,y,color='red', marker='o', linestyle='dashed',
            #linewidth=2, markersize=3)
    #print(f'{decalage:03} nb de solutions {len(filtered_solutions)}')

    #print(f'{decalage:03} calcul barycentre')
    return euclidean_barycenter(filtered_solutions)

def get_file_names_from_google(google_sheet_id,root_path,numero_ligne=None):

    video_paths=[]

    CSV_DATA = get_data_from_google_sheet(google_sheet_id)

    # Lire les données jusqu'à la ligne spécifique
    for donnees in CSV_DATA:
        # On arrette quand il n'y a plus de path dans la collone VIDEO_PATH
        if donnees['VIDEO_PATH']:
            numero = int(donnees['NUMERO'])
            date_video = donnees['DATE_VIDEO']
            video_path= root_path / donnees['VIDEO_PATH']
            #print(f'{numero=}\t{date_video=}\t{video_path}')
            if numero_ligne is not None and numero_ligne == numero :
                video_paths.append((video_path,date_video,numero))
            if numero_ligne is  None :
                video_paths.append((video_path,date_video,numero))

    return video_paths


def compute_center(video_data,output_path):
    video_path,date_video, numero = video_data
    input_video_filename = os.path.basename(video_path)

    # if numero >2:
    #     return

    print(f'Computing {input_video_filename}')

    full_video_path= video_path
    # Vérifier si le fichier existe
    if not os.path.isfile(full_video_path):
        print(f"Erreur: Le fichier {full_video_path} n'existe pas.")
        sys.exit()

    liste_centres=[]

    video_file = cv2.VideoCapture(full_video_path)
    if not video_file.isOpened():
        print('#################################################################################################')
        print(f"Error while opening video file {full_video_path}")
        print('#################################################################################################')
        return
    frame_available, background_frame = video_file.read()

    # Obtenir le nombre total de frames
    total_frames = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = round(video_file.get(cv2.CAP_PROP_FPS))
    movie_length_seconds = round(total_frames / frames_per_second)
    frame_time = 1 / frames_per_second  # Durée d'un frame en secondes
    detection_center = frame_width // 2, frame_height // 2

    video_file.release()

    seuils_classes_distances = [70 , 90] # ok pour 5,5
#    seuils_classes_distances = [90 , 100] # ok pour 5,5
    window_size_seconds = 5

    windows_shift_seconds = 5
    movie_length_seconds = min(movie_length_seconds, 100)

    cpu_nb = len(psutil.Process().cpu_affinity())
    decalage_list = list(range(0, movie_length_seconds - window_size_seconds, windows_shift_seconds))
    array_arguments_for_calcul_centre =  [(full_video_path,seuils_classes_distances, window_size_seconds, decalage,len(decalage_list), index ) for index, decalage in enumerate(decalage_list)]

    cpu_nb = min(cpu_nb, len(decalage_list))
    print(f'{cpu_nb=}')
#    print(array_arguments_for_calcul_centre)

    with Pool(processes=cpu_nb,initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        # Utiliser pool.map pour appliquer la fonction calculer_vitesse_bulles à chaque élément
        #  de la array_arguments_for_calculer_vitesse_bulles
        barycenters=pool.starmap(calcul_centre, array_arguments_for_calcul_centre)



    filtered_centers=[]
    plt.imshow(background_frame)
    for center in barycenters:
        x,y = center[0],center[1]
        #print( math.dist(center,detection_center))
        #if x>0 and x< frame_width and y > 0 and y < frame_height and math.dist(center,detection_center)<800:
        if x>0 and x< frame_width and y > 0 and y < frame_height :
            filtered_centers.append(center)
            plt.plot(x,y,color='blue', marker='o', linewidth=2, markersize=1)

    ultimate_center =euclidean_barycenter(filtered_centers)

    print(f'ultimate_center for {date_video} {video_path} {ultimate_center[0]},{ultimate_center[1]}')
    if SHOW_IMAGES or SAVE_PLOTS:
        plt.plot(ultimate_center[0],ultimate_center[1],color='red', marker='o', linewidth=2, markersize=3)
    #plt.plot(centre_sarah[0],centre_sarah[1],color='yellow', marker='o', linewidth=2, markersize=7)
    plt.tight_layout()

    if SHOW_IMAGES :
        plt.show()
    if SAVE_PLOTS:
        center_fig_name=output_path / f'{date_video}_{input_video_filename}_center.png'

        plt.savefig(center_fig_name,dpi=150)
    plt.close()

    return (video_path,int(ultimate_center[0][0]),int(ultimate_center[1][0]))

if __name__ == '__main__':

    video_datas = []
    freeze_support() # For Windows support

    # Get parameters from a shared google sheet
    load_dotenv() # Load secrets from .env
    google_sheet_id = os.getenv("GG_SHEET_ID")
    if google_sheet_id is None:
        print("""
              Id of google sheet is required to process data
              in the .env file
              ex : GG_SHEET_ID=1dfsfsdfljkgmfdjg322RfeDF""")
        sys.exit()

    root_data_path = os.getenv("ROOT_DATA_PATH")
    if root_data_path is None:
        print("""
            Path of films directory is required to process data
            in the .env file
            ex linux/mac : ROOT_DATA_PATH=/data/toto
            ex windows : ROOT_DATA_PATH=e:\\ """)
        sys.exit()

    output_path = os.getenv("OUTPUT_PATH")
    if output_path is None:
        print("""
            Output path for results directory required to process data
            in the .env file
            ex linux/mac : OUTPUT_PATH=/data/results
            ex windows : OUTPUT_PATH=e:\\results """)
        sys.exit()


    root_data_path = Path(root_data_path)
    output_path = Path(output_path)


    parser = argparse.ArgumentParser(description="Trouver le centre de fichier vidéo.")
    #parser.add_argument('-v','--video_path', type=str, help="Le chemin vers le fichier vidéo.")

    parser = argparse.ArgumentParser(description="Numéro de la ligne à traiter dans le fichier Google.")
    parser.add_argument('-n','--numero_ligne', type=int)

    args = parser.parse_args()



    video_datas = get_file_names_from_google(google_sheet_id,root_data_path,args.numero_ligne)

    for ligne in video_datas:
        print(f'{ligne=}')


#    sys.exit()
    resultats = [
        compute_center(video_data, output_path)
        for video_data in video_datas
    ]

    # On retire des resultats les resultats nuls
    clean_resultats =  [x for x in resultats if x is not None]
    #_import__("IPython").embed()

    for resultat in clean_resultats:
        a,b,c = resultat
        print(f'{a}\t({int(b)}, {int(c)})')

    for resultat in clean_resultats:
        a,b,c = resultat
        print(f'({b}, {c})')
