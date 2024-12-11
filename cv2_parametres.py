# -*- coding: utf-8 -*-
import copy
import os
import sys
import datetime
import csv
import time
import json
import socket
from multiprocessing import Pool,cpu_count
from dataclasses import dataclass, field
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from scipy.interpolate import griddata
from dotenv import load_dotenv
from icecream import ic
from tqdm import trange


def frame_to_BGR2GRAY(self,frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



CELL_SIZE: int =  500
NB_BUBBLES: int = 3000


# Paramètres pour la détection de coins Shi-Tomasi et le suivi optique Lucas-Kanade
DETECTION_PARAMETERS = dict(maxCorners=NB_BUBBLES, qualityLevel=0.1, minDistance=0.5, blockSize=10)
TRACKING_PARAMETERS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))




video_path=('./2024_04_mayotte/24_04_08/DJI_0096.MOV')
detection_diameter = 482
interpolation_diameter = 703
detection_center = (1700, 950)
interpolation_center = (1670, 975)

video_file = cv2.VideoCapture(video_path)
if not video_file.isOpened():
    print(f"Error while opening video file {video_path}")
    sys.exit()

# Obtenir le nombre total de frames
total_frames = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Obtenir le taux de frames par seconde (fps)
frames_per_second = round(video_file.get(cv2.CAP_PROP_FPS))
# Libérer les ressources
video_file.release()

# Calculer la durée en secondes
movie_length_seconds = round(total_frames / frames_per_second)
frame_time = 1 / frames_per_second  # Durée d'un frame en secondes

# Masques pour les zones de détection
masque_detection = np.zeros((frame_height,frame_width), dtype=np.uint8)  # Crée un masque de la même taille que l'image, mais en niveaux de gris
# Dessine un cercle plein (rayon 500) sur le masque avec une valeur de 255 (blanc)
cv2.circle(masque_detection, detection_center, detection_diameter, 255, thickness=-1)
mask_interpolation = np.zeros((frame_height,frame_width), dtype=np.uint8)
cv2.circle(mask_interpolation, interpolation_center, interpolation_diameter, 255, -1)


frame_available, frame = video_file.read()


image_precedente_grise = frame_to_BGR2GRAY(frame)
#image_precedente_grise = frame_to_grey_sum(image_precedente)

# Utilise le masque circulaire pour la détection des caractéristiques
positions_initiales = cv2.goodFeaturesToTrack(image_precedente_grise,mask=masque_detection, **DETECTION_PARAMETERS)

masque_suivi = np.zeros_like(image_precedente)

positions_initiales = cv2.goodFeaturesToTrack(cv2.cvtColor(frame_repartition_positions_initiales, cv2.COLOR_BGR2GRAY), mask=masque_detection, **DETECTION_PARAMETERS)
if positions_initiales is not None:
    initial_positions = positions_initiales.reshape(-1, 2)  # Reshape p0 pour enlever la dimension inutile

    # Accéder aux vitesses moyennes depuis le dictionnaire
    for position, index in zip(initial_positions, range(len(initial_positions))):
        if index in vitesse_moyenne_totale:
            speed = vitesse_moyenne_totale[index]
            color,speed_class= color_and_class_classification_for_speed(speed)
            cv2.circle(frame_repartition_positions_initiales, (int(position[0]), int(position[1])), 5, color, -1)

    draw_legend(frame_repartition_positions_initiales)

    cv2.putText(frame_repartition_positions_initiales, f'Date de la video: {date_video}', (2800, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)

    filename = f'Repartition_des_positions_initiales_des_points_{date_video}_{window_size_seconds}_{debut_enchantillonnage:03}.png'
    filepath = os.path.join(output_path, filename)
    cv2.imwrite(filepath, frame_repartition_positions_initiales)

        #cv2.imshow('Repartition des positions initailes des points', cv2.resize(first_frame, (960, 540)))
        #cv2.waitKey(0)
