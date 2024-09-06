# -*- coding: utf-8 -*-
import itertools
import os
import sys
import csv
import socket
from multiprocessing import Pool
from dataclasses import dataclass, field
import psutil
import requests
import cv2
import pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata
from scipy import ndimage
from dotenv import load_dotenv
from tqdm import trange,tqdm
from codetiming import Timer
import numpy.ma as ma
import datetime

@dataclass
class DzianiBullage:

    cpu_nb : int
    csv_file : str = None
    google_sheet_id: str = None
    line_number : int = 0

    #Video
    video_path : str = ""
    date_video : str = ""
    frames_per_second : int = 0
    total_frames : int = 0
    frame_width  : int = 0
    frame_height  : int = 0
    frame_time  : float = 0
    movie_length_seconds : float = 0

    ## Analysis
    gsd_hauteur : float = 0
    detection_diameter : int = 0
    interpolation_diameter: int = 0
    detection_center : tuple  = None
    interpolation_center: tuple = None
    colormap =  plt.cm.rainbow
    input_video_filename : str = ""
    output_path : str = ""
    root_data_path : str = "./"

    CELL_SIZE: int =  500
    NB_BUBBLES: int = 3000

    # images DPI:
    DPI_SAVED_IMAGES: int = None

    window_size_seconds : int = 20 #  seconds
    windows_shift_seconds: int = 5 # seconds

    ##### Si DISPLAY_PLOTS est vrai, le graphique est affiché à l'écran
    DISPLAY_PLOTS: bool = False

    ##### Si SAVE_PLOTS est vrai, le graphique est sauvegardé dans le répertoire spécifié
    SAVE_PLOTS: bool = True


    VITESSE_MIN_CLASSES_VITESSES : float = 0.1  # m/s
    VITESSE_MAX_CLASSES_VITESSES : float = 0.4 # m/s

    BORNE_INF_GRAPH : float = 0.25
    BORNE_SUP_GRAPH : float = 0.33


    # Liste pour stocker les résultats
    results: list  = field(default_factory=list)
    results_array: list  = field(default_factory=list)

    def __post_init__(self):
        """
        import des parametres par lecture d'un fichier csv
        les parametres seront dans la classe
        """
        # Parametres depuis CSV
        if self.csv_file:
            with open(self.csv_file, mode='r', newline='', encoding='utf-8') as fichier:
                CSV_DATA = csv.DictReader(fichier)
        # parametres en ligne
        elif self.google_sheet_id:
            print(f'google_sheet_id {self.google_sheet_id}')
            url = f'https://docs.google.com/spreadsheets/d/{self.google_sheet_id}/export?format=csv'
            response = requests.get(url)
            if response.status_code == 200:
                decoded_content = response.content.decode('utf-8')
                CSV_DATA = csv.DictReader(decoded_content.splitlines(), delimiter=',')
            else :
                print(f"Google sheet \n{url} is not available")
                sys.exit()

            # Vérifier que les colonnes nécessaires sont présentes
                # Définir les nouvelles colonnes requises
            colonnes_requises = ['VIDEO_PATH','NUMERO','commentaires',
                                 'VITESSE_MAX_CLASSES_VITESSES',
                                 'seuil', 'DATE_VIDEO', 'GSD_HAUTEUR', 'DIAMETRE_DETECTION',
                                 'DIAMETRE_INTERPOLATION', 'aire_detection_m2',
                                 'aire_interpolation_m2', 'CENTRE_ZONE_DE_DETECTION',
                                 'CENTRE_INTERPOLATION']

            for column in colonnes_requises:
                if column not in CSV_DATA.fieldnames:
                    raise ValueError(f"{column} is missing in the google sheet or in the csv file.")

        # Lire les données jusqu'à la ligne spécifique
                for index, ligne in enumerate(CSV_DATA):
                    if index == self.line_number:
                        donnees = ligne
        self.video_path = self.root_data_path+donnees['VIDEO_PATH']
        self.date_video = donnees['DATE_VIDEO']
        self.gsd_hauteur = float(donnees['GSD_HAUTEUR'])
        self.detection_diameter = int(donnees['DIAMETRE_DETECTION'])
        self.interpolation_diameter = int(donnees['DIAMETRE_INTERPOLATION'])
        self.detection_center = eval(donnees['CENTRE_ZONE_DE_DETECTION'])
        self.interpolation_center = eval(donnees['CENTRE_INTERPOLATION'])
        self.VITESSE_MAX_CLASSES_VITESSES = float(donnees['VITESSE_MAX_CLASSES_VITESSES'])

        self.all_points = []


        # Paramètres pour la détection de coins Shi-Tomasi et le suivi optique Lucas-Kanade
        self.DETECTION_PARAMETERS = dict(maxCorners=self.NB_BUBBLES, qualityLevel=0.1, minDistance=0.5, blockSize=10)
        self.TRACKING_PARAMETERS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



        self.input_video_filename = os.path.basename(self.video_path)
        self.output_path = f'{self.line_number}_resultats_{self.window_size_seconds}s_{self.windows_shift_seconds}s_{self.date_video}_{self.input_video_filename}'
        self.tag_file=f'_{self.line_number}_{self.window_size_seconds}s_{self.windows_shift_seconds}s_{self.date_video}_{self.input_video_filename}'
        self.results_csv_filepath = os.path.join(self.output_path, f'results{self.tag_file}.csv')
        self.results_np_X_filepath = os.path.join(self.output_path, f'results_X_{self.tag_file}.npy')
        self.results_np_Y_filepath = os.path.join(self.output_path, f'results_Y_{self.tag_file}.npy')
        self.results_np_speeds_filepath = os.path.join(self.output_path, f'results_speeds_{self.tag_file}.npy')
        self.results_pickle_filepath = os.path.join(self.output_path, f'results{self.tag_file}.pkl')

        print(f'{self.video_path=}\n{self.date_video=}\n{self.gsd_hauteur=}\n'
              f'{self.detection_diameter=}\n{self.interpolation_diameter=}\n'
              f'{self.detection_center=}\n{self.interpolation_center=}\n'
              f'{self.VITESSE_MAX_CLASSES_VITESSES=}\n{self.output_path}'
              )

        # Nom des fichiers de sorties et répertoires
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        detection_area_pixels= np.pi * ((self.detection_diameter / 2) ** 2)
        detection_area_meters = detection_area_pixels * (self.gsd_hauteur ** 2)
        print(f"L'aire de la zone de detection est de {detection_area_pixels:.2f} pixels")
        print(f"L'aire de la zone de detection est de {detection_area_meters:.2f} m²")




    def speed_to_color(self,speed):
        """
        Convertit une vitesse en une couleur sur une échelle de couleurs
        speed : vitesse à convertir
        """

        norm = plt.Normalize(self.VITESSE_MIN_CLASSES_VITESSES, self.VITESSE_MAX_CLASSES_VITESSES,clip=True)
    #    norm = plt.Normalize(self.VITESSE_MIN_CLASSES_VITESSES, self.VITESSE_MAX_CLASSES_VITESSES,clip=False)

        color = self.colormap(norm(speed))
        return tuple(int(c * 255) for c in color[2::-1])




    def draw_color_scale(self,frame,  position_echelle_couleur, hauteur_echelle_couleur, largeur_echelle_couleur):
        """
        Dessine une échelle de couleurs représentant une gamme de vitesses sur une image
        position_echelle_couleur : position (x, y) de l'échelle sur l'image
        """
        hauteur_echelle_couleur *= 2  # Double la hauteur pour grossir l'échelle
        position_echelle_couleur = (position_echelle_couleur[0], position_echelle_couleur[1] + 100)  # Descend l'échelle de 100 pixels

        for i in range(hauteur_echelle_couleur):
            # Inverser l'échelle de vitesse
            speed = self.VITESSE_MAX_CLASSES_VITESSES - (self.VITESSE_MAX_CLASSES_VITESSES - self.VITESSE_MIN_CLASSES_VITESSES) * (i / hauteur_echelle_couleur)
            color = self.speed_to_color(speed)
            cv2.line(frame, (position_echelle_couleur[0], position_echelle_couleur[1] + i), (position_echelle_couleur[0] + largeur_echelle_couleur, position_echelle_couleur[1] + i), color, 1)

        # Ajouter des annotations pour la vitesse minimale et maximale
        cv2.putText(frame, f'{self.VITESSE_MIN_CLASSES_VITESSES} m/s', (position_echelle_couleur[0] + largeur_echelle_couleur + 10, position_echelle_couleur[1] + hauteur_echelle_couleur), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
        cv2.putText(frame, f'{self.VITESSE_MAX_CLASSES_VITESSES} m/s', (position_echelle_couleur[0] + largeur_echelle_couleur + 5, position_echelle_couleur[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)



    def calculate_speed(self,point1, point2):
        """
        Calcule la vitesse entre deux points sur une image en fonction du FPS (frames per second)
        frames_per_second : nombre de frames par seconde de la vidéo
        """
        # Calculer la distance euclidienne entre les deux points
        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        # Le temps entre les frames est l'inverse du FPS
        time_interval = 1 / self.frames_per_second
        # La vitesse est distance par temps (pixels par seconde dans ce cas)
        return distance / time_interval



    def calculate_moving_average(self,speed_list, window_size):
        """
        Calcule la moyenne mobile d'une liste de vitesses
        - speed_list : liste des vitesses
        - window_size : taille de la fenêtre pour la moyenne mobile
        - return quoi ?
        """
        cumsum = np.cumsum(np.insert(speed_list, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)



    def draw_legend(self,img):
        """
        Dessine une légende sur une image indiquant les classes de vitesse et leurs couleurs respectives
        """
        # Définition des couleurs
        colors = {
            f'Vitesses faibles (< {self.BORNE_INF_GRAPH:.2f} m/s)': (0, 255, 255),  # Jaune
            f'Vitesses moyennes(>={self.BORNE_INF_GRAPH:.2f} - <{self.BORNE_SUP_GRAPH:.2f} m/s)': (0, 165, 255),  # Orange
            f'Vitesses elevees(>= {self.BORNE_SUP_GRAPH:.2f} m/s)': (0, 0, 255)  # Rouge
        }
        # Position de départ pour dessiner la légende
        position_y_debut = 50  # Ajuster selon l'espace disponible en haut de l'image

        # Parcourir chaque couleur dans le dictionnaire pour créer la légende
        for (label, color) in colors.items():
            # Dessiner le rectangle de la couleur
            cv2.rectangle(img, (50, position_y_debut), (200, position_y_debut + 100), color, -1)
            # Ajouter le texte descriptif
            cv2.putText(img, label, (210, position_y_debut + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

            # Incrémenter le y pour le prochain élément de légende, avec plus d'espace
            position_y_debut += 120


    def tracer_vitesse_vs_temps(self,numero_bulles_triees,speed_matrix , time_steps,debut_echantillonnage):

        fig, ax = plt.subplots()
        normalisation = plt.Normalize(vmin=self.VITESSE_MIN_CLASSES_VITESSES, vmax=self.VITESSE_MAX_CLASSES_VITESSES)  # Normalisation des données de vitesse pour l'échelle de couleur

        color_mesh = ax.pcolormesh(time_steps, np.arange(len(numero_bulles_triees)), speed_matrix, cmap=self.colormap, norm=normalisation, shading='auto')
        color_bar_echelle = plt.colorbar(color_mesh, ax=ax)  # Ajout d'une barre de couleurs à l'échelle
        color_bar_echelle.set_label('Speed (m/s)')  # Étiquette de la barre de couleurs

        plt.xlim(0, self.window_size_seconds)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Points suivis par l\'algorithme')
        ax.set_title(f'Evolution des vitesses au cours du temps \nDate de la vidéo: {self.date_video} {self.input_video_filename} {self.window_size_seconds}s {self.windows_shift_seconds}s')

        #ax.text(1.7, 1.02, f'Date de la vidéo: {self.date_video}', transform=ax.transAxes, horizontalalignment='right', fontsize=10, color='black')

        if self.SAVE_PLOTS :
            filename = f'Evolution_des_vitesses_au_cours_du_temps_{self.line_number}_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
            filepath = os.path.join(self.output_path, filename)
            fig.savefig(filepath,dpi=self.DPI_SAVED_IMAGES)

        if self.DISPLAY_PLOTS:
            plt.show()


    def frame_to_BGR2GRAY(self,frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def frame_to_grey_sum(self,frame):
        return (frame[:,:,0]+frame[:,:,1]+frame[:,:,2]).astype(np.uint8)

    def save_trajet(self,masque_suivi, frame,points_encore_suivis,frame_count,debut_echantillonnage):
        self.draw_color_scale(frame, (200, 300), 500, 50)
        cv2.putText(frame, f'Nombre de points suivis: {len(points_encore_suivis)}', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)
        cv2.putText(frame, f'Date de la video: {self.date_video}', (2800, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)

        current_time = frame_count / self.frames_per_second
        time_text = f"Duree: {current_time:.2f}s / {self.window_size_seconds:.2f}s"
        cv2.putText(frame, time_text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)


        img = cv2.add(frame, masque_suivi)
        if self.DISPLAY_PLOTS:
            cv2.imshow('frame', img)

        if self.SAVE_PLOTS :
            filename = f'Trajets_des_bulles_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
            filepath = os.path.join(self.output_path, filename)
            cv2.imwrite(filepath, img)


    def get_video_data(self):

        # Ouvrir la vidéo
        video_file = cv2.VideoCapture(self.video_path)
        if not video_file.isOpened():
            print(f"Erreur: impossible d'ouvrir la vidéo {self.video_path}")
            sys.exit()

        # Obtenir le nombre total de frames
        self.total_frames = int(video_file.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Obtenir le taux de frames par seconde (fps)
        self.frames_per_second = round(video_file.get(cv2.CAP_PROP_FPS))
        # Libérer les ressources
        video_file.release()

        # Calculer la durée en secondes
        self.movie_length_seconds = round(self.total_frames / self.frames_per_second)
        self.frame_time = 1 / self.frames_per_second  # Durée d'un frame en secondes

    def calculer_vitesse_bulles(self, debut_echantillonnage ):

        """
        Calcule et retourne la vitesse des bulles dans une vidéo en utilisant des méthodes de détection de caractéristiques
        et de suivi optique

        return [ debut_echantillonnage, points, speeds]

        """
        print(f'{self.input_video_filename} Calcul calculer_vitesse_bulles for offset {debut_echantillonnage:03} avec une fenetre de {self.window_size_seconds} secondes start')
        nb_shift = int(debut_echantillonnage/self.windows_shift_seconds)

        video_file = cv2.VideoCapture(self.video_path)
        if not video_file.isOpened():
            print(f"Error while opening video file {self.video_path}")
            sys.exit()

        frames_per_window = int(self.window_size_seconds * self.frames_per_second)

        # Decalage du film pour se mettre au bon endroit pour les calculs
        video_file.set(cv2.CAP_PROP_POS_FRAMES, debut_echantillonnage * self.frames_per_second)

        # Lecture de la première frame et création
        frame_available, first_frame = video_file.read()
        if not frame_available:
            print(f"Erreur de lecture de la première frame de {self.input_video_filename}")
            video_file.release()
            sys.exit()
        # Copy de la frame pour autre usage
        first_frame_copy=np.array(first_frame)

        # Cercles de détection et d'interpolation
        # Masque pour définir le cercle de détection
        masque_detection = np.zeros((self.frame_height,self.frame_width), dtype=np.uint8)  # Crée un masque de la même taille que l'image, mais en niveaux de gris
        # Dessine un cercle plein = cercle de détection sur le masque avec une valeur de 255 (blanc)
        cv2.circle(masque_detection, self.detection_center, self.detection_diameter, 255, thickness=-1)

        # #Image avec la position du cercle de détection
        if debut_echantillonnage == 0 :
            cv2.circle(first_frame_copy, self.detection_center, self.detection_diameter, 255, thickness= 2)
            filename = f'Cercle_detection_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
            filepath = os.path.join(self.output_path, filename)
            cv2.imwrite(filepath, first_frame_copy)

        # Détermination des caractéristiques de détection
        first_frame_gray = self.frame_to_BGR2GRAY(first_frame)
        positions_initiales = cv2.goodFeaturesToTrack(first_frame_gray,mask=masque_detection, **self.DETECTION_PARAMETERS)
        masque_suivi = np.zeros_like(first_frame)

        #Définition des résultats des calculs
        distances_totales = {}  # Distances totales parcourues par chaque point
        total_times = {}  # Temps total de suivi pour chaque point
        #all_points = [] # Liste pour stocker toutes les positions X et Y des points
        all_X = [] # Liste pour stocker toutes les positions X des points
        all_Y = [] # Liste pour stocker toutes les positions Y des points
        speeds_m_per_sec = [] # Liste pour stocker les vitesses en m/s pour chaque point
        speed_m_per_sec_par_trajet = {} # Dictionnaire où chaque trajet correspond à une liste qui contient les vitesses prises par chaque point du trajet en m/s


        status_update_seconds= 3

        #table_colors = cmap(np.linspace(0, 1,))
        nb_shift_total =int(self.movie_length_seconds - self.window_size_seconds/self.windows_shift_seconds)
        table_colors = plt.colormaps.get_cmap('plasma').resampled(nb_shift_total).colors
        #custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

        #print(f'{debut_echantillonnage:03} {nb_shift} {colors.rgb2hex(table_colors[nb_shift])}')
        #return []

        # https://matplotlib.org/stable/gallery/color/individual_colors_from_cmap.html

       # Boucle de traitement pour chaque frame jusqu'à atteindre frames_per_window
        t=trange(frames_per_window,desc=f'{debut_echantillonnage:03} ',
                                  mininterval=status_update_seconds,
                                  position=nb_shift,
                                  nrows=40,
                                  colour=colors.rgb2hex(table_colors[nb_shift]))

        #Renommage de la first frame pour l'initiation de la boucle
        previous_frame_gray = first_frame_gray.copy()

        for frame_count in t:
            t.set_postfix(refresh=False)
            #    print(f'{debut_echantillonnage:03} {frame_count}/{frames_per_window}')

            #avant_read_frame = time.time()
            frame_available, frame = video_file.read()
            #apre_read_frame = time.time()
            #print('##############################################################################')
            #print(f'duree  read frame {apre_read_frame - avant_read_frame}')

            if not frame_available:
                break

            #Calcul du flux optique pour suivre les caractéristiques d'une frame à l'autre
            frame_gray = self.frame_to_BGR2GRAY(frame)

            positions_suivies, statuts,_ = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, positions_initiales, None, **self.TRACKING_PARAMETERS)
            #ic(positions_suivies)
            #ic(statuts)
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
                speed_px_per_sec = np.linalg.norm([x_new_point - x_old_point, y_new_point - y_old_point]) / self.frame_time  # Calcule la vitesse en px/sec
                speed_m_per_sec = speed_px_per_sec * self.gsd_hauteur  # Convertit la vitesse en m/sec

                #all_points.append(new)  # Stocker le point
                all_X.append(x_new_point) # Stocker le X du point
                all_Y.append(y_new_point) # Stocker le Y du point

                speeds_m_per_sec.append(speed_m_per_sec)
                #print(f'{speed_m_per_sec=}')
                rayon_cercle_largeur_ligne = 2

                color = self.speed_to_color(speed_m_per_sec)
                #cv2.line(masque_suivi, (int(x_newPoint), int(y_newPoint)), (int(x_oldPoint), int(y_oldPoint)), color, rayon_cercle_largeur_ligne)
                if debut_echantillonnage == 0 :
                    cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), rayon_cercle_largeur_ligne, color, -1)
                #cv2.circle(frame, (int(x_newPoint), int(y_newPoint)), rayon_cercle_largeur_ligne, color, -1)

                if i not in speed_m_per_sec_par_trajet:
                    speed_m_per_sec_par_trajet[i] = []
                speed_m_per_sec_par_trajet[i].append(speed_m_per_sec)


                # Initialisation s'ils n'existent pas déjà
                if i not in distances_totales:
                    distances_totales[i] = 0
                    total_times[i] = 0

                # Mise à jour des distances et des temps
                distances_totales[i] += distance
                total_times[i] += 1

                #initial_positions = positions_initiales.copy()
                #initial_positions = np.array(positions_initiales)


            # on sauve l'image à la derniere frame pour début_echantillonnage = 0
            if debut_echantillonnage == 0 and frame_count == frames_per_window -1 and (self.SAVE_PLOTS or self.DISPLAY_PLOTS):
                self.save_trajet(masque_suivi, frame,points_encore_suivis,frame_count,debut_echantillonnage)

            previous_frame_gray = frame_gray.copy()
            positions_initiales = points_encore_suivis.reshape(-1, 1, 2 )

        video_file.release()

        #print(f'Fin traitement video for offset {debut_echantillonnage:03}')


        #Vitesses au cours du temps
        vitesses_moyennes = {}
        for i, speed_list_per_frame in speed_m_per_sec_par_trajet.items():
            if len(speed_list_per_frame) >= self.frames_per_second:
    #           print(f'{debut_echantillonnage:03} speed_list {speed_list_per_frame}')
                ma_speeds = self.calculate_moving_average(speed_list_per_frame, self.frames_per_second)
                #ma_speeds[::X] prend un point tout les X points
                vitesses_moyennes[i] = ma_speeds[::self.frames_per_second]  # Extraire une moyenne tous les fps frames
    #          print(f'{debut_echantillonnage:03} vitesses_moyennes[i] {i} {vitesses_moyennes[i]}')

    #    for i, speeds in vitesses_moyennes.items():
    #        print(f"Point {i}: Moyenne des vitesses sur chaque seconde = {speeds}")
    #        print(f'type speed {type(speed)} ')
        bubble_ids = list(vitesses_moyennes.keys())
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        #print(f'{debut_echantillonnage:03} (len(speeds) for speeds in vitesses_moyennes.values() {(len(speeds) for speeds in vitesses_moyennes.values())}')
        longest_length = max((len(speeds) for speeds in vitesses_moyennes.values()),default=-9999)

        if  longest_length == -9999:
            print('-------------------------------------------------------------------')
            print(f'{debut_echantillonnage:03} le nb de vitesse calculees dans vitesses_moyennes[0] est {len(vitesses_moyennes[0])}')
            print(f'{debut_echantillonnage:03} mais il en faudrait {self.window_size_seconds}')
            print(f'{debut_echantillonnage:03} donc on sort')
            return []

        time_steps = np.linspace(0, (longest_length - 1) * 1, longest_length)
        #time_steps = np.arange(total_frames) / fps




        speed_matrix = np.full((len(bubble_ids), longest_length), np.nan)

        vitesse_moyenne_totale = {bubble_id: np.mean(speeds) for bubble_id, speeds in speed_m_per_sec_par_trajet.items()}

        #vitesses_globales_moyennes = [np.mean(speeds) for speeds in speed_m_per_sec_par_trajet.values() if len(speeds) > 0]


        # Trier les bulles par leur vitesse moyenne (croissante pour ce cas)
        sorted_bubble_ids = sorted(vitesse_moyenne_totale, key=vitesse_moyenne_totale.get)


        longest_length = max(len(speeds) for speeds in speed_m_per_sec_par_trajet.values())
        speed_matrix = np.full((len(sorted_bubble_ids), longest_length), np.nan)
        time_steps = np.linspace(0, (longest_length - 1) * 1, longest_length)


        for idx, bubble_id in enumerate(sorted_bubble_ids):
            speeds = speed_m_per_sec_par_trajet[bubble_id]
            speed_matrix[idx, :len(speeds)] = speeds

        # Convertir l'axe des temps en secondes (supposons 1 mesure par seconde ici)
        time_steps = np.linspace(0, self.window_size_seconds, frames_per_window)



        # Graph plot VS speed
        #print(f"{debut_echantillonnage:03} tracer_vitesse_vs_temps @@")
        #print(f"{debut_echantillonnage:03} Shapes {sorted_bubble_ids} {speed_matrix} {time_steps}")
        if debut_echantillonnage == 0 :
            self.tracer_vitesse_vs_temps(sorted_bubble_ids,speed_matrix,time_steps,debut_echantillonnage)

        #return [all_points,speeds_m_per_sec]
        return [all_X, all_Y, speeds_m_per_sec]


    def video_file_analysis(self):
        print("Working on the video file ")
        array_arguments_for_calculer_vitesse_bulles =  list(range(0, self.movie_length_seconds - self.window_size_seconds, self.windows_shift_seconds))

        with Pool(processes=self.cpu_nb) as pool:
            # Utiliser pool.map pour appliquer la fonction calculer_vitesse_bulles à chaque élément
            #  de la array_arguments_for_calculer_vitesse_bulles
            results_local=pool.map(self.calculer_vitesse_bulles, array_arguments_for_calculer_vitesse_bulles)

        self.results_array=results_local
        #print(f'results_local.shape {results_local.shape}')
        #print(f'results_array.shape {self.results_array.shape}')
        #__import__("IPython").embed()

        with Timer(text="{name}: {:.4f} seconds", name="Conversion en tableaux NumPy"):
            self.convert_result_to_np()

    def convert_result_to_np(self):
        #taille_data_results = sum(len(item[0]) for item in self.results_array)
        positions_X_tmp = []
        positions_Y_tmp = []
        speeds_tmp = []

        for item in self.results_array:
            Xs = item[0]
            Ys = item[1]
            values = item[2]

            for X, Y, value in zip(Xs,Ys, values):
                positions_X_tmp.append(X)
                positions_Y_tmp.append(Y)
                speeds_tmp.append(value)

        self.np_X = np.array(positions_X_tmp)
        self.np_Y = np.array(positions_Y_tmp)
        self.np_speeds =np.array(speeds_tmp)


    def save_results_numpy(self):

        print('Sauvegarde des resultats en numpy')
        np.save(self.results_np_X_filepath, self.np_X )
        np.save(self.results_np_Y_filepath, self.np_Y )
        np.save(self.results_np_speeds_filepath, self.np_speeds )

    def load_results_numpy(self):
        self.np_X =  np.load(self.results_np_X_filepath )
        self.np_Y =  np.load(self.results_np_Y_filepath )
        self.np_speeds =  np.load(self.results_np_speeds_filepath )




    def process_cell(self,args):
        i, j, x_edges, y_edges, positions_X, positions_Y, speeds, density, max_sample_size, target_density = args

        # Définir les limites de la cellule
        x_lower, x_upper = x_edges[i], x_edges[i + 1]
        y_lower, y_upper = y_edges[j], y_edges[j + 1]

        # Créer le masque
        mask = (positions_X >= x_lower) & (positions_X < x_upper) & (positions_Y >= y_lower) & (positions_Y < y_upper)

        # Points dans la cellule
        cell_positions_X = positions_X[mask]
        cell_positions_Y = positions_Y[mask]
        cell_speeds = speeds[mask]

        # Déterminer le facteur d'échantillonnage
        cell_density = density[i, j]

        sampled_X, sampled_Y, sampled_speeds = [], [], []

        if cell_density > 0:
            # Ajuster le nombre de points échantillonnés proportionnellement à la densité
            sample_size = min(max_sample_size, max(1, int(target_density / (cell_density / np.mean(density) + 1))))

            if len(cell_positions_X) > sample_size:
                indices = np.random.choice(len(cell_positions_X), size=sample_size, replace=False)
                sampled_X = cell_positions_X[indices]
                sampled_Y = cell_positions_Y[indices]
                sampled_speeds = cell_speeds[indices]
            else:
                sampled_X = cell_positions_X
                sampled_Y = cell_positions_Y
                sampled_speeds = cell_speeds

        return sampled_X, sampled_Y, sampled_speeds

    def interpolation(self):


        # Récupéation des datas
        positions_X = self.np_X
        positions_Y = self.np_Y
        speeds = self.np_speeds



        ## Echantillonnage des données pour l'interpolation en fonction de la densité des points
        with Timer(text="{name}: {:.4f} seconds", name="Echantillonnage des données pour l interpolation en fonction de la densité des points"):
            # Découper l'image en cellules d'échantillonnage
            x_min, x_max = np.min(positions_X), np.max(positions_X)
            y_min, y_max = np.min(positions_Y), np.max(positions_Y)
            num_cells = 100  # Ajuster ce paramètre pour plus ou moins de cellules
            x_edges = np.linspace(x_min, x_max, num_cells + 1)
            y_edges = np.linspace(y_min, y_max, num_cells + 1)

            # Liste pour stocker les points échantillonnés
            sampled_positions_X = []
            sampled_positions_Y = []
            sampled_speeds = []

        # Calculer la densité des points dans chaque cellule
        with Timer(text="{name}: {:.4f} seconds", name="Calculer la densité des points dans chaque cellule"):
            density, _, _ = np.histogram2d(positions_X, positions_Y, bins=[x_edges, y_edges])

        # Définir un nombre cible et un nombre max de points à échantillonner
        with Timer(text="{name}: {:.4f} seconds", name="Définir un nombre cible et un nombre max de points à échantillonner"):
            total_points = len(positions_X)
            target_density = total_points / (num_cells ** 2) #nombre cible
            max_sample_size = int(np.ceil(target_density / 1000) * 1000)  #nombre max = nombre cible arrondi au millier supérieur

        # Parcourir chaque cellule
        with Timer(text="{name}: {:.4f} seconds", name="Parcourir chaque cellule"):
            args = [
                (i, j, x_edges, y_edges, positions_X, positions_Y, speeds, density, max_sample_size, target_density)
                for i, j in itertools.product(range(len(x_edges) - 1), range(len(y_edges) - 1))
                ]

            sampled_positions_X, sampled_positions_Y, sampled_speeds = [], [], []

            # Utilisation de Pool pour exécuter la fonction sur plusieurs cœurs
            with Pool(self.cpu_nb) as pool:
                results = pool.map(self.process_cell, args)

            # Combiner les résultats avec np.concatenate
            for res in results:
                sampled_X, sampled_Y, sampled_speeds_ = res
                sampled_positions_X = np.concatenate((sampled_positions_X, sampled_X))
                sampled_positions_Y = np.concatenate((sampled_positions_Y, sampled_Y))
                sampled_speeds = np.concatenate((sampled_speeds, sampled_speeds_))


        # Créer une carte des points échantillonnés à interpoler en gradient de couleur
        with Timer(text="{name}: {:.4f} seconds", name="Créer une carte des points échantillonnés à interpoler en gradient de couleur"):
            fig, ax = plt.subplots(figsize=(10, 8))
            # Tracer les points avec une colormap pour les vitesses
            sc = ax.scatter(sampled_positions_X, sampled_positions_Y, c=sampled_speeds, cmap=self.colormap, vmin=0.1, vmax=0.4, s=10, edgecolor='none')
            plt.colorbar(sc, ax=ax, label='Speed')
            plt.gca().invert_yaxis()
            # Ajouter des étiquettes et un titre
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Scatter Plot of sampled_positions with Speed Colormap')
            # Ajuster les limites des axes si nécessaire
            ax.set_xlim([np.min(sampled_positions_X), np.max(sampled_positions_X)])
            ax.set_ylim([np.min(sampled_positions_Y), np.max(sampled_positions_Y)])
            # Sauvegarder la figure dans un fichier spécifié
            filepath = os.path.join(self.output_path, f'Points_echantillonnes_{self.date_video}.png')
            plt.savefig(filepath, dpi=300)
            plt.close(fig)



        ## Réaliser l'interpolation
        with Timer(text="{name}: {:.4f} seconds", name="Réaliser l interpolation"):
            # Créer une grille avec une résolution fixe
            resolution_x = 200
            resolution_y = 200
            x = np.linspace(x_min, x_max, resolution_x)
            y = np.linspace(y_min, y_max, resolution_y)
            grid_X, grid_Y = np.meshgrid(x, y)

        # Interpolation sur la grille
        with Timer(text="{name}: {:.4f} seconds", name="Interpolation sur la grille"):
            grid_speeds = griddata(
                (sampled_positions_X, sampled_positions_Y),  # Coordonnées des points échantillonnés
                sampled_speeds,                      # Valeurs à interpoler
                (grid_X, grid_Y),                    # Coordonnées de la grille
                method='linear',                     # Méthode d'interpolation
                fill_value=np.nan                    # Valeurs à utiliser pour les points en dehors des données
            )

        # Appliquer un filtre de moyenne pour lisser le signal
        with Timer(text="{name}: {:.4f} seconds", name="Appliquer un filtre de moyenne pour lisser le signal"):
            sigma = 1.5  # Paramètre de lissage
            smoothed_grid_speeds = ndimage.gaussian_filter(grid_speeds, sigma=sigma)

            # Créer une figure du résultat de l'interpolation
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.gca().invert_yaxis()
            contour = ax.contourf(grid_X, grid_Y, smoothed_grid_speeds, cmap=self.colormap, levels=100, vmin=0.1, vmax=0.4)
            cbar = plt.colorbar(contour, ax=ax, label='Speeds')
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_title('Interpolated Grid with Smoothing Applied')

            # Sauvegarder l'image résultante
            filepath = os.path.join(self.output_path,  f'Interpolation_{self.date_video}.png')
            plt.savefig(filepath, dpi=300)
            plt.close(fig)




def main():



    print(datetime.datetime.now())
    file_analysis = False
    interpolation = True

    # nb of CPU to use
    #cpu_nb = cpu_count()
    cpu_nb = len(psutil.Process().cpu_affinity())

    duree_fenetre_analyse_seconde = 20

    root_data_path = './' if 'ncpu' in socket.gethostname() else 'E:/'
    numeros_des_lignes_a_traiter = [11]
    numero_ligne_a_traiter = 11

    duree_fenetre_analyse_seconde = 20
    # Get parameters from a shared google sheet
    load_dotenv() # Load secrets from .env
    google_sheet_id = os.getenv("GG_SHEET_ID")
    if google_sheet_id is None:
        print("""
              Id of google sheet is required to process data
              in the .env file
              ex : GG_SHEET_ID=1dfsfsdfljkgmfdjg322RfeDF""")

    dziani_bullage = DzianiBullage(google_sheet_id=google_sheet_id,line_number=numero_ligne_a_traiter,
                                    root_data_path=root_data_path,window_size_seconds=duree_fenetre_analyse_seconde,
                                    DPI_SAVED_IMAGES=120, DISPLAY_PLOTS=False,cpu_nb=cpu_nb)

    # Get data from video file
    dziani_bullage.get_video_data()
    if file_analysis :

        # modifier la longueur d'analyse du fichier.
        #dziani_bullage.movie_length_seconds = 300

        with Timer(text="{name}: {:.4f} seconds", name="video_file_analysis"):
            dziani_bullage.video_file_analysis()

        #with Timer(text="{name}: {:.4f} seconds", name="save_results_cPickle"):
        #    dziani_bullage.save_results_pickle()

        with Timer(text="{name}: {:.4f} seconds", name="save_results_numpy"):
            dziani_bullage.save_results_numpy()

    if interpolation :
        print("############################")
        print("##### INTERPOLATION #######")
        print("############################")

        # with Timer(text="{name}: {:.4f} seconds", name="load_results_cPickle"):
        #     dziani_bullage.load_results_pickle()

        print(datetime.datetime.now())

        with Timer(text="{name}: {:.4f} seconds", name="load_results_numpy"):
            dziani_bullage.load_results_numpy()

        print(datetime.datetime.now())

        with Timer(text="{name}: {:.4f} seconds", name="interpolation"):
            dziani_bullage.interpolation()

        print(datetime.datetime.now())


if __name__ == "__main__":
    main()
