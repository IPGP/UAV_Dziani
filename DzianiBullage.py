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




@dataclass
class DzianiBullage:

    csv_file : str = None
    google_sheet_id: str = None
    line_number : int = 0

    #Video
    video_path : str = ""
    date_video : str = ""
    frames_per_second : int = 0
    total_frames : int = 0
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
            colonnes_requises = ['VIDEO_PATH','numero','commentaires',
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
        self.results_npy_filepath = os.path.join(self.output_path, f'results{self.tag_file}.npy')

        print(f'{self.video_path=}\n{self.date_video=}\n{self.gsd_hauteur=}\n'
              f'{self.detection_diameter=}\n{self.interpolation_diameter=}\n'
              f'{self.detection_center=}\n{self.interpolation_center=}\n'
              f'{self.VITESSE_MAX_CLASSES_VITESSES=}\n{self.output_path}'
              )

        self.input_video_filename = os.path.basename(self.video_path)
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



    def tracer_carte_vitesses_interpolees(self,frame, masked_speeds, debut_echantillonnage):

        """
        Trace une carte des vitesses interpolées avec une échelle de couleurs
        """
        # Calcul de l'aire d'un pixel en mètres carrés
        dimension_pixel_grille_m2 = (self.gsd_hauteur * 100) ** 2

        # Dimensions de l'image pour l'affichage
        largeur_pixel_carte_interpolee = frame.shape[1] // 100
        hauteur_pixel_carte_interpolee = frame.shape[0] // 100

        # Création de la figure et affichage des vitesses interpolées
        plt.figure(figsize=(10, 10))
        normalisation = plt.Normalize(vmin=self.VITESSE_MIN_CLASSES_VITESSES, vmax=self.VITESSE_MAX_CLASSES_VITESSES)
        plt.imshow(masked_speeds.T, extent=(0, largeur_pixel_carte_interpolee, hauteur_pixel_carte_interpolee, 0), origin='upper', cmap=self.colormap, norm=normalisation)
        echelle_color_bar = plt.colorbar()
        echelle_color_bar.set_label('Vitesse (m/s)')
        plt.title(f"Carte des vitesses interpolées\n{self.date_video} {self.input_video_filename} {self.window_size_seconds}s {self.windows_shift_seconds}s")


        # Ajout de l'échelle sur le graphique
        position_x_pixel_legende = 30
        position_y_pixel_legende = 18  # Positionner le carré plus bas
        taille_pixel_legende = 1  #avec dans l'échelle de la carte interpolée

        # Ajouter un carré représentant un pixel
        pixel_legende = patches.Rectangle((position_x_pixel_legende, position_y_pixel_legende),
                                        taille_pixel_legende, taille_pixel_legende, linewidth=0, edgecolor='white', facecolor='red')
        plt.gca().add_patch(pixel_legende)

        # Ajout de l'échelle sur le graphique
        legend_text = f"{dimension_pixel_grille_m2:.2f} m²"
        plt.text(position_x_pixel_legende + 0.02, position_y_pixel_legende - 0.02 * hauteur_pixel_carte_interpolee,
                legend_text, color='black', fontsize=8, fontweight='bold')


        if self.SAVE_PLOTS :
            filename = f'Carte_des_vitesses_interpolees_{self.line_number}_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
            filepath = os.path.join(self.output_path, filename)
            plt.savefig(filepath,dpi=self.DPI_SAVED_IMAGES)

        if self.DISPLAY_PLOTS:
            plt.show()



    def tracer_carte_vitesses_integrees_video_totale(self,nouvel_array_moyenne_high_res):
        # Créer une nouvelle figure avec une instance de Figure et Axes

        fig, ax = plt.subplots(figsize=(12, 9))
    #    colormap = plt.get_cmap('Reds') #  Choix de la carte de couleurs
    #    colormap.set_bad(color='white')  # Définir la couleur pour les valeurs NaN à blanc

        normalisation = plt.Normalize(vmin=self.VITESSE_MIN_CLASSES_VITESSES, vmax=self.VITESSE_MAX_CLASSES_VITESSES)
        im = ax.imshow(nouvel_array_moyenne_high_res, cmap=self.colormap, interpolation='bilinear',norm=normalisation)
        ax.set_title(f"Carte des vitesses moyennes intégrées \n{self.date_video} {self.input_video_filename} {self.window_size_seconds}s {self.windows_shift_seconds}s", fontsize=12)

        #Ajout d'une échelle pour l'illustration
        largeur_pixel_carte_interpolee, hauteur_pixel_carte_interpolee = nouvel_array_moyenne_high_res.shape[1], nouvel_array_moyenne_high_res.shape[0]
        # Calcul de l'aire d'un pixel en mètres carrés
        dimension_pixel_grille_m2 = (self.gsd_hauteur * 100) ** 2

        # Ajout de l'échelle sur le graphique
        position_x_pixel_legende, position_y_pixel_legende = largeur_pixel_carte_interpolee * 0.85, hauteur_pixel_carte_interpolee * 0.75
        pixel_size_display = 1
        pixel_legende = patches.Rectangle((position_x_pixel_legende, position_y_pixel_legende), pixel_size_display, pixel_size_display, linewidth=1, edgecolor='red', facecolor='red')
        ax.add_patch(pixel_legende)
        ax.text(position_x_pixel_legende - 2, position_y_pixel_legende - 1, f"{dimension_pixel_grille_m2:.2f} m²", color='black', fontsize=10, fontweight='bold')
        # Ajouter une barre de couleur
        echelle_color_bar=fig.colorbar(im)
        echelle_color_bar.set_label('Vitesse (m/s)')

        if self.SAVE_PLOTS :
            # Chemin et nom du fichier pour sauvegarder
            filename = f'Evolution_des_vitesses_au_cours_du_temps_{self.line_number}_{self.date_video}_final.png'
            filepath = os.path.join(self.output_path, filename)
            fig.savefig(filepath,dpi=self.DPI_SAVED_IMAGES)  # Sauvegarde de la figure

        # Afficher le graphique
        if self.DISPLAY_PLOTS :
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
        # Masque pour définir le cercle d'interpolation
        mask_interpolation = np.zeros((self.frame_height,self.frame_width), dtype=np.uint8)
        # Dessine un cercle plein = cercle d'interpolation sur le masque avec une valeur de 255 (blanc)
        cv2.circle(mask_interpolation, self.interpolation_center, self.interpolation_diameter, 255, -1)
        #Image avec la position des cercles de détection et d'interpolation
        filename = f'Cercles_interpolation_detection_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
        filepath = os.path.join(self.output_path, filename)
        cv2.imwrite(filepath, first_frame_copy)

        # Détermination des caractéristiques de détection
        first_frame_gray = self.frame_to_BGR2GRAY(first_frame)
        positions_initiales = cv2.goodFeaturesToTrack(first_frame_gray,mask=masque_detection, **self.DETECTION_PARAMETERS)
        masque_suivi = np.zeros_like(first_frame)

        #Définition des résultats des calculs
        distances_totales = {}  # Distances totales parcourues par chaque point
        total_times = {}  # Temps total de suivi pour chaque point
        speed_m_per_sec_par_trajet = {} # Dictionnaire où chaque trajet correspond à une liste qui contient les vitesses prises par chaque point du trajet en m/s
        all_points = [] # Liste pour stocker tous les points de trajectoire
        speeds_m_per_sec = [] # Liste pour stocker les vitesses en m/s pour chaque point


        status_update_seconds= 10

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

            positions_suivies, statuts,err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, frame_gray, positions_initiales, None, **self.TRACKING_PARAMETERS)
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

                all_points.append(new)  # Stocker le point
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
                initial_positions = np.array(positions_initiales)


            # on sauve l'image à la derniere frame pour début_echantillonnage = 0
            if debut_echantillonnage == 0 :
                if frame_count == frames_per_window -1:
                    if self.SAVE_PLOTS or self.DISPLAY_PLOTS :
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


        ### Calcul interpolation
        #print(f"Calcul interpolation {debut_echantillonnage:03}")

        # points = np.array(all_points)
        # speeds = np.array(speeds_m_per_sec)


        # # Définition de la grille pour l'interpolation
        # grid_x, grid_y = np.mgrid[0:frame.shape[1]:100, 0:frame.shape[0]:100]

        # # Interpolation des vitesses sur la grille
        # grid_z = griddata(points, speeds, (grid_x, grid_y), method='nearest')

        # masked_speeds = np.where(mask_interpolation[grid_y, grid_x], grid_z, np.nan)
        # nan_speed_mask = np.isnan(grid_z) & (mask_interpolation[grid_y, grid_x] == 255)

        # aire_pixels= np.pi * ((self.interpolation_diameter / 2) ** 2)
        # aire_metres = aire_pixels * (self.gsd_hauteur ** 2)

        # #print(f"L'aire de la zone d'interpolation est de {aire_pixels:.2f} pixels")
        # #print(f"L'aire de la zone d'interpolation est de {aire_metres:.2f} m²")

        # self.tracer_carte_vitesses_interpolees(frame, masked_speeds, debut_echantillonnage)


        # low_speed_mask = (grid_z < self.BORNE_INF_GRAPH) & (mask_interpolation[grid_y, grid_x] == 255)
        # medium_speed_mask = ((grid_z >= self.BORNE_INF_GRAPH) & (grid_z < self.BORNE_SUP_GRAPH)) & (mask_interpolation[grid_y, grid_x] == 255)
        # high_speed_mask = (grid_z >= self.BORNE_SUP_GRAPH) & (mask_interpolation[grid_y, grid_x] == 255)

        # # Calcul de l'aire en pixels pour chaque classe
        # low_speed_area_pixels = np.sum(low_speed_mask)
        # medium_speed_area_pixels = np.sum(medium_speed_mask)
        # high_speed_area_pixels = np.sum(high_speed_mask)

        # mask_active_pixels = np.sum(mask_interpolation[grid_y, grid_x] == 255)

        # # Conversion en mètres carrés en utilisant le GSD
        # low_speed_area_m2_grille = (low_speed_area_pixels * (aire_pixels/mask_active_pixels)) * (self.gsd_hauteur ** 2)
        # medium_speed_area_m2_grille = (medium_speed_area_pixels * (aire_pixels/mask_active_pixels)) * (self.gsd_hauteur ** 2)
        # high_speed_area_m2_grille = (high_speed_area_pixels * (aire_pixels/mask_active_pixels)) * (self.gsd_hauteur ** 2)

        # mask_area_m2 = (mask_active_pixels * (aire_pixels/mask_active_pixels)) * (self.gsd_hauteur ** 2)

        #print(f"Aire des faibles vitesses: {low_speed_area_m2_grille:.2f} m²")
        #print(f"Aire des vitesses moyennes: {medium_speed_area_m2_grille:.2f} m²")
        #print(f"Aire des hautes vitesses: {high_speed_area_m2_grille:.2f} m²")
        #print(f"Aire du masque d'interpolation: {mask_area_m2:.2f} m²")


    #     data_to_save = {
    #     'grid_x': grid_x.tolist(),
    #     'grid_y': grid_y.tolist(),
    #     'grid_z': grid_z.tolist(),
    #     'masked_speeds': masked_speeds.tolist()
    # }
    #     data_filepath = os.path.join(self.output_path, f'donnees_interpolees_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.json')

    #     data_to_save = {
    #     'points': points.tolist(),
    #     'Speeds': speeds.tolist(),
    # }
        # data_filepath = os.path.join(self.output_path, f'donnees_vitesses_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.json')

        #Définir une variable globale = tableau des vitesses en fonction de leur posisition
        #nom_table_vitesse_2 = f'donnees_vitesses_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}'
        #globals ()[nom_table_vitesse_2] = data_to_save

        # # Sauvegarde en JSON
        # with open(data_filepath, 'w') as json_file:
        #     json.dump(data_to_save, json_file)

        #print(f"Les données interpolées ont été sauvegardées dans le fichier {data_filepath}")


        #print(f'Fin calculer_vitesse_bulles for offset {debut_echantillonnage:03}')


        #return [ debut_echantillonnage, low_speed_area_m2_grille, medium_speed_area_m2_grille, high_speed_area_m2_grille]
        #self.results.append([ debut_echantillonnage, low_speed_area_m2_grille, medium_speed_area_m2_grille, high_speed_area_m2_grille])
        #print(self.results)
        #if points:
        #    self.all_points.append(debut_echantillonnage, points) # Liste pour stocker tous les points de trajectoire
        return [all_points,speeds_m_per_sec]
        #return [ debut_echantillonnage, low_speed_area_m2_grille, medium_speed_area_m2_grille, high_speed_area_m2_grille]


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

    def save_results(self):
        # Affichage des résultats pour vérification

        print('Sauvegarde des resultats en Numpy')
        # Convertir la liste des résultats en tableau NumPy
        self.results_array = np.array(self.results)

        # Sauvegarder le tableau NumPy dans un fichier
        #np.save('results.npy', results_array)

        with open(self.results_npy_filepath, 'wb') as f:
            np.save(f, self.results_array)

        print('Sauvegarde des resultats en CSV')
        self.save_results_list_to_csv()

    def save_results_list_to_csv(self):
    #    __import__("IPython").embed()

        header = ['Offset'] + ['low_speed_area_m2_grille'] + ['medium_speed_area_m2_grille']+ ['high_speed_area_m2_grille']
        with open(self.results_csv_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.results)




    def moyennage_part_2(self):
        self.results_array = np.load(self.results_npy_filepath)

        #Ajout carte des vitesses moyennes interpolées intégrée sur toute la vidéo

        # Liste pour stocker les données des fichiers JSON
        self.donnees_json = []

        print('lecture des fichiers JSON')

        # Parcourir tous les fichiers dans le dossier
        for filename in os.listdir(self.output_path):
            # Vérifier si le fichier est un fichier JSON
            if filename.endswith('.json'):
                # Construire le chemin complet vers le fichier JSON
                filepath = os.path.join(self.output_path, filename)
                #print(f'lecture du fichier JSON {filepath}')
                # Ouvrir le fichier JSON en mode lecture
                with open(filepath, 'r') as file:
                    # Charger les données JSON
                    data = json.load(file)
                    # Vérifier si la clé 'masked_speeds' existe dans les données
                    if 'masked_speeds' in data:
                        # Ajouter les données au dictionnaire avec le nom du fichier comme clé
                        self.donnees_json.append({filename[:-5]: data['masked_speeds']})

        # Créer les variables à partir des données importées
        for donnees in self.donnees_json:
            nom_variable, data = list(donnees.items())[0]
            globals()[nom_variable] = np.array(data)

        # Vérifier les noms et les dimensions des variables créées
        #print("Variables créées :", list(locals().keys()))
        print("Variables créées :", list(globals().keys()))

        # Créer une liste contenant tous les tableaux à additionner
        #tous_les_tableaux = [valeur for nom, valeur in locals().items() if nom.startswith(f'donnees_interpolees_{self.date_video}')]
        tous_les_tableaux = [valeur for nom, valeur in globals().items() if nom.startswith(f'donnees_interpolees_{self.date_video}')]

        # Moyenner tous les tableaux à l'aide de np.mean()
        nouvel_array_moyenne = np.mean(tous_les_tableaux, axis=0)

        # Afficher le nouvel array additionné
        #print("Nouvel array moyenné :\n", nouvel_array_moyenne)

        # Additionner tous les tableaux à l'aide de np.sum()
        #nouvel_array_addition = np.sum(tous_les_tableaux, axis=0)

        # Afficher le nouvel array additionné
        #print("Nouvel array additionné :\n", nouvel_array_addition)
        # Créer une image moyenne
        # Tripler la résolution de la grille
        #new_shape = (nouvel_array_moyenne.shape[0] * 1, nouvel_array_moyenne.shape[1] * 3)
        nouvel_array_moyenne_high_res = np.kron(nouvel_array_moyenne, np.ones((1, 1)))

        print("Carte des vitesses moyennes intégrées")
        self.tracer_carte_vitesses_integrees_video_totale(nouvel_array_moyenne_high_res)


def main():


    start_time = time.time()
    now = datetime.datetime.now()
    print('##############################################################################')
    print(f'{now} Start')


    # part = 1 for video treatment
    # part = 2 for moyennage // interpolation
    part = 1

    root_data_path = './' if 'ncpu' in socket.gethostname() else 'E:/'
    numeros_des_lignes_a_traiter = [11]

    duree_fenetre_analyse_seconde = 20
    # Get parameters from a shared google sheet
    # Load secrets from .env
    load_dotenv()
    #csv_input_parameters_file = 'parametres.csv'
    google_sheet_id = os.getenv("GG_SHEET_ID")
    if google_sheet_id is None:
        print('Id of google sheet is required to process data')
        print('in the .env file')
        print('ex : GG_SHEET_ID=1dfsfsdfljkgmfdjg322RfeDF')
    #for numero_ligne in range(0,10) :
    for numero_ligne in numeros_des_lignes_a_traiter :
        dziani_bullage = DzianiBullage(google_sheet_id=google_sheet_id,line_number=numero_ligne,
                                       root_data_path=root_data_path,window_size_seconds=duree_fenetre_analyse_seconde,
                                       DPI_SAVED_IMAGES=120, DISPLAY_PLOTS=False)
        # Get data from video file
        dziani_bullage.get_video_data()

        # modifier la longueur d'analyse du fichier.
        dziani_bullage.movie_length_seconds = 60

        if part == 1:



            # nb of CPU to use
            cpu_nb = cpu_count()
            print("Working on the video file ")
            array_arguments_for_calculer_vitesse_bulles =  list(range(0, dziani_bullage.movie_length_seconds - dziani_bullage.window_size_seconds, dziani_bullage.windows_shift_seconds))

            with Pool(processes=cpu_nb) as pool:
                # Utiliser pool.map pour appliquer la fonction calculer_vitesse_bulles à chaque élément
                #  de la array_arguments_for_calculer_vitesse_bulles
                results_local=pool.map(dziani_bullage.calculer_vitesse_bulles, array_arguments_for_calculer_vitesse_bulles)

            dziani_bullage.results = results_local
            # On sauve les resultats
            #dziani_bullage.save_results()

            #print("Résultats:")
            print(dziani_bullage.results)

            fin_traitement_video = time.time()
            now_fin_traitement_video = datetime.datetime.now()
            print(f'{now_fin_traitement_video} fin traitement_fichier video ')
            print('##############################################################################')
            print(f'duree  {fin_traitement_video - start_time}')
            print("Working on the data ")

            #dziani_bullage.moyennage_part_2()

        elif part == 2 :
            print("Working on the data ")
            dziani_bullage.moyennage_part_2()

if __name__ == "__main__":
    main()
