import argparse
import csv
import datetime
from multiprocessing import cpu_count
import os
from pathlib import Path
import socket
import sys
from codetiming import Timer
from dataclasses import dataclass

import cv2
from dotenv import load_dotenv
from matplotlib import colors, pyplot as plt
import numpy as np
import psutil
from tqdm import trange
from utils import get_data_from_google_sheet


MAX_DIST_PX=5

class opticalflow:
    def __init__(self,google_sheet_id,line_number,root_data_path,output_path,csv_file=False,):

        self.VITESSE_MIN_CLASSES_VITESSES : float = 0.1  # m/s
        self.VITESSE_MAX_CLASSES_VITESSES : float = 1 # m/s
        self.colormap =  plt.cm.rainbow
        self.SAVE_PLOTS = True
        self.DISPLAY_PLOTS = False
        self.output_path  = Path(output_path)
        self.root_data_path = Path(root_data_path)
        self.line_number  =  line_number
        self.NB_BUBBLES = 700
        self.date_video=2023_04_20
        self.window_size_seconds = 20
        self.detection_center = 0
        self.detection_diameter = 0
        self.csv_file = csv_file
        self.google_sheet_id  = google_sheet_id
        self.windows_shift_seconds=5

        self.winSize=23
        self.maxLevel=1
        # Paramètres pour la détection de coins Shi-Tomasi et le suivi optique Lucas-Kanade
        self.DETECTION_PARAMETERS = dict(maxCorners=self.NB_BUBBLES, qualityLevel=0.1, minDistance=0.5, blockSize=10)
        self.TRACKING_PARAMETERS = dict(winSize=(self.winSize, self.winSize), maxLevel=self.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if self.csv_file:
                    with open(self.csv_file, mode='r', newline='', encoding='utf-8') as fichier:
                        CSV_DATA = csv.DictReader(fichier)
                # parametres en ligne
        elif self.google_sheet_id:
            CSV_DATA = get_data_from_google_sheet(self.google_sheet_id)
      # Lire les données jusqu'à la ligne spécifique
        for index, ligne in enumerate(CSV_DATA):
            if index == self.line_number:
                donnees = ligne
        self.video_path = self.root_data_path / donnees['VIDEO_PATH']
        self.date_video = donnees['DATE_VIDEO']
        self.alti_abs_lac=donnees['ALTI_ABS_LAC']
        self.gsd_hauteur = float(donnees['GSD_HAUTEUR'])
        self.detection_diameter = int(donnees['DIAMETRE_DETECTION'])
        #self.interpolation_diameter = int(donnees['DIAMETRE_INTERPOLATION'])
        self.detection_center = eval(donnees['CENTRE_ZONE_DE_DETECTION'])
        self.alti_abs_lac = float(donnees['ALTI_ABS_LAC'])
        self.alti_abs_drone = float(donnees['ALTI_ABS_DRONE'])
        self.distance_lac = self.alti_abs_drone - self.alti_abs_lac
        # Sensor size is in mm
        self.sensor_data = eval(donnees['SENSOR_DATA'])

        self.all_points = []



        self.input_video_filename = os.path.basename(self.video_path)
        self.output_path = self.output_path  / f'{self.line_number}_resultats_{self.window_size_seconds}s_{self.windows_shift_seconds}s_{self.date_video}_{self.input_video_filename}'
        self.tag_file=f'_{self.line_number}_{self.window_size_seconds}s_{self.windows_shift_seconds}s_{self.date_video}_{self.input_video_filename}'
        self.results_csv_filepath = self.output_path / f'results{self.tag_file}.csv'
        self.results_np_X_filepath = self.output_path /  f'results_X_{self.tag_file}.npy'
        self.results_np_Y_filepath = self.output_path /  f'results_Y_{self.tag_file}.npy'
        self.results_np_speeds_filepath = self.output_path /  f'results_speeds_{self.tag_file}.npy'

        self.results_grid_X_filepath = self.output_path /  f'results_grid_X_{self.tag_file}.npy'
        self.results_grid_Y_filepath = self.output_path /  f'results_grid_Y_{self.tag_file}.npy'
        self.results_smoothed_grid_speeds_filepath = self.output_path /  f'results_smoothed_grid_speeds_{self.tag_file}.npy'

        self.results_pickle_filepath = self.output_path /  f'results{self.tag_file}.pkl'



        print(f'{self.video_path=}\n{self.date_video=}\n{self.gsd_hauteur=}\n'
              f'{self.detection_diameter=}\n'
              f'{self.detection_center=}\n'
              f'{self.output_path}'
              )

        # Nom des fichiers de sorties et répertoires
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.get_video_data()
        self.gsd_hauteur=self.get_gsd()

    def get_gsd(self):

        #GSDh= hauteur de vol x hauteur de capteur / longueur focale x hauteur de l'image.
        GSDh = self.distance_lac*self.sensor_data[0] / (self.sensor_data[2] * self.frame_height)
        GSDw = self.distance_lac*self.sensor_data[1] / (self.sensor_data[2] * self.frame_width)
        #print(f'{GSDh}\t{GSDw}\t {min(GSDh,GSDw)}')
        ## 36 * 8.8 / (8.8 * 3648)
        ## 36 * 13.2 / (8.8 * 5472)
        print("####################################################")
        print(f'le GSD calculé du film {self.input_video_filename} et de numero {self.line_number}  est : {min(GSDh,GSDw)}')
        print("####################################################")
        return min(GSDh,GSDw)
        #hauteur de vol x hauteur de capteur / longueur focale x hauteur de l'image.
        #GSDw= hauteur de vol x largeur de capteur / longueur focale x largeur de l'image.

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

        print(f'''{self.total_frames=}
{self.frame_width=}
{self.frame_height=}
{self.frames_per_second=}
{self.movie_length_seconds= }
{self.frame_time=}''')

    def frame_to_BGR2GRAY(self,frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def frame_to_grey_sum(self,frame):
        return (frame[:,:,0]+frame[:,:,1]+frame[:,:,2]).astype(np.uint8)

    def find_particules(self,debut_echantillonnage):
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
        first_frame_copy_2=np.array(first_frame)

        # Masque pour définir le cercle de détection
        masque_detection = np.zeros((self.frame_height,self.frame_width), dtype=np.uint8)  # Crée un masque de la même taille que l'image, mais en niveaux de gris
        # Dessine un cercle plein = cercle de détection sur le masque avec une valeur de 255 (blanc)
        cv2.circle(masque_detection, self.detection_center, self.detection_diameter, 255, thickness=-1)

        # #Image avec la position du cercle de détection
        if debut_echantillonnage == 0 :
            cv2.circle(first_frame_copy, self.detection_center, self.detection_diameter, 255, thickness= 2)
            filename = f'Cercle_detection_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}.png'
            filepath = self.output_path /  filename
            cv2.imwrite(filepath, first_frame_copy)



        # Détermination des caractéristiques de détection
        first_frame_gray = self.frame_to_BGR2GRAY(first_frame)
        positions_initiales = cv2.goodFeaturesToTrack(first_frame_gray,mask=masque_detection, **self.DETECTION_PARAMETERS)
        masque_suivi = np.zeros_like(first_frame)

        #Définition des résultats des calculs
        distances_pixel_totales = {}  # Distances totales parcourues par chaque point
        total_times = {}  # Temps total de suivi pour chaque point
        #all_points = [] # Liste pour stocker toutes les positions X et Y des points
        all_X = [] # Liste pour stocker toutes les positions X des points
        all_Y = [] # Liste pour stocker toutes les positions Y des points
        speeds_m_per_sec = [] # Liste pour stocker les vitesses en m/s pour chaque point

        speed_m_per_sec_par_trajet = {} # Dictionnaire où a chaque trajet correspond à une liste qui contient les vitesses prises par chaque point du trajet en m/s
        all_X_dict = {}
        all_Y_dict = {}

        indices_particules_a_virer = []
        points_high_speed = []


        status_update_seconds= 3

        #table_colors = cmap(np.linspace(0, 1,))
        nb_shift_total =int(self.movie_length_seconds - self.window_size_seconds/self.windows_shift_seconds)
        table_colors = plt.colormaps.get_cmap('plasma').resampled(nb_shift_total).colors


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
                print('frame_available')
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
                distance_pixel = np.sqrt((x_new_point - x_old_point) ** 2 + (y_new_point - y_old_point) ** 2)
                speed_px_per_sec = distance_pixel / self.frame_time  # Calcule la vitesse en px/sec
                speed_m_per_sec = speed_px_per_sec * self.gsd_hauteur  # Convertit la vitesse en m/sec



                if (distance_pixel > MAX_DIST_PX)  :
                #if (speed_m_per_sec> MAX_SPEED_MS) or (distance_pixel*self.gsd_hauteur > (MAX_SPEED_MS/self.frames_per_second)) :
                    # print(f'''@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    #     {i=}\t{x_new_point=}\t{y_new_point=}
                    #     {distance_pixel=}\t{speed_m_per_sec=}\t
                    #       {distance_pixel*self.gsd_hauteur=}\t{MAX_SPEED_MS/self.frames_per_second=}
                    #         ''')

                    indices_particules_a_virer.append(i)
                    points_high_speed.append([(int(x_old_point),int(y_old_point)), (int(x_new_point),int(y_new_point))])

                else :
                    #all_points.append(new)  # Stocker le point
                    all_X.append(x_new_point) # Stocker le X du point
                    all_Y.append(y_new_point) # Stocker le Y du point
                    speeds_m_per_sec.append(speed_m_per_sec)
                    #print(f'{speed_m_per_sec=}')
                    #cv2.line(masque_suivi, (int(x_newPoint), int(y_newPoint)), (int(x_oldPoint), int(y_oldPoint)), color, rayon_cercle_largeur_ligne)



                #figure pour la 1ere sequence
                if debut_echantillonnage == 0 & (distance_pixel < MAX_DIST_PX):
                    rayon_cercle_largeur_ligne = 1
                    color = self.speed_to_color(speed_m_per_sec)
                    cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), rayon_cercle_largeur_ligne, color, -1)

                    # if speed_m_per_sec > MAX_SPEED_MS:
                    #     #print(f'High speed ! {speed_m_per_sec=}  {x_old_point=},{y_old_point=}<=>{x_new_point=},{y_new_point=} ')


                    #     cv2.line(masque_suivi,(int(x_old_point),int(y_old_point)), (int(x_new_point),int(y_new_point)),  (155, 155, 155),20)
                    #     # cv2.circle(masque_suivi, (int(x_old_point), int(y_old_point)), 5, (255, 255, 255), -1)
                    #     # cv2.circle(masque_suivi, (int(x_new_point), int(y_new_point)), 5,  (255, 255, 255), -1)


                #cv2.circle(frame, (int(x_newPoint), int(y_newPoint)), rayon_cercle_largeur_ligne, color, -1)

                # Initialisation s'ils n'existent pas déjà
                if i not in speed_m_per_sec_par_trajet:
                    all_X_dict[i] = []
                    all_Y_dict[i] = []
                    speed_m_per_sec_par_trajet[i] = []
                    distances_pixel_totales[i] = 0
                    total_times[i] = 0
                # Mise à jour des distances et des temps
                all_X_dict[i].append(x_new_point)
                all_Y_dict[i].append(y_new_point)
                speed_m_per_sec_par_trajet[i].append(speed_m_per_sec)
                distances_pixel_totales[i] += distance_pixel
                total_times[i] += 1

            # on sauve l'image  pour début_echantillonnage = 0 à la derniere frame
            if debut_echantillonnage == 0 and frame_count == frames_per_window -1 and (self.SAVE_PLOTS or self.DISPLAY_PLOTS):
                # traits entre les high speed points
                for high_speed_point in points_high_speed:
                    old_p,new_p = high_speed_point
                    #print(f'{old_p=},{new_p=}')
                    cv2.line(first_frame_copy_2,old_p, new_p,  (255, 0, 255),40)

                self.save_trajet(masque_suivi, first_frame_copy_2,points_encore_suivis,frame_count,debut_echantillonnage)

            previous_frame_gray = frame_gray.copy()
            positions_initiales = points_encore_suivis.reshape(-1, 1, 2 )

        video_file.release()

        # np.save(self.output_path /  f'all_X_{self.tag_file}.npy', np.array(all_X))
        # np.save(self.output_path /  f'all_Y_{self.tag_file}.npy', np.array(all_Y))
        # np.save(self.output_path /  f'speeds_m_per_sec_{self.tag_file}.npy', np.array(speeds_m_per_sec))

        #__import__("IPython").embed()
        sys.exit()

        self.tracer_vitesses_vs_temps(speed_m_per_sec_par_trajet,debut_echantillonnage)
        np.save(self.output_path /  f'all_X_dict_{self.tag_file}.npy', all_X_dict)
        np.save(self.output_path /  f'all_Y_dict_{self.tag_file}.npy', all_Y_dict)
        np.save(self.output_path /  f'speed_m_per_sec_par_trajet_dict_{self.tag_file}.npy', speed_m_per_sec_par_trajet)


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
            filename = f'Trajets_des_bulles_{self.date_video}_{self.window_size_seconds}_{debut_echantillonnage:03}_winsize_{self.winSize}_maxLevel_{self.maxLevel}.png'
            filepath = self.output_path /  filename
            cv2.imwrite(filepath, img)


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




    def speed_to_color(self,speed):
        """
        Convertit une vitesse en une couleur sur une échelle de couleurs
        speed : vitesse à convertir
        """

        norm = plt.Normalize(self.VITESSE_MIN_CLASSES_VITESSES, self.VITESSE_MAX_CLASSES_VITESSES,clip=True)
    #    norm = plt.Normalize(self.VITESSE_MIN_CLASSES_VITESSES, self.VITESSE_MAX_CLASSES_VITESSES,clip=False)

        color = self.colormap(norm(speed))
        return tuple(int(c * 255) for c in color[2::-1])




### https://medium.com/thedeephub/object-tracking-and-path-mapping-using-lucas-kanade-optical-flow-in-opencv-2ea018e391d4
### https://medium.com/@VK_Venkatkumar/optical-flow-shi-tomasi-corner-detection-sparse-lucas-kanade-horn-schunck-dense-gunnar-e1dae9600df
### https://github.com/VK-Ant/OpticalFlow_Different-Methods
### https://nanonets.com/blog/optical-flow/

def main():

    parser = argparse.ArgumentParser(description="opticalflow traite un fichier video.")

    # Ajout de l'argument numero_ligne
    # parser.add_argument('numero_ligne', type=int, help='Le numéro de ligne à traiter qui doit être un entier.')
    # parser.add_argument('--file_analysis','-a',default=False, action='store_true', help='flag pour faire l\'analyse du fichier')

    # Parsing des arguments
    args = parser.parse_args()

    print(datetime.datetime.now())




    # Scapad
    if 'ncpu' in socket.gethostname():
        cpu_nb = len(psutil.Process().cpu_affinity())
    # macseb
    elif 'mac' in socket.gethostname():
        cpu_nb = cpu_count()
    # Windows
    else:
        cpu_nb = cpu_count()-1

    print(f'Using {cpu_nb=} CPU on {socket.gethostname()}')

    duree_fenetre_analyse_seconde = 20


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

    opti = opticalflow(google_sheet_id=google_sheet_id,root_data_path=root_data_path,output_path=output_path,line_number=5)
    opti.find_particules(0)

    # if args.file_analysis :

    #     # modifier la longueur d'analyse du fichier.
    #     #dziani_bullage.movie_length_seconds = 300

    #     with Timer(text="{name}: {:.4f} seconds", name="=> video_file_analysis"):



if __name__ == "__main__":
    main()

