# -*- coding: utf-8 -*-
import socket
from multiprocessing import Pool,cpu_count
import time
import os
import datetime
from dotenv import load_dotenv
from DzianiBullage import DzianiBullage
from icecream import ic



if __name__ == '__main__':


    # Where are the data relative to this script
    root_data_path='E:/'

    start_time = time.time()
    now = datetime.datetime.now()
    print('##############################################################################')
    print(f'{now} Start')


    # part = 1 for video treatment
    # part = 2 for moyennage // interpolation
    part = 1

    # Where are the data relative to this script
    root_data_path='E:/'
    cpu_nb = 10

    if 'ncpu' in socket.gethostname() or 'ipgp' in socket.gethostname() or '.local' in socket.gethostname():
        root_data_path='./'
        cpu_nb = cpu_count()

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
            print("Working on the video file ")
            array_arguments_for_calculer_vitesse_bulles =  list(range(0, dziani_bullage.movie_length_seconds - dziani_bullage.window_size_seconds, dziani_bullage.windows_shift_seconds))

            with Pool(processes=cpu_nb) as pool:
                # Utiliser pool.map pour appliquer la fonction calculer_vitesse_bulles à chaque élément
                #  de la array_arguments_for_calculer_vitesse_bulles
                results_local=pool.map(dziani_bullage.calculer_vitesse_bulles, array_arguments_for_calculer_vitesse_bulles)

            dziani_bullage.results = results_local
            # On sauve les resultats
            #dziani_bullage.save_results()

            print("les résultats sont dans dziani_bullage.results")

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
