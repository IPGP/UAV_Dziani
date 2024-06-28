# -*- coding: utf-8 -*-
from DzianiBullage import DzianiBullage
from dotenv import load_dotenv
import os

# part = 1 for video treatment
# part = 2 for moyennage // interpolation
part = 1

# Where are the data relative to this script
root_data_path='./'

numeros_des_lignes_a_traiter = 11

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
dziani_bullage = DzianiBullage(google_sheet_id=google_sheet_id,numero_ligne=numeros_des_lignes_a_traiter,
                                root_data_path=root_data_path,duree_analyse=duree_fenetre_analyse_seconde,
                                DPI_SAVED_IMAGES=120)
dziani_bullage.moyennage_part_2()
