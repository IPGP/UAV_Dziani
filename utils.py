import csv
import sys
import numpy as np
import requests
from scipy import interpolate
import matplotlib.pyplot as plt

def get_data_from_google_sheet(google_sheet_id):

    print(f'google_sheet_id {google_sheet_id}')
    url = f'https://docs.google.com/spreadsheets/d/{google_sheet_id}/export?format=csv'
    response = requests.get(url)
    if response.status_code == 200:
        decoded_content = response.content.decode('utf-8')
        CSV_DATA = csv.DictReader(decoded_content.splitlines(), delimiter=',')
    else :
        print(f"Google sheet \n{url} is not available")
        sys.exit()

    colonnes_requises = ['VIDEO_PATH','NUMERO','commentaires','VITESSE_MAX_CLASSES_VITESSES',
                                'DATE_VIDEO', 'DATE_AFFICHAGE',
                                'ALTI_ABS_LAC','ALTI_ABS_DRONE','SENSOR_DATA', 'RAYON_DETECTION',
                                'CENTRE_ZONE_DE_DETECTION','CENTRE_INTERPOLATION','MAX_DISTANCE','MAX_SPEED',]
    # Vérification des colonnes
    for column in colonnes_requises:
        if column not in CSV_DATA.fieldnames:
            raise ValueError(f"{column} is missing in the google sheet or in the csv file.")


    return CSV_DATA


def extrapolate_nans(x, y, v):
    if x.size == 0 or y.size == 0:
        # nothing to do
        return np.array([[]])
    if np.ma.is_masked(v):
        nans = v.mask
    else:
        nans = np.isnan(v)
    notnans = np.logical_not(nans)
    v[nans] = interpolate.griddata((x[notnans], y[notnans]), v[notnans],
                                   (x[nans], y[nans]),
                                   method='nearest', rescale=True)
    return v
