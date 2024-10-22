import csv
import sys

import requests


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
                                'DATE_VIDEO', 'ALTI_ABS_LAC','ALTI_ABS_DRONE','SENSOR_DATA',
                                'GSD_HAUTEUR', 'DIAMETRE_DETECTION','DIAMETRE_INTERPOLATION',
                                'CENTRE_ZONE_DE_DETECTION', 'CENTRE_INTERPOLATION']
    # Vérification des colonnes
    for column in colonnes_requises:
        if column not in CSV_DATA.fieldnames:
            raise ValueError(f"{column} is missing in the google sheet or in the csv file.")
    return CSV_DATA
