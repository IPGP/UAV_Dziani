import csv
import sys
import numpy as np
import requests

import dask.array as da
from scipy import interpolate
import matplotlib.pyplot as plt
from dask import delayed


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



def dask_interpolate(dask_valid_x1, dask_valid_y1, dask_valid_z1, dask_xx, dask_yy, algorithm='cubic', vis_out='dask_par.png'):
    # gd_chunked = [delayed(rbf_wrapped)(x1, y1, newarr, xx, yy) for \
    gd_chunked = [delayed(gd_wrapped)(x1.flatten(), y1.flatten(), newarr.flatten(), xx, yy, algorithm) for \
                x1, y1, newarr, xx, yy \
                in \
                zip(dask_valid_x1.to_delayed().flatten(),
                    dask_valid_y1.to_delayed().flatten(),
                    dask_valid_z1.to_delayed().flatten(),
                    dask_xx.to_delayed().flatten(),
                    dask_yy.to_delayed().flatten())]
    gd_out = delayed(da.concatenate)(gd_chunked, axis=0)
    gd_out.visualize(vis_out)
    gd1 = np.array(gd_out.compute())
    print(gd1)
    print(gd1.shape)

    # prove we have no more nans in the data
    assert ~np.isnan(np.sum(gd1))
    return gd1

def dask_gd2(xx, yy, z_array, target_xi, target_yi, algorithm='cubic', **kwargs):
#def dask_gd2( xx, yy, z_array, target_xi, target_yi, ncpu,  algorithm='cubic', **kwargs):
    """!
    @brief general parallel interpolation using dask and griddata
    @param xx 1d or 2d array of x locs where data is known
    @param yy 1d or 2d array of x locs where data is known
    @param z_array 1d or 2d array of x locs where data is known
    @param target_xi 2d array (or 1d grid spacing array)
    @param target_yi 2d array (or 1d grid spacing array)
    """
    n_jobs = kwargs.pop("n_jobs", 4)
    chunk_size = kwargs.get("chunk_size", int(xx.size / (n_jobs - 1)))
    if len(target_xi.shape) < 2:
        xxt, yyt = np.meshgrid(target_xi, target_yi)
    elif len(target_xi.shape) > 2:
        raise RuntimeError
    else:
        xxt, yyt = target_xi, target_yi
    assert xxt.shape == yyt.shape
    z_target = np.full(np.shape(xxt), np.nan)

    # evenly mix nans into dataset.  nans mark where data is needed
    n_splits = n_jobs * 8
    sp_xx, sp_xxt = np.array_split(xx.flatten(), n_splits), np.array_split(xxt.flatten(), n_splits)
    sp_yy, sp_yyt = np.array_split(yy.flatten(), n_splits), np.array_split(yyt.flatten(), n_splits)
    sp_zz, sp_zzt = np.array_split(z_array.flatten(), n_splits), np.array_split(z_target.flatten(), n_splits)

    #print(f'{sp_xx=}\t{sp_xxt=}\t{len(sp_xx)=}\t{len(sp_xxt)=}')
#    print(f'{sp_xxt=}')
    # np.array(sp_xx, sp_xxt)
    all_x = np.concatenate(np.array((sp_xx, sp_xxt)).T.flatten())
    all_y = np.concatenate(np.array((sp_yy, sp_yyt)).T.flatten())
    all_z = np.concatenate(np.array((sp_zz, sp_zzt)).T.flatten())

    # make dask arrays
    import pdb; pdb.set_trace()
    dask_xx = da.from_array(all_x, chunks=chunk_size, name="dask_x")
    dask_yy = da.from_array(all_y, chunks=chunk_size, name="dask_y")
    dask_zz = da.from_array(all_z, chunks=chunk_size, name="dask_z")

    dask_valid_x1 = dask_xx[~da.isnan(dask_zz)]
    dask_valid_y1 = dask_yy[~da.isnan(dask_zz)]
    dask_valid_z1 = dask_zz[~da.isnan(dask_zz)]

    # where to interplate to
    dask_target_x = dask_xx[da.isnan(dask_zz)]
    dask_target_y = dask_yy[da.isnan(dask_zz)]

    # interpolate for missing values
    zz_grid = dask_interpolate(dask_valid_x1, dask_valid_y1, dask_valid_z1, dask_target_x, dask_target_y, algorithm=algorithm, **kwargs)
    return zz_grid.reshape(xxt.shape)





def gd_wrapped(x, y, v, xp, yp, algorithm='cubic', extrapolate=True):
    # source: https://programtalk.com/python-examples/scipy.interpolate.griddata.ravel/
    print("local x.size: ", x.size)
    if x.size == 0 or xp.size == 0:
        # nothing to do
        return np.array([[]])
    if algorithm not in ['cubic', 'linear', 'nearest']:
        raise ValueError("Invalid interpolation algorithm: " + str(algorithm))
    # known data (x, y, v)  can be either 1d or 2d arrays of same size
    # target grid: (xp, yp), xp, yp must be 2d arrays of the same shape
    grid = interpolate.griddata((x, y), v, (xp, yp),
                                method=algorithm, rescale=True)
    if extrapolate and algorithm != 'nearest' and np.any(np.isnan(grid)):
        grid = extrapolate_nans(xp, yp, grid)
    return grid


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
