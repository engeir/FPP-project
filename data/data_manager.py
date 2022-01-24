#!/home/een023/.virtualenvs/uit_scripts/bin/python
"""This script includes helper functions to read / load data from different txt files.
"""

import numpy as np
import matplotlib.pyplot as plt


def open_recons_files():
    # M08, BHM, CPS, DA, OIE, PAI, PCR
    files = look_at_recons('PCR')
    print(len(files[0][0]), files[1].shape)


def look_at_recons(file_name):
    """Load the txt files related to the PAGES2k paper.

    The available files / simulations are
        M08, BHM, CPS, DA, OIE, PAI, PCR

    Args:
        file_name (str): the name of the file / simulation

    Returns:
        list: first element is the description, second is a numpy array of the data
    """
    with open(f'data/recons/{file_name}.txt', 'r') as f:
        b = f.readlines()
        b = [x.strip() for x in b]
        b = [x.split('\t') for x in b]
    b = np.asarray(b)
    a1 = np.atleast_2d(b[0, :])
    a2 = b[1:, :].astype(float)
    return [a1, a2]


def look_at_txt(file_name):
    """Open a txt file with only numeric data in one single column.

    Args:
        file_name (str): the path to the file you want to load

    Returns:
        np.ndarray: numpy array of the data
    """
    with open(file_name, 'r') as f:
        b = f.readlines()
        b = [x.strip() for x in b]

    return np.asarray(b).astype(float)


def look_at_full_ensemble():
    """Look at the ensemble data from the PAGES2k paper.

    Similar to `look_at_recons`, except this consider the ensemble file.

    Returns:
        list: first element is the description, second is a numpy array of the data
    """
    with open('data/recons/Full_ensemble_median_and 95pct_range.txt', 'r') as f:
        b = f.readlines()
        b = [x.strip() for x in b]
        b = [x.split('\t') for x in b]
    a1 = b[4]
    b = b[4:]
    a = np.array([])
    for el in b[1:]:
        l = np.genfromtxt(el)
        if a.shape == (0,):
            a = np.r_[a, l]
        else:
            a = np.c_[a, l]
    return [a1, a.T]


def look_at_jones_mann():
    """Look at the txt file from figure 7 of the Jones and Mann paper.

    Returns:
        list: first element is the description, second is a numpy array of the data
    """
    with open('data/jones_mann/jones_mann_fig7.txt', 'r') as f:
        b = f.readlines()
        b = [x.strip() for x in b]
        b = [x.split() for x in b]
    a1 = b[:17]
    b = b[17:]
    a2 = np.asarray(b).astype(float)
    return [a1, a2]


def plot_list_data(the_type, v='M08'):
    plt.figure()
    if the_type == 'jones_mann':
        a = look_at_jones_mann()
        x = a[1][:, 0]
        y = a[1][:, 1:]
        y[y == -999.990] = np.nan
        plt.plot(x, y)
        plt.legend(['GHG (Crowley)',
                    'Aer (Crowley)',
                    'Solar (Crowley)',
                    'Volc(Crowley)',
                    'Volc(Ammann) average of v1, v2, v3 and v4',
                    'Solar(Ammann)',
                    'v1(Ammann for 0-30N)',
                    'v2(Ammann for 0-30S)',
                    'v3(Ammann for 30-90N)',
                    'v4(Ammann for 30-90S)',
                    r'Solar(Bertrand - $\mathrm{TSI_L}$)',
                    r'Volc(Bertrand - $\mathrm{VOLC_C}$)'])
    elif the_type == 'pages_ens':
        a = look_at_full_ensemble()
        x = a[1][:, 0]
        y = a[1][:, 1:]
        plt.plot(x, y)
        plt.legend(['Cowtan & Way instrumental target',
                    'Full ensemble median',
                    'Full ensemble 2.5th percentile',
                    'Full ensemble 97.5th percentile',
                    'Cowtan & Way instrumental target 31-year filtered',
                    '31-year filtered full ensemble median',
                    '31-year filtered full ensemble 2.5th percentile',
                    '31-year filtered full ensemble 97.5th percentile'])
    elif the_type == 'pages':
        a = look_at_recons(v)
        x = a[1][:, 0]
        y = a[1][:, 1:100]
        plt.plot(x, y)
    plt.show()
