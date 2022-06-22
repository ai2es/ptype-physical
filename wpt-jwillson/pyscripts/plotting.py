import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def ptype_hist(df, col, dataset, bins=None):
    ra = df[col][df['ra_percent'] > 0]
    sn = df[col][df['sn_percent'] > 0]
    pl = df[col][df['pl_percent'] > 0]
    fzra = df[col][df['fzra_percent'] > 0]
    
    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    
    if bins is None:
        ax[0,0].hist(ra, density=True)
        ax[0,0].set_title('{} {} ra'.format(dataset, col))
        ax[0,1].hist(sn, density=True)
        ax[0,1].set_title('{} {} sn'.format(dataset, col))
        ax[1,0].hist(pl, density=True)
        ax[1,0].set_title('{} {} pl'.format(dataset, col))
        ax[1,1].hist(fzra, density=True)
        ax[1,1].set_title('{} {} fzra'.format(dataset, col))
    else:
        ax[0,0].hist(ra, bins=bins, density=True)
        ax[0,0].set_title('{} {} ra'.format(dataset, col))
        ax[0,1].hist(sn, bins=bins, density=True)
        ax[0,1].set_title('{} {} sn'.format(dataset, col))
        ax[1,0].hist(pl, bins=bins, density=True)
        ax[1,0].set_title('{} {} pl'.format(dataset, col))
        ax[1,1].hist(fzra, bins=bins, density=True)
        ax[1,1].set_title('{} {} fzra'.format(dataset, col))