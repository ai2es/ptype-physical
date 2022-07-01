import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
        
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, filename=None):
    if not title:
        if normalize:
            title = 'Confusion Matrix (normalized)'
        else:
            title = 'Confusion Matrix'

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(classes, fontsize=14)
    ax.set_yticklabels(classes, fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
            
    if filename:
        path = '/glade/u/home/jwillson/ptype-physical/wpt-jwillson/images/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    return ax