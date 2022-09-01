import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def ptype_hist(df, col, dataset, bins=None, filename=None):
    """
    Function to plot a histogram of a specified variable when
    the percent of each ptype is greater than 0.
    """
    ra = df[col][df['ra_percent'] > 0]
    sn = df[col][df['sn_percent'] > 0]
    pl = df[col][df['pl_percent'] > 0]
    fzra = df[col][df['fzra_percent'] > 0]
    classes = ['ra', 'sn', 'pl', 'fzra']
    
    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    
    if bins is None:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, density=True)
            ax.ravel()[p].set_title(f'{dataset} {col} {classes[p]}')
    else:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, bins=bins, density=True)
            ax.ravel()[p].set_title(f'{dataset} {col} {classes[p]}')
        
    if filename:
        path = '/glade/u/home/jwillson/winter-ptype/images/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    plt.show()
        
        
def plot_2d_hist(x, y, bins=None, title=None, xlabel=None, ylabel=None, filename=None):
    """
    Function to plot a 2D histogram of the joint 
    distribution of 2 variables.
    """
    fig, ax = plt.subplots(dpi=150)
    cmap = cm.binary
    if bins:
        ax.hist2d(x, y, bins, cmap=cmap) 
    else:
        ax.hist2d(x, y, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    plt.colorbar(cm.ScalarMappable(cmap=cmap))
    ax.grid(True, alpha=0.25)

    if filename:
        path = '/glade/u/home/jwillson/winter-ptype/images/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
    
    plt.show()
    
def plot_scatter(x, y, title=None, xlabel=None, ylabel=None, filename=None):
    fig, ax = plt.subplots(dpi=150)
    ax.scatter(x, y, s=2, c='k')
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    x1 = np.linspace(-40, 36, 1000)
    y1 = x1
    ax.plot(x1, y1, '-b')
    ax.grid(True, alpha=0.25)

    if filename:
        path = '/glade/u/home/jwillson/winter-ptype/images/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
    
    plt.show()
        
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, filename=None):
    """
    Function to plot a confusion matrix. 
    """
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
        path = '/glade/u/home/jwillson/winter-ptype/images/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    return ax