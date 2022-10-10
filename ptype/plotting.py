import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from cartopy import crs as ccrs
from cartopy import feature as cfeature


def ptype_hist(df, col, dataset, model_name, bins=None, filename=None):
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
        path = f'/glade/u/home/jwillson/winter-ptype/images/{model_name}/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    plt.show()
        
        
def plot_2d_hist(x, y, model_name, bins=None, title=None, xlabel=None, ylabel=None, filename=None):
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
        path = f'/glade/u/home/jwillson/winter-ptype/images/{model_name}/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
    
    plt.show()
    
def plot_scatter(x, y, model_name, title=None, xlabel=None, ylabel=None, filename=None):
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
        path = f'/glade/u/home/jwillson/winter-ptype/images/{model_name}/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
    
    plt.show()
        
def plot_confusion_matrix(y_true, y_pred, classes, model_name, normalize=False, title=None, cmap=plt.cm.Blues, filename=None):
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
    ax.figure.colorbar(im, ax=ax, shrink=0.80)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(classes, fontsize=18)
    ax.set_yticklabels(classes, fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_ylabel('True label', fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                   fontsize=18)
            
    if filename:
        path = f'/glade/u/home/jwillson/winter-ptype/images/{model_name}/'
        plt.savefig(path + filename, dpi=300, bbox_inches="tight")
        
    return ax


def conus_plot(df, 
               dataset = "mping", 
               column = "pred_label", 
               title = "Predicted", 
               save_path = False):
    
    latN = 54.0
    latS = 20.0
    lonW = -63.0
    lonE = -125.0
    cLat = (latN + latS)/2
    cLon = (lonW + lonE )/2
    colors = {0:'lime', 1:'dodgerblue', 2:'red', 3:'black'}

    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = '50m'  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1,2,4,3]
    if dataset == 'ASOS':
        df['rand_lon'] = [df['lon'].to_numpy()[i]+np.random.normal(scale=scale) for i in range(len(df['lon']))]
        df['rand_lat'] = [df['lat'].to_numpy()[i]+np.random.normal(scale=scale) for i in range(len(df['lat']))]
        for i in range(4):
            ax.scatter(df["rand_lon"][df[column] == i]-360,
                       df["rand_lat"][df[column] == i],
                       c=df["true_label"][df[column] == i].map(colors),
                       s=3, transform=ccrs.PlateCarree(), zorder=zorder[i], alpha = 0.2)
    else:
        for i in range(4):
            ax.scatter(df["lon"][df[column] == i]-360,
                       df["lat"][df[column] == i],
                       c=df[column][df[column] == i].map(colors),
                       s=60, transform=ccrs.PlateCarree(), zorder=zorder[i], alpha = 0.2)

    first_day = str(min(df['datetime'])).split(' ')[0]
    last_day = str(max(df['datetime'])).split(' ')[0]
    plt.legend(colors.values(), labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=24, markerscale=3, loc="lower right")
    plt.title(f"{dataset} {first_day} to {last_day} {title} Labels", fontsize=30)
    if save_path is not False:
        fn = os.path.join(save_path, f'{image_path}_{timeframe}_truelabels.png')
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()