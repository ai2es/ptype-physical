# plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import numpy as np
from sklearn.metrics import confusion_matrix
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import os

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def ptype_hist(df, col, dataset, model_name, bins=None, save_location=None):
    """
    Function to plot a histogram of a specified variable when
    the percent of each ptype is greater than 0.
    """
    ra = df[col][df["ra_percent"] > 0]
    sn = df[col][df["sn_percent"] > 0]
    pl = df[col][df["pl_percent"] > 0]
    fzra = df[col][df["fzra_percent"] > 0]
    classes = ["ra", "sn", "pl", "fzra"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    if bins is None:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, density=True)
            ax.ravel()[p].set_title(f"{dataset} {col} {classes[p]}")
    else:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, bins=bins, density=True)
            ax.ravel()[p].set_title(f"{dataset} {col} {classes[p]}")

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")

    plt.show()


def plot_2d_hist(
    x,
    y,
    model_name,
    bins=None,
    title=None,
    xlabel=None,
    ylabel=None,
    save_location=None,
):
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

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")

    plt.show()


def plot_scatter(
    x, y, model_name, title=None, xlabel=None, ylabel=None, save_location=None
):
    fig, ax = plt.subplots(dpi=150)
    ax.scatter(x, y, s=2, c="k")
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    x1 = np.linspace(-40, 36, 1000)
    y1 = x1
    ax.plot(x1, y1, "-b")
    ax.grid(True, alpha=0.25)

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    data, classes, font_size=10, normalize=None, axis=1, cmap=plt.cm.Blues, save_location=None
):
    """
    Function to plot a confusion matrix using seaborn heatmap

    data: dictonary, generally has test, validate, and training data
    classes: different p-types to be tested, list
    normalize: if you want the confusion matrix to be normalized or not. Needs to be None or 'true'
    """
    if not type(data) is dict:
        raise TypeError("Data neets to be a dictionary")

    fig, axs = plt.subplots(
        nrows=1, ncols=len(data), figsize=(10, 3.5), sharex="col", sharey="row"
    )

    for i, (key, ds) in enumerate(data.items()):
        ax = axs[i]
        cm = confusion_matrix(ds["true_label"], ds["pred_label"], normalize=normalize)

        if normalize == 'true':    
            sns.heatmap(cm,
                annot=True,
                xticklabels=classes,
                yticklabels=classes,
                cmap=cmap,
                vmin=0, 
                vmax=1,
                fmt='.2f',         
                ax=ax)
        elif normalize == None:
            sns.heatmap(cm,
                annot=True,
                xticklabels=classes,
                yticklabels=classes,
                cmap=cmap,
                fmt='.0f',         
                ax=ax)

        ax.set_title(key.title(), fontsize=font_size)
        ax.tick_params(axis='y', rotation=0)
        
        if i == 0:
            ax.set_ylabel("True label", fontsize=font_size)
        ax.set_xlabel("Predicted label", fontsize=font_size)
    
    plt.tight_layout()
    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")


def compute_cov(df, col="pred_conf", quan="uncertainty", ascending=False):
    df = df.copy()
    df = df.sort_values(col, ascending=ascending)
    df["dummy"] = 1
    df[f"cu_{quan}"] = df[quan].cumsum() / df["dummy"].cumsum()
    df[f"cu_{col}"] = df[col].cumsum() / df["dummy"].cumsum()
    df[f"{col}_cov"] = df["dummy"].cumsum() / len(df)
    return df


def coverage_figures(
    test_data, output_cols, colors=None, title=None, save_location=None
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey="col")

    test_data["accuracy"] = (
        test_data["pred_label"] == test_data["true_label"]
    ).values.astype(int)

    _test_data_sorted = compute_cov(test_data, col="pred_conf", quan="accuracy")
    ax1.plot(_test_data_sorted["pred_conf_cov"], _test_data_sorted["cu_accuracy"])

    num_classes = test_data["true_label"].nunique()
    for label in range(num_classes):
        cond = test_data["true_label"] == label
        _test_data_sorted = compute_cov(
            test_data[cond], col="pred_conf", quan="accuracy"
        )
        ax2.plot(
            _test_data_sorted["pred_conf_cov"],
            _test_data_sorted["cu_accuracy"],
            c=colors[label],
        )

    if "evidential" in test_data:
        _test_data_sorted = compute_cov(
            test_data, col="evidential", quan="accuracy", ascending=True
        )
        ax1.plot(
            _test_data_sorted["evidential_cov"],
            _test_data_sorted["cu_accuracy"],
            ls="--",
        )
        for label in range(num_classes):
            c = test_data["true_label"] == label
            _test_data_sorted = compute_cov(
                test_data[c], col="evidential", quan="accuracy", ascending=True
            )
            ax2.plot(
                _test_data_sorted["evidential_cov"],
                _test_data_sorted["cu_accuracy"],
                c=colors[label],
                ls="--",
            )

    if title is not None:
        ax1.set_title(title)

    ax1.set_ylabel("Cumulative accuracy")
    ax1.set_xlabel("Coverage (sorted by confidence/uncertainty)")
    ax2.set_xlabel("Coverage (sorted by confidence/uncertainty)")
    ax1.legend(["Confidence", "Uncertainty"], loc="best")
    ax2.legend(output_cols, loc="best")
    plt.tight_layout()

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")


def conus_plot(
    df, dataset="mping", column="pred_label", title="Predicted", save_path=False
):

    latN = 54.0
    latS = 20.0
    lonW = -63.0
    lonE = -125.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    colors = {0: "lime", 1: "dodgerblue", 2: "red", 3: "black"}

    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    _ = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1, 2, 4, 3]
    if dataset == "ASOS":
        for i in range(4):
            ax.scatter(
                df["rand_lon"][df[column] == i] - 360,
                df["rand_lat"][df[column] == i],
                c=df["true_label"][df[column] == i].map(colors),
                s=3,
                transform=ccrs.PlateCarree(),
                zorder=zorder[i],
                alpha=0.2,
            )
    else:
        for i in range(4):
            ax.scatter(
                df["lon"][df[column] == i] - 360,
                df["lat"][df[column] == i],
                c=df[column][df[column] == i].map(colors),
                s=60,
                transform=ccrs.PlateCarree(),
                zorder=zorder[i],
                alpha=0.2,
            )

    first_day = str(min(df["datetime"])).split(" ")[0]
    last_day = str(max(df["datetime"])).split(" ")[0]
    plt.legend(
        colors.values(),
        labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
        fontsize=24,
        markerscale=3,
        loc="lower right",
    )
    plt.title(f"{dataset} {first_day} to {last_day} {title} Labels", fontsize=30)
    if save_path is not False:
        fn = os.path.join(
            save_path, f"{dataset}_{column}_{first_day}_{last_day}_truelabels.png"
        )
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def labels_video(
    test_data,
    case,
    dataset="mping",
    column="pred_label",
    title="Predicted",
    save_path=False,
):

    latN = 50.0
    latS = 23.0
    lonW = -74.0
    lonE = -120.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    colors = {0: "lime", 1: "dodgerblue", 2: "red", 3: "black"}
    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1, 2, 4, 3]

    def update(k):
        # fig = plt.figure(figsize=(12, 8))
        ax.cla()
        ax.set_extent([lonW, lonE, latS, latN])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))

        case_ids = case[k : (k + 1)]
        CCC = test_data["day"].isin(case_ids)
        df = test_data[CCC].copy()
        if dataset == "ASOS":
            for i in range(4):
                ax.scatter(
                    df["lon"][df[column] == i] - 360,
                    df["lat"][df[column] == i],
                    c=df["true_label"][df[column] == i].map(colors),
                    s=10,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder[i],
                    alpha=0.25,
                )
        else:
            for i in range(4):
                ax.scatter(
                    df["lon"][df[column] == i] - 360,
                    df["lat"][df[column] == i],
                    c=df[column][df[column] == i].map(colors),
                    s=10,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder[i],
                    alpha=0.25,
                )

        first_day = str(min(df["datetime"])).split(" ")[0]
        ax.legend(
            colors.values(),
            labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
            fontsize=10,
            markerscale=1,
            loc="lower right",
        )
        ax.set_title(f"{first_day} {title}", fontsize=12)
        plt.tight_layout()
        return ax

    ani = FuncAnimation(fig, update, frames=np.arange(len(case)))
    plt.show()
    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif, dpi=300)


def video(
    test_data,
    case,
    col="pred_conf",
    label="probability",
    title="NENoreaster",
    save_path=False,
):

    latN = 50.0
    latS = 23.0
    lonW = -74.0
    lonE = -120.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    def update(k):
        ax.cla()
        ax.set_extent([lonW, lonE, latS, latN])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))
        case_ids = case[k : (k + 1)]
        CCC = test_data["day"].isin(case_ids)
        df = test_data[CCC].copy()
        for i in range(4):
            sc = ax.scatter(
                df["lon"][df["pred_label"] == i] - 360,
                df["lat"][df["pred_label"] == i],
                c=df[col][df["pred_label"] == i],
                s=10,
                transform=ccrs.PlateCarree(),
                cmap="cool",
                vmin=0,
                vmax=df[col].max(),
            )
        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.025, shrink=0.9325)
        cbar.set_label(f"{label}", size=12)
        ax.set_title(case_ids[0])
        plt.tight_layout()
        return (ax,)

    ani = FuncAnimation(fig, update, frames=np.arange(len(case)))
    # plt.show()
    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif, dpi=300)
