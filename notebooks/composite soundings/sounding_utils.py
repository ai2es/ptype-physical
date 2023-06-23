import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from joblib import load

from metpy.plots import SkewT
from metpy.calc import (
    relative_humidity_from_dewpoint,
    mixing_ratio_from_relative_humidity,
)
from metpy.units import units

from wrf import wetbulb, enable_xarray
import os

from typing import List

# temp r, dpt bl, wetbulb  c (cyan)

COLOR_DICT = {"t_h": "r", "dpt_h": "b", "wb_h": "c"}
BBOX = {'lon_min': 225.90453, 
        'lon_max': 299.0828, 
        'lat_min': 21.138123, 
        'lat_max': 52.615654
        }

def wet_bulb_from_rel_humid(ds):
    enable_xarray()
    rh_h = relative_humidity_from_dewpoint(ds.t_h * units.degC, ds.dpt_h * units.degC)
    mr_h = mixing_ratio_from_relative_humidity(
        ds.isobaricInhPa_h * units.hPa, ds.t_h * units.degC, rh_h
    )
    ds["wb_h"] = wetbulb(
        ds.isobaricInhPa_h * 100, ds.t_h + 273.15, mr_h, units="degC"
    )  # output wbulb in degC
    return ds


def filter_latlon(ds):
    mask = (
            (ds.latitude <= BBOX['lat_max']) &
            (ds.latitude >= BBOX['lat_min']) &
            (ds.longitude <= BBOX['lon_max']) &
            (ds.longitude >= BBOX['lon_min'])
           )
    return ds.where(mask.compute(), drop=True)


def open_ds_dkimpara(hour, model, concat_dim="time", parallel=False):
    if "glade" in os.getcwd():
        ds = xr.open_mfdataset(
            (
                f"""/glade/scratch/dkimpara/ptype_case_studies/"""
                f"""kentucky/{model}/20220223/{hour}/*.nc"""
            ),
            concat_dim=concat_dim,
            combine="nested",
            parallel=parallel
        )
    else:  # running locally
        ds = xr.open_mfdataset(
            f"""/Users/dkimpara/gh/ptype-data/{model}/{hour}/*.nc""",
            concat_dim=concat_dim,
            combine="nested",
        )
    ds.attrs["nwp"] = model

    # todo: filter latlon

    #if "wb_h" not in list(ds.keys()):
    #    ds = wet_bulb_from_rel_humid(ds)
    return ds


################################################################################################################
######################################## Plotting Code #########################################################
################################################################################################################


### refactoring to use linespecs in the future


# set up the fig and ax
def skewCompositeFigAx(figsize=(5, 5)):
    fig = plt.figure(figsize)
    ax = fig.add_subplot((0, 0, 1, 1), projection="skewx", rotation=30)
    ax.grid(which="both")

    major_ticks = np.arange(-100, 100, 5)
    ax.set_xticks(major_ticks)
    ax.grid(which="major", alpha=0.5)

    # minor_ticks = np.arange(xlowlim - 60, xhighlim, 1)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.grid(which='minor', alpha=0.2)

    ax.axvline(x=0, ymin=0, ymax=1, c="0")

    ax.set_ylabel("Height above ground (m)")
    ax.set_ylim(-100, 5100)
    ax.set_xlim(-20, 20)

    return fig, ax


def frac_abv_zero(ds, x_col, total):
    num_over_zero = (ds[x_col] > 0).any(dim="heightAboveGround").sum().values
    return num_over_zero / total


def plot_masked(
    fig,
    ax,
    ds,
    predtype,
    x_col,
    y,
    c,
    m,
    quantiles,
    plot_median,
    plot_mask=None,
    l_style="solid",
    **kwargs,
):
    subset = ds.where(
        (
            (ds["crain"] == 1)
            | (ds["csnow"] == 1)
            | (ds["cicep"] == 1)
            | (ds["cfrzr"] == 1)
        )
    )
    if plot_mask:
        subset = subset.where(plot_mask)
    else:
        subset = subset.where(ds[predtype] == 1)

    if plot_median:
        profile = subset[x_col].median(dim=("x", "y", "time"))
    else:
        profile = subset[x_col].mean(dim=("x", "y", "time"))

    num_obs = subset[x_col].count(dim=("x", "y", "time")).values[0]
    frac_above_zero = frac_abv_zero(subset, x_col, num_obs)

    if x_col == "dpt_h":
        l_style = "dashed"
    elif x_col == "wb_h":
        l_style = "dotted"
    else:
        l_style = "solid"

    (line,) = ax.plot(profile, y, c, linewidth=2, marker=m, linestyle=l_style)

    line.set_label(
        (
            f'{"" if "ML" in predtype else subset.attrs["nwp"] + "_"}{predtype} '
            f"{x_col}\nnum obs: {num_obs}\nFrac above 0: {frac_above_zero:.2f}"
        )
    )
    if quantiles:
        q_vals = (
            subset[x_col]
            .chunk(dict(time=-1))
            .quantile(
                np.concatenate((quantiles, 1 - quantiles)), dim=("x", "y", "time")
            )
        )
        num_qs = len(quantiles)
        for i in range(num_qs):
            ax.fill_betweenx(y, q_vals[i], q_vals[i + num_qs], alpha=0.1, color=c)
    return fig, ax


def quantile_composites(
    ds: xr.Dataset,
    x_col: str,
    cols: List[str],
    markers: List[str],
    colors: List[str],
    quantiles: List[float] = [0.1],
    plot_median: bool = False,
    xlowlim=-20,
    xhighlim=20,
    y_col: str = "heightAboveGround",
    lstyle: bool = False,
) -> None:
    """function to output composite soundings with shaded quantile regions
    list of quantiles should be a list of the lower quantile to plot (upper equiv plotted automatically)
    x_col: which data variable
    cols: which model preds to mask on
    input list of predictions to plot and corresponding colors"""

    quantiles = np.array(quantiles)

    fig, ax = skewCompositeFigAx()
    y = ds[y_col].values

    for predtype, m, c in zip(cols, markers, colors):
        fig, ax = plot_masked(**locals())
    if quantiles:
        ax.plot(
            [],
            [],
            " ",
            label=f"""Displayed Quantiles: 
                {np.concatenate((quantiles,1-quantiles))}""",
        )

    ax.set_title(
        (
            f"{'Median' if plot_median else 'Mean' } interp. "
            f"{x_col} when models predicts {cols[0][1:]}"
        )
    )

    ax.set_xlabel(x_col)
    ax.set_xlim(xlowlim, xhighlim)
    ax.legend()

    return fig, ax


# distinguishing with colors
def composites_multi_x_v2(
    ds: xr.Dataset,
    x_cols: List[str],
    cols: List[str],
    markers: List[str],
    colors: List[str] = ["b", "g"],
    quantiles: List[float] = [],
    plot_median: bool = False,
    xlowlim=-20,
    xhighlim=20,
    y_col: str = "heightAboveGround",
    lstyle: bool = False,
) -> None:
    """function to output composite soundings for multiple data vars. Quantiles off by default for clarity
    list of quantiles should be a list of the lower quantile to plot (upper equiv plotted automatically)
    input list of predictions to plot and corresponding colors"""

    quantiles = np.array(quantiles)

    fig, ax = skewCompositeFigAx()
    y = ds[y_col].values

    # plot both dpt, temp and wet bulb
    for predtype, m, c in zip(cols, markers, colors):
        for x_col in x_cols:
            fig, ax = plot_masked(**locals())
    if quantiles:
        ax.plot(
            [],
            [],
            " ",
            label=f"""Displayed Quantiles: 
                {np.concatenate((quantiles,1-quantiles))}""",
        )

    ax.set_title(
        (
            f"{'Median' if plot_median else 'Mean'} Interp. "
            f"{x_cols} when models predicts {cols[0][1:]}"
        )
    )

    ax.set_xlabel(x_cols)
    ax.set_xlim(xlowlim, xhighlim)
    ax.legend()
    plt.savefig(f"{cols}.png")

    return fig, ax


def composites_multiplot(
    datasets: List[xr.Dataset],
    x_col: str,
    cols: List[str],
    colors: List[str],
    markers: List[str],  # for differentiating datasets
    plot_median: bool = False,
    prob: float = 0.0,
    y_col: str = "heightAboveGround",
    xlowlim=-20,
    xhighlim=20,
) -> None:
    """function to output composite soundings with multiple NWPs"""

    fig, ax = skewCompositeFigAx()

    for i, ds in enumerate(datasets):
        y = ds[y_col].values
        marker = markers[i]

        for predtype, c in zip(cols, colors):
            if "ML" in predtype:
                subset = ds.where(
                    (ds["crain"] == 1)
                    | (ds["csnow"] == 1)
                    | (ds["cicep"] == 1)
                    | (ds["cfrzr"] == 1)
                )
                if prob > 0.0:
                    subset = subset.where((ds[predtype] > prob))
                else:  # categorical
                    subset = subset.where((ds[predtype] == 1))
            else:
                subset = ds.where(ds[predtype] == 1)

            if plot_median:
                profile = subset[x_col].median(dim=("x", "y", "time"))
            else:
                profile = subset[x_col].mean(dim=("x", "y", "time"))
            (line,) = ax.plot(profile, y, c, linewidth=2, marker=marker, markersize=8)

            # label with num of observations
            num_obs = subset[x_col].count(dim=("x", "y", "time")).values[0]
            frac_above_zero = frac_abv_zero(subset, x_col, num_obs)

            line.set_label(
                (
                    f"{subset.attrs['nwp']}_{predtype}"
                    f"{'>' + str(prob) if ('ML' in predtype) and (prob >0.0) else ''}"
                    f"{x_col}\nnum obs: {num_obs}\nFrac above 0: {frac_above_zero:.2f}"
                )
            )

    ax.set_title(f"{'Median' if plot_median else 'Mean'} Interpolated {x_col}")
    ax.set_xlabel(f"{x_col}")
    ax.set_xlim(xlowlim, xhighlim)

    ax.legend()
    return fig, ax


# distinguishing with markers oly, colors come from COLOR_DICT
def composites_multi_x(
    ds: xr.Dataset,
    x_cols: List[str],
    cols: List[str],
    markers: List[str],
    quantiles: List[float] = [],
    plot_median: bool = False,
    xlowlim=-20,
    xhighlim=20,
    y_col: str = "heightAboveGround",
    lstyle: bool = False,
) -> None:
    """function to output composite soundings for multiple data vars. Quantiles off by default for clarity
    list of quantiles should be a list of the lower quantile to plot (upper equiv plotted automatically)
    input list of predictions to plot and corresponding colors"""

    quantiles = np.array(quantiles)

    fig, ax = skewCompositeFigAx()
    y = ds[y_col].values

    # plot both dpt, temp and wet bulb
    for predtype, m in zip(cols, markers):
        for x_col in x_cols:
            c = COLOR_DICT[x_col]
            fig, ax = plot_masked(**locals())
    if quantiles:
        ax.plot(
            [],
            [],
            " ",
            label=f"""Displayed Quantiles: 
                {np.concatenate((quantiles,1-quantiles))}""",
        )

    ax.set_title(
        f"{'Median' if plot_median else 'Mean'} Interp. {x_cols} when models predicts {cols[0][1:]}"
    )

    ax.set_xlabel(x_cols)
    ax.set_xlim(xlowlim, xhighlim)
    ax.legend()
    plt.savefig(f"{cols}.png")
    return fig, ax
