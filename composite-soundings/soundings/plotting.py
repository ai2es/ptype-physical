import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils import *
from typing import List


COLOR_DICT = {"t_h": "r", "dpt_h": "b", "wb_h": "c"}


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