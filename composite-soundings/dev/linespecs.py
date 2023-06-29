# in progress
# refactoring plotting code

import numpy as np
import xarray as xr

from dataclasses import dataclass, field
from typing import List

from sounding_utils import skewCompositeFigAx, frac_abv_zero

LSTYLE = {"dpt_h": "dashed", "wbulb": "dotted", "t_h": "solid"}


@dataclass
class LineSpec:
    ds: xr.Dataset
    model: str
    ptype: str
    profile_var: str
    color: str
    marker: str
    lstyle: str = None
    median: bool = False
    prob: float = 0.0
    quantiles: List[float] = field(default_factory=list)
    optionalMask: xr.DataArray = None


def plot_specs(linespecs):
    fig, ax = skewCompositeFigAx()
    for spec in linespecs:
        fig, ax = plot_linespec(fig, ax, **spec.__dict__)
    return fig, ax


def plot_linespec(
    fig,
    ax,
    ds,
    model,
    ptype,
    profile_var,
    color,
    marker,
    lstyle,
    median,
    prob,
    quantiles,
    optionalMask,
):
    ##################### filtering #######################
    if model == "nwp":
        predtype = "c" + ptype
    elif model == "ML":
        predtype = "ML_c" + ptype
    else:
        raise ValueError("invalid LineSpec model must be str of ML or nwp")

    subset = ds.where(
        (
            (ds["crain"] == 1)
            | (ds["csnow"] == 1)
            | (ds["cicep"] == 1)
            | (ds["cfrzr"] == 1)
        )
    )
    subset = subset.where(ds[predtype] == 1)
    if prob:
        subset = subset.where(ds["ML_" + ptype] > prob)
    if optionalMask:
        subset = subset.where(optionalMask)

    #################### plotting ############################
    if median:
        profile = subset[profile_var].median(dim=("x", "y", "time"))
    else:
        profile = subset[profile_var].mean(dim=("x", "y", "time"))

    num_obs = subset[profile_var].count(dim=("x", "y", "time")).values[0]
    frac_above_zero = frac_abv_zero(subset, profile_var, num_obs)

    y = ds.heightAboveGround.values

    if not lstyle:
        lstyle = LSTYLE[profile_var]

    (line,) = ax.plot(profile, y, color, marker=marker, linestyle=lstyle)

    line.set_label(
        (
            f"{'' if 'ML' in predtype else subset.attrs['nwp'] + '_'}{predtype}"
            f"{'>' + str(prob) if ('ML' in predtype) and (prob >0.0) else ''}"
            f" {profile_var}\nnum obs: {num_obs}\nFrac above 0: {frac_above_zero:.2f}"
        )
    )
    ##################### quantiles ###########################
    quantiles = np.array(quantiles)
    if quantiles:
        q_vals = (
            subset[profile_var]
            .chunk(dict(time=-1))
            .quantile(
                np.concatenate((quantiles, 1 - quantiles)), dim=("x", "y", "time")
            )
        )
        num_qs = len(quantiles)
        for i in range(num_qs):
            ax.fill_betweenx(y, q_vals[i], q_vals[i + num_qs], alpha=0.1, color=color)
    return fig, ax, line
