import numpy as np
import xarray as xr
from collections.abc import Sequence


def mean_from_hist(hist):  # need really small bins
    hist = hist / hist.isel(heightAboveGround=0).sum(dim="bin").values

    return (hist * hist.bin).sum(dim="bin")


class SoundingQuery:
    def __init__(self, datasets, concat_dim=""):
        datasets = self._to_sequence(datasets)
        if concat_dim:
            self.ds = xr.concat(datasets, dim=concat_dim)
        else:
            self.ds = xr.merge(datasets)

    def num_obs(self, predtypes, sel={}):
        predtypes = self._to_sequence(predtypes)
        sel = self._sel_to_list(sel) #converts singleton values to list to keep dims after sel
        ds = self.ds.sel(sel | {'predtype':predtypes})
        return ds['num_obs'].sum(dim=("case_study_day", "step", "init_hr"))

    def query(self, predtypes, variables, stats, sel={}):
        # code to change single inputs to a list
        predtypes = self._to_sequence(predtypes)
        variables = self._to_sequence(variables)
        stats = self._to_sequence(stats)
        sel = self._sel_to_list(sel) #converts singleton values to list to keep dims after sel
        
        ds = self.ds.sel(sel)
        ds = ds.sel({"predtype": predtypes})
        query_vars = [f"{var}_{stat}" for var in variables for stat in stats]
 
        dims_to_reduce = ("case_study_day", "step", "init_hr")
        total_obs = ds["num_obs"].sum(dim=dims_to_reduce)
        res = ds[query_vars] * ds["num_obs"]

        return res.sum(dim=dims_to_reduce) / total_obs

    def quantile(self, quantiles, predtypes, variables, sel={}):
        # code to change single inputs to a list
        quantiles = np.sort(self._to_sequence(quantiles))

        if np.any(quantiles > 1.0) or np.any(quantiles < 0.0):
            raise ValueError("Specified quantiles has value less than 0")

        hist = self.query(predtypes, variables, "hist", sel)

        norm_const = hist.isel(heightAboveGround=0).sum(dim="bin")

        results = []
        for var in list(hist.keys()):
            cdf = self._compute_cdf(hist[var])
            for q in quantiles:
                q_csum = cdf.where(cdf >= (q * norm_const[var]))
                qs = (q_csum
                      .idxmin(dim="bin")
                      .expand_dims({"quantile": [q]})
                      .rename(f'{var[:-5]}_q')
                      )
                results.append(qs)
        return xr.merge(results)

    def _sel_to_list(self, sel):
        for k,v in sel.items():
            sel[k] = self._to_sequence(v)
        return sel

    def _compute_cdf(self, hist):
        csum = hist.cumsum(dim="bin")
        return csum

    def _to_sequence(self, obj):
        if self._seq_but_not_str(obj):
            return obj
        else:
            return [obj]

    def _seq_but_not_str(self, obj):
        return isinstance(obj, (Sequence, np.ndarray)) and not isinstance(
            obj, (str, bytes, bytearray)
        )
