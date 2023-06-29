import numpy as np
import xarray as xr

from xhistogram.xarray import histogram

def quantile_from_hist(quantiles, hist):
    norm_const = hist.isel(heightAboveGround=0).sum(dim='bin').values
    quantiles = np.sort(quantiles) * norm_const
    cdf = compute_cdf(hist)
    
    results = []
    for q in quantiles:
        q_csum = cdf.where(cdf >= q)
        qs = q_csum.idxmin(dim='bin')
        qs = qs.expand_dims({'quantile': [q / norm_const]})
        
        results.append(qs)
    return xr.concat(results, dim='quantile')
    
def compute_cdf(hist):
    csum = hist.cumsum(dim='bin')
    return csum
    
def mean_from_hist(hist): #need really small bins
    hist = hist / hist.isel(heightAboveGround=0).sum(dim='bin').values

    return (hist * hist.bin).sum(dim='bin')