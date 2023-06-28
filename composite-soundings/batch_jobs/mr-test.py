import argparse
import sys
sys.path.append('../') # lets us import sounding_utils
import time
from joblib import Parallel, delayed
import xarray as xr

import sounding_utils
import xr_map_reduce as xmr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process a model initialization.')

    parser.add_argument('-o', help='outfile name')
    args = parser.parse_args()
    
    if args.o:
        print(f'saving to /glade/work/dkimpara/ptype-aggs/{args.o}.nc')
    else:
        raise ValueError('need to pass in outfile dir')

    tic = time.time()

    dirpaths = ['/glade/campaign/cisl/aiml/ptype/ptype_case_studies/kentucky/rap/20220224/0000',
                '/glade/campaign/cisl/aiml/ptype/ptype_case_studies/kentucky/rap/20220224/0600',
                '/glade/campaign/cisl/aiml/ptype/ptype_case_studies/kentucky/rap/20220224/1200']
    
    results = Parallel(n_jobs=-1)(delayed(xmr.xr_map)(path, xmr.compute_func) for path in dirpaths)

    res = xr.concat(results, dim=('time')) #each result ds will be for a different time
    res.to_netcdf(f'/glade/work/dkimpara/ptype-aggs/{args.o}.nc')
    
    sounding_utils.timer(tic)