import argparse
import sys
sys.path.append('../') # lets us import sounding_utils
import time

import sounding_utils
import xr_map_reduce as xmr

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process a model initialization.')

    parser.add_argument('-m', help='specify model to run all compute')
    parser.add_argument('-d', help='toplevel dir to run compute')
    parser.add_argument('-o', help='outfile name')
    args = parser.parse_args()

    if args.d:
        dirpath = args.d
    else:
        dirpath = f"/glade/campaign/cisl/aiml/ptype/ptype_case_studies/"

    print(f'opening {dirpath}')
    
    if args.o:
        print(f'saving to /glade/work/dkimpara/ptype-aggs/{args.o}.nc')
    else:
        raise ValueError('need to pass in outfile name')

    tic = time.time()

    res = xmr.xr_map_reduce(dirpath, args.m, xmr.compute_func, -1)
    res.to_netcdf(f'/glade/work/dkimpara/ptype-aggs/{args.o}.nc')
    
    sounding_utils.timer(tic)