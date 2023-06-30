import sys
sys.path.append('../') # lets us import sounding_utils
import time

import soundings.utils as utils
import xr_map_reduce as xmr

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process a model initialization.')

    parser.add_argument('-m', help='specify model to run all compute')
    parser.add_argument('-d', help='toplevel dir to run compute')
    parser.add_argument('-o', help='outfile name')
    args = parser.parse_args()

    if not args.d:
        raise ValueError('need to specify toplevel dir')
    
    if not args.o:
        raise ValueError('need to pass in outfile name')
    
    if not args.m:
        raise ValueError('need to pass in model name (rap, hrrr, gfs)')

    dirpath = args.d
    print(f"opening {dirpath}")
    print(f'saving to /glade/work/dkimpara/ptype-aggs/{args.o}.nc')

    tic = time.time()

    res = xmr.xr_map_reduce(dirpath, args.m, xmr.compute_func, -1)
    res.to_netcdf(f'/glade/work/dkimpara/ptype-aggs/{args.o}.nc')
    
    utils.timer(tic)