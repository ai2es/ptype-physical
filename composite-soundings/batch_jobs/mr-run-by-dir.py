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

    save_file = f'/glade/work/dkimpara/ptype-aggs/{args.o}.nc'
    print(f'saving to {save_file}\n')
    intermediate_save_file = f'/glade/work/dkimpara/ptype-aggs/{args.o}_pre_merge.dump'

    tic = time.time()

    res = xmr.xr_map_reduce(dirpath, args.m, xmr.compute_func,
                            intermediate_save_file, -1)
    res.to_netcdf(save_file)
    
    utils.timer(tic, args.o)