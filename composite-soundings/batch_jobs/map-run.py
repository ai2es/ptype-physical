import argparse
import sys
sys.path.append('../') # lets us import sounding_utils
import time
from os.path import join

import soundings.utils as utils
import xr_map_reduce as xmr

import argparse
from joblib import load

CASE_DICT = {0: 'kentucky',
             1: 'new_york_1',
             2: 'new_york_2'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process a model initialization.')

    parser.add_argument('-f', help='joblib dump file')
    parser.add_argument('-o', help='outfile')
    args = parser.parse_args()
    
    save_file = f'/glade/work/dkimpara/ptype-aggs/{}'
        print(f'saving to {save_file}\n')
        intermediate_save_file = f'/glade/work/dkimpara/ptype-aggs/{args.o}_{case_study}_inter.dump'
    else:
        raise ValueError('need to pass in outfile name')

    tic = time.time()

    results = load()
    res = xmr.xr_map_reduce(dirpath, args.m, xmr.compute_func,
                            intermediate_save_file, -1)
    res.to_netcdf(save_file)
    
    utils.timer(tic, case_study)