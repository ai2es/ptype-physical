import argparse
import sys
sys.path.append('../') # lets us import sounding_utils
import time
from os.path import join

import soundings.utils as utils
import xr_map_reduce as xmr

import argparse


CASE_DICT = {0: 'kentucky',
             1: 'new_york_1',
             2: 'new_york_2'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process a model initialization.')

    parser.add_argument('-i', help='case study index')
    parser.add_argument('-m', help='model name')
    parser.add_argument('-o', help='outfile base name (base_name_[case study])')
    args = parser.parse_args()

    case_study = CASE_DICT[int(args.i)]
    
    dirpath = join("/glade/campaign/cisl/aiml/ptype/ptype_case_studies/", case_study) 

    print(f'opening {dirpath}\n')
    
    if args.o:
        save_file = f'/glade/work/dkimpara/ptype-aggs/{args.o}_{case_study}.nc'
        print(f'saving to {save_file}\n')
    else:
        raise ValueError('need to pass in outfile name')

    tic = time.time()

    res = xmr.xr_map_reduce(dirpath, args.m, xmr.compute_func, -1)
    res.to_netcdf(save_file)
    
    utils.timer(tic, case_study)