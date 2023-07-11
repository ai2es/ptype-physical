from os.path import join
import argparse
from functools import partial

CASE_DICT = {0: "kentucky", 1: "new_york_1", 2: "new_york_2"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process a case study.")
    parser.add_argument("-i", help="case study index")
    parser.add_argument("-m", help="model name")
    parser.add_argument("-o", help="outfile base name (base_name_[case study])")
    parser.add_argument("-d", help="toplevel dir of all the files to run compute")
    parser.add_argument('-f', help="function to compute, will be evaluated by python. ds needs to be the last argument and unfilled")
    args = parser.parse_args()
    print(args.f)

    ############## compute #####################
    # eval evaluates the string
    #res = xmr.xr_map_reduce(dirpath, args.m, eval(args.f), save_file, n_jobs=-1, intermediate_file='') #running without intermediate save