from os.path import join
import argparse
import sys

sys.path.append("../")

from ptype.soundings import xr_map_reduce as xmr
from functools import partial  # partial may be used in the eval statement

CASE_DICT = {0: "kentucky", 1: "new_york_1", 2: "new_york_2"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process a case study.")
    parser.add_argument("-i", help="case study index")
    parser.add_argument("-m", help="model name")
    parser.add_argument("-o", help="outfile base name (base_name_[case study])")
    parser.add_argument("-d", help="toplevel dir of all the files to run compute")
    parser.add_argument(
        "-f",
        help="function to compute, will be evaluated by python. ds needs to be the last argument and unfilled",
    )
    args = parser.parse_args()

    if not args.o:
        raise ValueError("need to pass in outfile name")
    if not args.m:
        raise ValueError("need to pass in model name (rap, hrrr, gfs)")
    if not args.f:
        raise ValueError("need to pass in a function to compute")
    ######### setting save file names and data file dirs #############
    if args.d:  # if specifying a directory
        dirpath = args.d
        save_file = f"/glade/work/dkimpara/ptype-aggs/{args.o}.nc"
        intermediate_save_file = (
            "" f"/glade/work/dkimpara/ptype-aggs/{args.o}_pre_merge.dump"
        )
    elif args.i:  # if specifying a case study
        case_study = CASE_DICT[int(args.i)]
        dirpath = join(
            "/glade/campaign/cisl/aiml/ptype/ptype_case_studies/", case_study
        )
        save_file = f"/glade/work/dkimpara/ptype-aggs/{args.o}_{case_study}.nc"
        intermediate_save_file = (
            f"/glade/work/dkimpara/ptype-aggs/{args.o}_{case_study}_pre_merge.dump"
        )
    else:
        raise ValueError("need to specify either -i or -d")

    ############## compute #####################
    # eval evaluates the string defining the (partial) function
    res = xmr.xr_map_reduce(
        dirpath, args.m, eval(args.f), save_file, n_jobs=-1, intermediate_file=""
    )  # running without intermediate save
