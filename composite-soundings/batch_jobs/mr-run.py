from os.path import join
import argparse

import xr_map_reduce as xmr

CASE_DICT = {0: "kentucky", 1: "new_york_1", 2: "new_york_2"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process a case study.")
    parser.add_argument("-i", help="case study index")
    parser.add_argument("-m", help="model name")
    parser.add_argument("-o", help="outfile base name (base_name_[case study])")
    parser.add_argument("-d", help="toplevel dir of all the files to run compute")
    args = parser.parse_args()

    if not args.o:
        raise ValueError("need to pass in outfile name")
    if not args.m:
        raise ValueError("need to pass in model name (rap, hrrr, gfs)")
    ######### setting save file names and data file dirs #############
    if args.d:
        dirpath = args.d
        save_file = f"/glade/work/dkimpara/ptype-aggs/{args.o}.nc"
        intermediate_save_file = (
            f"/glade/work/dkimpara/ptype-aggs/{args.o}_pre_merge.dump"
        )
    elif args.i:
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

    print(f"opening {dirpath}\n")
    print(f"saving to {save_file}\n")

    ############## compute #####################
    res = xmr.xr_map_reduce(
        dirpath, args.m, xmr.compute_func, intermediate_save_file, -1
    )
    res.to_netcdf(save_file)

    print(f"write to {save_file} successful")
