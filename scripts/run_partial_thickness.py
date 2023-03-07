from ptype.baselines import PartialThicknessClassifier
from ptype.data import load_ptype_data_subset
import pandas as pd
from os.path import join, exists
from os import makedirs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument("-e", "--end", help="End Date")
    parser.add_argument("-d", "--data", help="Path to p-type data")
    parser.add_argument("-t", "--type", help="ASOS or mPING")
    parser.add_argument("-o", "--out", help="Output directory")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel processors")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbose level")
    args = parser.parse_args()

    input_data = load_ptype_data_subset(args.data, args.type, args.start, args.end,
                                        n_jobs=args.jobs, verbose=args.verbose)
    meta_cols = ["obdate", "obtime", "lon", "lat", "datetime", "precip_count_byhr"]
    ptc = PartialThicknessClassifier(n_jobs=args.jobs, verbose=args.verbose)
    pt_preds = pd.DataFrame(ptc.predict_proba(input_data), index=input_data.index, columns=ptc.p_type_labels)
    pt_pred_meta = pd.merge(input_data[meta_cols], pt_preds, left_index=True, right_index=True)
    if not exists(args.out):
        makedirs(args.out)
    pt_pred_meta.to_parquet(join(args.out, "partial_thickness_preds.parquet"))
    return


if __name__ == "__main__":
    main()
