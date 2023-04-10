import argparse
import os
import yaml
import pandas as pd
import cfgrib
import xarray as xr
from herbie import Herbie, Herbie_latest, FastHerbie

def main():

    username = os.environ.get('USER')
    save_dir = f"/glade/scratch/{username}/herbie/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--out", default=f"/glade/scratch/{username}/rap_ncei_nc/", help="Output path")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["frequency"])
    H = FastHerbie(dates,
                   model=config["model"],
                   save_dir=save_dir,
                   fxx=range(config["first_forecast_hour"], config["last_forecast_hour"], config["fh_step"]))

    # download Grib files
    H.download()

    # retrieve
    files = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir)])
    datasets = []
    for f in files:
        gds = xr.merge(cfgrib.open_datasets(f, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}))
        datasets.append(gds)
    ds = xr.concat(datasets, dim='valid_time') # expensive, don't know why.

    ### Reshape, Interpolate


    ### Transform for ML, Load ML, Predict ML


    ### Populate dataset of predictions, save out, delete grib files


if __name__ == "__main__":
    main()