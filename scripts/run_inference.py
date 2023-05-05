import argparse
import os
import yaml
import pandas as pd
from ptype.inference import download_data, load_data, convert_and_interpolate, get_file_paths
from ptype.inference import load_model, transform_data, grid_preditions, save_data

import time
def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    # with open(args.config) as config_file:
    with open("/Users/cbecker/Desktop/Projects/ptype-physical/config/inference.yml") as config_file:
        config = yaml.safe_load(config_file)
    username = os.environ.get('USER')
    out_path = config["out_path"].replace("username", username)
    nwp_model = config["model"]
    forecast_hours = range(config["forecast_range"]["start"],
                           config["forecast_range"]["end"] + config["forecast_range"]["interval"],
                           config["forecast_range"]["interval"])
    model, transformer = load_model(model_path=config["ML_model_path"])

    dates = pd.date_range(start=config["dates"]["start"],
                          end=config["dates"]["end"],
                          freq=config["dates"]["frequency"])
    for date in dates:
        t = time.time()
        download_data(date=date,
                      model=config["model"],
                      product=config["variables"]["model"][nwp_model]["product"],
                      save_dir=out_path,
                      forecast_range=config["forecast_range"])
        print(f"Time to download: {time.time() - t}")
        files = get_file_paths(base_path=out_path,
                               model=nwp_model,
                               date=date)
        for file, fh in zip(files, forecast_hours):
            t = time.time()
            ds, df, surface_vars = load_data(var_dict=config["variables"]["model"][nwp_model],
                                             file=file,
                                             model=nwp_model,
                                             drop=config["drop_input_data"])
            print(f"Time to load: {time.time() - t}")
            t = time.time()
            data = convert_and_interpolate(data=df,
                                           surface_data=surface_vars,
                                           pressure_levels=ds["isobaricInhPa"],
                                           height_levels=config["height_levels"])
            print(f"Time to interpolate: {time.time() - t}")
            t = time.time()
            x_data = transform_data(input_data=data,
                                    transformer=transformer)
            print(f"Time to transform: {time.time() - t}")
            t = time.time()
            predictions = model.predict(x_data)
            print(f"Time to predict: {time.time() - t}")
            t = time.time()
            gridded_preds = grid_preditions(data=ds,
                                            preds=predictions)
            print(f"Time to grid: {time.time() - t}")
            t = time.time()
            save_data(dataset=gridded_preds,
                      out_path=out_path,
                      date=date,
                      model=config["model"],
                      forecast_hour=fh)
            print(f"Time to save: {time.time() - t}")
            print(f"Total time: {time.time() - start}")
if __name__ == "__main__":
    main()
