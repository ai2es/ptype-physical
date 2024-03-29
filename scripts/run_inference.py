import argparse
import os
import yaml
import pandas as pd
from ptype.inference import (download_data, load_data, convert_and_interpolate,
    load_model, transform_data, grid_predictions, save_data)
import itertools
from multiprocessing import Pool
from dask.distributed import Client
from dask_jobqueue import PBSCluster


def main(config, username, date, forecast_hour):
    print("starting", date, "for forecast hour: ", forecast_hour)
    out_path = config["out_path"].replace("username", username)
    print(out_path)
    nwp_model = config["model"]
    model, transformer, input_features = load_model(model_path=config["ML_model_path"],
                                                    input_scaler_file=config["input_scaler_file"])
    mod_file = download_data(date=date,
                             model=config["model"],
                             product=config["variables"]["model"][nwp_model]["product"],
                             save_dir=out_path,
                             forecast_hour=forecast_hour)

    ds, df, surface_vars = load_data(var_dict=config["variables"]["model"][nwp_model],
                                     file=mod_file,
                                     model=nwp_model,
                                     extent=config["extent"],
                                     drop=config["drop_input_data"])
    data, interpolated_pl = convert_and_interpolate(data=df,
                                                    surface_data=surface_vars,
                                                    pressure_levels=ds["isobaricInhPa"],
                                                    height_levels=config["height_levels"])

    x_data = transform_data(input_data=data,
                            transformer=transformer,
                            input_features=input_features)

    if config["evidential"]:
        predictions = model.predict_uncertainty(x_data)
    else:
        predictions = model.predict(x_data, batch_size=2048)

    gridded_preds = grid_predictions(data=ds,
                                     predictions=predictions,
                                     interp_df=data,
                                     interpolated_pl=interpolated_pl,
                                     height_levels=config['height_levels'],
                                     add_interp_data=config["add_interp_data"],
                                     evidential=config["evidential"])
    save_data(dataset=gridded_preds,
              out_path=out_path,
              date=date,
              model=config["model"],
              forecast_hour=forecast_hour,
              save_format=config["save_format"])
    os.remove(str(mod_file))  # delete grib file
    del ds, df, surface_vars, data, x_data, interpolated_pl, mod_file, model, transformer, predictions, gridded_preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    username = os.environ.get('USER')

    if config["dates"]["most_recent"]:
        dates = ["most_recent"]
    else:
        dates = pd.date_range(start=config["dates"]["start"],
                              end=config["dates"]["end"],
                              freq=config["dates"]["frequency"])

    forecast_hours = range(config["forecast_range"]["start"],
                           config["forecast_range"]["end"] + config["forecast_range"]["interval"],
                           config["forecast_range"]["interval"])

    main_args = itertools.product([config], [username], dates, forecast_hours)
    if config["use_dask"]:

        cluster = PBSCluster(**config["dask_params"]["PBS"])
        client = Client(cluster)
        cluster.scale(jobs=config["dask_params"]["n_jobs"])
        print(f"Use this link to monitor the workload: {cluster.dashboard_link}")
        tasks = []
        for arguments in main_args:
            tasks.append(client.submit(main, *arguments))
        _ = [tasks[i].result() for i in range(len(tasks))]

    else:
        n_procs = int(config["n_processors"])
        if n_procs == 1:
            for main_arg in main_args:
                main(*main_arg)
        else:
            with Pool(n_procs) as pool:
                pool.starmap(main, main_args)
