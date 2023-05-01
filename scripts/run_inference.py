import argparse
import os
import yaml
import pandas as pd
from ptype.inference import download_data, load_data, interpolate_data, get_file_paths
from ptype.inference import load_model, transform_data, grid_preditions, save_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
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
        download_data(date=date,
                      model=config["model"],
                      product=config["variables"]["model"][nwp_model]["product"],
                      save_dir=out_path,
                      forecast_range=config["forecast_range"])

        files = get_file_paths(base_path=out_path,
                               model=nwp_model,
                               date=date)
        for file, fh in zip(files, forecast_hours):
            ds, df, surface_vars = load_data(var_dict=config["variables"]["model"][nwp_model],
                                             file=file,
                                             model=nwp_model,
                                             drop=config["drop_input_data"])
            data = interpolate_data(data=df,
                                    surface_data=surface_vars,
                                    pressure_levels=ds["isobaricInhPa"],
                                    height_levels=config["height_levels"])

            x_data = transform_data(input_data=data,
                                    transformer=transformer)
            predictions = model.predict(x_data)
            gridded_preds = grid_preditions(data=ds,
                                            preds=predictions)
            save_data(dataset=gridded_preds,
                      out_path=out_path,
                      date=date,
                      model=config["model"],
                      forecast_hour=fh)


if __name__ == "__main__":
    main()
