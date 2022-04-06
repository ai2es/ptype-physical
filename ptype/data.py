import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_ptype_data(data_path, source, train_val_test_proportion=[70,20,10], seed=1):
    """
    Load Precip Type data
    Args:
        data_path (str): Path to data
        source (str): Precip observation source. Supports 'ASOS' or 'mPING'.
        train_val_test_proportion (list of int): proportion of data to use in each train, validation, and test split.
        seed (int): random seed used in random shuffle.
    Returns:
    Dictionary of Pandas dataframes of training / validation / test data
    """
    
    dates = [x[-16:-8] for x in os.listdir(os.path.join(data_path, source))]
    random.Random(seed).shuffle(dates)
    
    data = {}
    train_split = int(train_val_test_proportion[0]/100 * len(dates))
    val_split = int((train_val_test_proportion[0] + train_val_test_proportion[1])/100 * len(dates))
    data['train'] = dates[:train_split]
    data['val'] = dates[train_split:val_split]
    data['test'] = dates[val_split:]
    
    for split in data.keys():
        dfs = []
        for date in tqdm(data[split], desc=f"{split}"):
            f = f"{source}_rap_{date}.parquet"
            dfs.append(pd.read_parquet(os.path.join(data_path, source, f)))
        data[split] = pd.concat(dfs, ignore_index=True)            

    return data

def preprocess_data(data, input_features, output_features, scaler_type="standard"):
    """
    Function to select features and scale data for ML
    Args:
        data (dictionary of dataframes for training and validation data):
        input_features (list): Input features
        output_features (list): Output features
        scaler_type: Type of scaling to perform (supports "standard" and "minmax")

    Returns:
        Dictionary of scaled data, dictionary of scaler objects
    """
    scalar_obs = {"minmax": MinMaxScaler, "standard": StandardScaler}
    scalers, scaled_data = {}, {}

    scalers["input"], scalers["output"] = scalar_obs[scaler_type](), scalar_obs[scaler_type]()
    scaled_data["train_x"] = pd.DataFrame(scalers["input"].fit_transform(data["train"][input_features]),
                                          columns=input_features)
    scaled_data["val_x"] = pd.DataFrame(scalers["input"].transform(data["val"][input_features]), columns=input_features)
    scaled_data["test_x"] = pd.DataFrame(scalers["input"].transform(data["test"][input_features]), columns=input_features)
    scaled_data["train_y"] = pd.DataFrame(
        scalers["output"].fit_transform(data["train"][output_features].values.reshape(-1, len([output_features]))),
        columns=[output_features])
    scaled_data["val_y"] = pd.DataFrame(
        scalers["output"].transform(data["val"][output_features].values.reshape(-1, len([output_features]))),
        columns=[output_features])
    scaled_data["test_y"] = pd.DataFrame(
        scalers["output"].transform(data["test"][output_features].values.reshape(-1, len([output_features]))),
        columns=[output_features])
    
    return scaled_data, scalers