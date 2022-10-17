import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
                                  OneHotEncoder, LabelEncoder


def load_ptype_data(data_path, source, train_start='20130101', train_end='20181108',
                    val_start='20181109', val_end='20200909',
                    test_start='20200910', test_end='20210501'):
    """
    Load Precip Type data
    Args:
        data_path (str): Path to data
        source (str): Precip observation source. Supports 'ASOS' or 'mPING'.
        train_start (str): Train split start date (format yyyymmdd).
        train_end (str): Train split end date (format yyyymmdd).
        val_start (str): Valid split start date (format yyyymmdd).
        val_end (str): Valid split end date (format yyyymmdd).
        test_start (str): Test split start date (format yyyymmdd).
        test_end (str): Test split end date (format yyyymmdd).        
    Returns:
    Dictionary of Pandas dataframes of training / validation / test data
    """
    
    dates = sorted([x[-16:-8] for x in os.listdir(data_path)])
    
    data = {}
    data['train'] = dates[dates.index(train_start) : dates.index(train_end) + 1]
    data['val'] = dates[dates.index(val_start) : dates.index(val_end) + 1]
    data['test'] = dates[dates.index(test_start) : dates.index(test_end) + 1]
    
    for split in data.keys():
        dfs = []
        for date in tqdm(data[split], desc=f"{split}"):
            f = f"{source}_rap_{date}.parquet"
            dfs.append(pd.read_parquet(os.path.join(data_path, f)))
        data[split] = pd.concat(dfs, ignore_index=True)            

    return data


def load_ptype_data_subset(data_path, source, start_date, end_date, n_jobs=1, verbose=1):
    """
    Load a single range of dates from the mPING or ASOS parquet files into memory. Supports parallel loading with joblib.

    Args:
        data_path: Path to appropriate p-type directory containing parquet files.
        source: "mPING" or "ASOS"
        start_date: Pandas-supported Date string for first day in time range (inclusive)
        end_date: Pandas supported Date string for last day in time range (inclusive)
        n_jobs: Number of parallel processes to use for data loading (default 1)
        verbose: verbose level
    Returns:
        data: Pandas DataFrame containing all sounding and p-type data from start_date to end_date.

    """
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    data_files = sorted(os.listdir(data_path))
    all_dates = pd.DatetimeIndex([x[-16:-8] for x in data_files])
    selected_dates = all_dates[(all_dates >= start_timestamp) & (all_dates <= end_timestamp)]
    dfs = []
    if n_jobs == 1:
        for date in tqdm(selected_dates):
            date_str = date.strftime("%Y%m%d")
            filename = f"{source}_rap_{date_str}.parquet"
            dfs.append(pd.read_parquet(os.path.join(data_path, filename)))
    else:
        date_strs = selected_dates.strftime("%Y%m%d")
        dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            [delayed(pd.read_parquet)(os.path.join(data_path, f"{source}_rap_{date_str}.parquet"))
             for date_str in date_strs])
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess_data(data, input_features, output_features, scaler_type="standard", encoder_type="onehot"):
    """
    Function to select features and scale data for ML
    Args:
        data (dictionary of dataframes for training and validation data):
        input_features (list): Input features
        output_feature (list): Output feature
        scaler_type: Type of scaling to perform (supports "standard" and "minmax")
        encoder_type: Type of encoder to perform (supports "label" and "onehot")

    Returns:
        Dictionary of scaled and one-hot encoded data, dictionary of scaler objects
    """
    scalar_obs = {"minmax": MinMaxScaler, "standard": StandardScaler}
    scalers, scaled_data = {}, {}

    scalers["input"] = scalar_obs[scaler_type]()
    scaled_data["train_x"] = pd.DataFrame(scalers["input"].fit_transform(data["train"][input_features]),
                                          columns=input_features)
    scaled_data["val_x"] = pd.DataFrame(scalers["input"].transform(data["val"][input_features]), columns=input_features)
    scaled_data["test_x"] = pd.DataFrame(scalers["input"].transform(data["test"][input_features]), columns=input_features)

    scalers["output_label"] = LabelEncoder()
    scaled_data["train_y"] = scalers["output_label"].fit_transform(data['train']['precip'])
    scaled_data["val_y"] = scalers["output_label"].transform(data['val']['precip'])
    scaled_data["test_y"] = scalers["output_label"].transform(data['test']['precip'])

    if encoder_type == "onehot":
        scalers["output_onehot"] = OneHotEncoder(sparse=False)
        scaled_data["train_y"] = scalers["output_onehot"].fit_transform(scaled_data["train_y"].reshape(len(scaled_data["train_y"]), 1))
        scaled_data["val_y"] = scalers["output_onehot"].transform(scaled_data["val_y"].reshape(len(scaled_data["val_y"]), 1))
        scaled_data["test_y"] = scalers["output_onehot"].transform(scaled_data["test_y"].reshape(len(scaled_data["test_y"]), 1))

    return scaled_data, scalers
  
  
def reshape_data_1dCNN(data, base_variables=['TEMP_C', 'T_DEWPOINT_C', 'UGRD_m/s', 'VGRD_m/s'], n_levels=67):
    arr = np.zeros(shape=(data.shape[0], n_levels, len(base_variables))).astype('float32')
    for i, var in enumerate(base_variables):
        profile_vars = [x for x in list(data.columns) if var in x]
        arr[:, :, i] = data[profile_vars].values.astype('float32')
    return arr
