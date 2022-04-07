import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
                                  OneHotEncoder, LabelEncoder

def load_ptype_data(data_path, source, train_val_test_proportion=[70,20,10]):
    """
    Load Precip Type data
    Args:
        data_path (str): Path to data
        source (str): Precip observation source. Supports 'ASOS' or 'mPING'.
        train_val_test_proportion (list of int): Proportion of data to use in each train, validation, and test split.
    Returns:
    Dictionary of Pandas dataframes of training / validation / test data
    """
    
    dates = sorted([x[-16:-8] for x in os.listdir(os.path.join(data_path, source))])
    
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

    scalers["output"] = LabelEncoder()
    scaled_data["train_y"] = scalers["output"].fit_transform(data['train']['precip'])
    scaled_data["val_y"] = scalers["output"].transform(data['val']['precip'])
    scaled_data["test_y"] = scalers["output"].transform(data['test']['precip'])

    if encoder_type == "onehot":
        scalers["output"] = OneHotEncoder(sparse=False)
        scaled_data["train_y"] = encoder.fit_transform(scaled_data["train_y"].reshape(len(scaled_data["train_y"]), 1))
        scaled_data["val_y"] = encoder.transform(scaled_data["val_y"].reshape(len(scaled_data["val_y"]), 1))
        scaled_data["test_y"] = encoder.transform(scaled_data["test_y"].reshape(len(scaled_data["test_y"]), 1))

    return scaled_data, scalers