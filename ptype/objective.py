import yaml
import pandas as pd
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import class_weight
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from typing import List, Dict
import sys
import random
import os
from collections import OrderedDict
from echo.src.base_objective import BaseObjective

logger = logging.getLogger(__name__)

# function to set universal seed
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    
# metric functions
metric = 'val_balanced_ece'

def average_acc(y_true, y_pred):
    ra = 0
    ra_tot = 0
    sn = 0
    sn_tot = 0
    pl = 0
    pl_tot = 0
    fzra = 0
    fzra_tot = 0
    preds = np.argmax(y_pred, 1)
    labels = np.argmax(y_true, 1)
    for i in range(len(y_true)):
        if labels[i] == 0:
            if preds[i] == 0:
                ra += 1
            ra_tot += 1
        if labels[i] == 1:
            if preds[i] == 1:
                sn += 1
            sn_tot += 1
        if labels[i] == 2:
            if preds[i] == 2:
                pl += 1
            pl_tot += 1
        if labels[i] == 3:
            if preds[i] == 3:
                fzra += 1
            fzra_tot += 1
    try:
        ra_acc = ra/ra_tot
    except ZeroDivisionError:
        ra_acc = np.nan
    try:
        sn_acc = sn/sn_tot
    except ZeroDivisionError:
        sn_acc = np.nan
    try:
        pl_acc = pl/pl_tot
    except ZeroDivisionError:
        pl_acc = np.nan
    try:
        fzra_acc = fzra/fzra_tot
    except ZeroDivisionError:
        fzra_acc = np.nan
        
    acc = [ra_acc, sn_acc, pl_acc, fzra_acc]
    return np.nanmean(acc, dtype=np.float64)

def ece(y_true, y_pred):
    """
    Calculates the expected calibration error of the
    neural network.
    """
    confidences = np.max(y_pred, 1)
    pred_labels = np.argmax(y_pred, 1)
    true_labels = np.argmax(y_true, 1)
    num_bins = 10
    
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    
    return ece*100

def balanced_ece(y_true, y_pred):
    """
    Calculates the balanced expected calibration error of the
    neural network.
    """
    probs = np.max(y_pred, 1)
    preds = np.argmax(y_pred, 1)
    labels = np.argmax(y_true, 1)
    num_bins = 10
    
    test_data = pd.DataFrame.from_dict(
        {"pred_labels": preds,
         "true_labels": labels, 
         "pred_conf": probs})
    
    cond0 = (test_data["true_labels"] == 0)
    cond1 = (test_data["true_labels"] == 1)
    cond2 = (test_data["true_labels"] == 2)
    cond3 = (test_data["true_labels"] == 3)
    results = OrderedDict()
    results['ra_percent'] = {
        "true_labels": test_data[cond0]["true_labels"].values,
        "pred_labels": test_data[cond0]["pred_labels"].values,
        "confidences": test_data[cond0]["pred_conf"].values
    }
    results['sn_percent'] = {
        "true_labels": test_data[cond1]["true_labels"].values,
        "pred_labels": test_data[cond1]["pred_labels"].values,
        "confidences": test_data[cond1]["pred_conf"].values
    }
    results['pl_percent'] = {
        "true_labels": test_data[cond2]["true_labels"].values,
        "pred_labels": test_data[cond2]["pred_labels"].values,
        "confidences": test_data[cond2]["pred_conf"].values
    }
    results['fzra_percent'] = {
        "true_labels": test_data[cond3]["true_labels"].values,
        "pred_labels": test_data[cond3]["pred_labels"].values,
        "confidences": test_data[cond3]["pred_conf"].values
    }
    
    ece_list = []

    for i, (name, data) in enumerate(results.items()):
        pred_labels = data["pred_labels"]
        true_labels = data["true_labels"]
        confidences = data["confidences"]
        
        assert(len(confidences) == len(pred_labels))
        assert(len(confidences) == len(true_labels))
        assert(num_bins > 0)
    
        bin_size = 1.0 / num_bins
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float64)
        bin_confidences = np.zeros(num_bins, dtype=np.float64)
        bin_counts = np.zeros(num_bins, dtype=np.int32)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
                bin_confidences[b] = np.mean(confidences[selected])
                bin_counts[b] = len(selected)

        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        ece_list.append(ece)
    
    balanced_ece = np.mean(ece_list)
    
    return balanced_ece*100

# callback function
def get_callbacks(config: Dict[str, str]) -> List[Callback]:
    callbacks = []
    if "callbacks" in config:
        config = config["callbacks"]
    else:
        return []
    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")
    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")
    return callbacks

    
class Objective(BaseObjective):

    def __init__(self, config, metric=metric, device="cuda"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):

        K.clear_session()

        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]

        try:
            train_history = trainer(conf)
        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
                raise optuna.TrialPruned()
            elif "reraise" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}.")
                raise optuna.TrialPruned()
            else:
                logging.warning(
                    f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E
            
        results_dict = {self.metric:max(train_history[self.metric])}
        return results_dict

    
def trainer(conf, trial=False, verbose=True):    
    # load data
    df = pd.read_parquet(conf['data_path'])
    
    # model config
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    n_splits = conf['trainer']['n_splits']
    train_size1 = conf['trainer']['train_size1'] # sets test size
    train_size2 = conf['trainer']['train_size2'] # sets valid size
    seed = conf['trainer']['seed']
    weights = conf['trainer']['weights']
    num_hidden_layers = conf['trainer']['num_hidden_layers']
    hidden_size = conf['trainer']['hidden_size']
    dropout_rate = conf['trainer']['dropout_rate']
    batch_size = conf['trainer']['batch_size']
    ra_weight = conf['trainer']['ra_weight']
    sn_weight = conf['trainer']['sn_weight']
    pl_weight = conf['trainer']['pl_weight']
    fzra_weight = conf['trainer']['fzra_weight']
    class_weights = {0:ra_weight, 1:sn_weight, 2:pl_weight, 3:fzra_weight}
    learning_rate = conf['trainer']['learning_rate']
    activation = conf['trainer']['activation']
    run_eagerly = conf['trainer']['run_eagerly']
    shuffle = conf['trainer']['shuffle']
    epochs = conf['trainer']['epochs']
    label_smoothing = conf['trainer']['label_smoothing']
    
    # set seed
    seed_everything(seed)
    
    # split and preprocess the data
    df['day'] = df['datetime'].apply(lambda x: str(x).split(' ')[0])
    
    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size1, random_state=seed)
    train_idx, test_idx = list(splitter.split(df, groups=df['day']))[0]
    train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]
    
    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size2, random_state=seed)
    train_idx, valid_idx = list(splitter.split(train_data, groups=train_data['day']))[0]
    train_data, valid_data = train_data.iloc[train_idx], train_data.iloc[valid_idx]
    
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(train_data[features])
    x_valid = scaler_x.transform(valid_data[features])
    x_test = scaler_x.transform(test_data[features])
    y_train = train_data[outputs].to_numpy()
    y_valid = valid_data[outputs].to_numpy()
    y_test = test_data[outputs].to_numpy()
    
    def build_model(input_size, hidden_size, num_hidden_layers, output_size):
        model = tf.keras.models.Sequential()
        
        if activation == 'leaky':
            model.add(tf.keras.layers.Dense(input_size))
            model.add(tf.keras.layers.LeakyReLU())
        
            for i in range(num_hidden_layers):
                if num_hidden_layers == 1:
                    model.add(tf.keras.layers.Dense(hidden_size))
                    model.add(tf.keras.layers.LeakyReLU())
                else:
                    model.add(tf.keras.layers.Dense(hidden_size))
                    model.add(tf.keras.layers.LeakyReLU())
                    model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            model.add(tf.keras.layers.Dense(input_size, activation=activation))
        
            for i in range(num_hidden_layers):
                if num_hidden_layers == 1:
                    model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
                else:
                    model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
                    model.add(tf.keras.layers.Dropout(dropout_rate))
        
        model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
    
        return model
    
    model = build_model(len(features), hidden_size, num_hidden_layers, len(outputs))
    model.build((batch_size, len(features)))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(loss=loss, optimizer=optimizer, metrics=[balanced_ece], run_eagerly=run_eagerly)
    callbacks = get_callbacks(conf)
    
    # train model
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), class_weight=class_weights, callbacks=callbacks,
                     batch_size=batch_size, shuffle=shuffle, epochs=epochs)
    
    K.clear_session()
    del model
    
    return history.history
    
    
if __name__ == '__main__':
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    config = 'config.yml'
    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    train_history = trainer(conf)
    print(min(train_history[metric]))