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
from echo.src.base_objective import BaseObjective
from callbacks import get_callbacks
from metrics import average_acc, ece, balanced_ece
from seed import seed_everything

logger = logging.getLogger(__name__)

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
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(loss=loss, optimizer=optimizer, metrics=[average_acc, ece, balanced_ece], run_eagerly=run_eagerly)
    callbacks = get_callbacks(conf)
    
    # train model
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), class_weight=class_weights, callbacks=callbacks,
                     batch_size=batch_size, shuffle=shuffle, epochs=epochs)
    
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
    
    config = 'mping_070622_config.yml'
    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    train_history = trainer(conf)