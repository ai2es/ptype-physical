import os
import yaml
import pandas as pd
import numpy as np

from evml.keras.models import CategoricalDNN
from bridgescaler import load_scaler

from evml.classifier_uq import brier_multi
import tensorflow as tf
from collections import defaultdict

import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.inspection import permutation_importance
from captum.attr import FeaturePermutation
from sklearn.metrics import accuracy_score, mean_squared_error
from multiprocessing import Pool
from copy import deepcopy
import traceback
fn_config = "/glade/p/cisl/aiml/ai2es/winter_ptypes/models/classifier_weighted/model.yml"

with open(fn_config) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)
    

# DJ's function: 

def feature_importance(x, y, model, metric_function, x_columns=None, permutations=5, processes=1,
                       col_start="perm_", seed=8272):
    """
    Calculate permutation feature importance scores for an arbitrary machine learning model.

    Args:
        x: ndarray of dimension (n_examples, n_features) that contains the input data for the ML model.
        y: ndarray of dimension (n_examples, ) that contains the true target values.
        model: machine learning model object in scikit-learn format (contains fit and predict methods).
        metric_function: scoring function with the input format (y_true, y_predicted) to match scikit-learn.
        x_columns (ndarray or None): list or array of column names. If not provided, indices will be used instead.
        permutations (int): Number of times a column is randomly shuffled.
        processes (int): Number of multiprocessor processes used for parallel computation of importances
        col_start (str): Start of output columns.
        seed (int): Random seed.

    Returns:
        pandas DataFrame of dimension (n_columns, permutations) that contains the change in score
        for each column and permutation.
    """
    if x_columns is None:
        x_columns = np.arange(x.shape[1])
    if type(x_columns) == list:
        x_columns = np.array(x_columns)
    predictions = model.predict(x)
    score = metric_function(y, predictions)
    print(score)
    np.random.seed(seed=seed)
    perm_matrix = np.zeros((x_columns.shape[0], permutations))

    def update_perm_matrix(result):
        perm_matrix[result[0]] = result[1]
    if processes > 1:
        pool = Pool(processes)
        for c in range(len(x_columns)):
            pool.apply_async(feature_importance_column,
                             (x, y, c, permutations, deepcopy(model), metric_function, np.random.randint(0, 100000)),
                              callback=update_perm_matrix)
        pool.close()
        pool.join()
    else:
        for c in range(len(x_columns)):
            result = feature_importance_column(x, y, c, permutations, model,
                                               metric_function, np.random.randint(0, 100000))
            update_perm_matrix(result)
    diff_matrix = score - perm_matrix
    out_columns = col_start + pd.Series(np.arange(permutations)).astype(str)
    return pd.DataFrame(diff_matrix, index=x_columns, columns=out_columns)

def feature_importance_column(x, y, column_index, permutations, model, metric_function, seed):
    """
    Calculate the permutation feature importance score for a single input column. It is the error score on
    a given set of data after the values in one column have been shuffled among the different examples.

    Args:
        x: ndarray of dimension (n_examples, n_features) that contains the input data for the ML model.
        y: ndarray of dimension (n_examples, ) that contains the true target values.
        column_index: Index of the x column being permuted
        permutations: Number of permutations run to calculate importance score distribution
        model: machine learning model object in scikit-learn format (contains fit and predict methods).
        metric_function: scoring function with the input format (y_true, y_predicted) to match scikit-learn.
        seed (int): random seed.

    Returns:
        column_index, permutation, perm_score
    """
    try:
        rs = np.random.RandomState(seed=seed)
        perm_indices = np.arange(x.shape[0])
        perm_scores = np.zeros(permutations)
        x_perm = np.copy(x)
        for p in range(permutations):
            print(column_index, p)
            rs.shuffle(perm_indices)
            x_perm[np.arange(x.shape[0]), column_index] = x[perm_indices, column_index]
            perm_pred = model.predict(x_perm)
            perm_scores[p] = metric_function(y, perm_pred)
        return column_index, perm_scores
    except Exception as e:
        print(traceback.format_exc())
        raise e
        
def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total    


input_features = (conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"])
model = CategoricalDNN.load_model(conf)
        
train_path = '/glade/scratch/schreck/repos/evidential/results/ptype/weighted/classifier/evaluate/train_1.parquet'
train_data = pd.read_parquet(train_path)

y_train = train_data['true_label'] #target
X_data = train_data[input_features]   

target = pd.get_dummies(y_train)
target = target.values.astype(np.float32)

permu_val = feature_importance(X_data.values, target, model, accuracy_score, permutations=10)

np.save("/glade/work/saavedrab/ptype-physical/notebooks/xai_vals/perm_10.npy", permu_val)
