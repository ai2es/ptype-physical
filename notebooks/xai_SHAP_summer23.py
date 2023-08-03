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

fn_config = "/glade/p/cisl/aiml/ai2es/winter_ptypes/models/classifier_weighted/model.yml"

with open(fn_config) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)
    
    input_features = (
    conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
)
output_features = conf["ptypes"]

#loading the pretrained model 
model = CategoricalDNN.load_model(conf)
input_scaler_loc = os.path.join(conf["save_loc"], "scalers", "input.json")

input_scaler = load_scaler(input_scaler_loc)

# Calculating SHAP values 
train_path='/glade/scratch/schreck/repos/evidential/results/ptype/weighted/classifier/evaluate/train_1.parquet'
train_data = pd.read_parquet(train_path)
train_data = train_data[input_features]

rand_data = train_data.sample(frac=0.50).values

explainer = shap.Explainer(model.model, rand_data)
shap_values = explainer.shap_values(rand_data)

np.save("/glade/work/saavedrab/ptype-physical/notebooks/xai_vals/shap_50.npy", shap_values)
