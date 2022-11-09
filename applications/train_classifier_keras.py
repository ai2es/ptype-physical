import logging
from echo.src.base_objective import BaseObjective
from echo.src.trial_suggest import trial_suggest_loader
import yaml
import shutil
import sys
import os
import gc
import optuna
import warnings
import numpy as np
from keras import backend as K

from ptype.callbacks import get_callbacks, MetricsCallback
from ptype.models import DenseNeuralNetwork
from ptype.data import load_ptype_data_day, preprocess_data

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        K.clear_session()
        gc.collect()
        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]
        conf = self.custom_updates(trial, conf)
        try:
            return {self.metric: trainer(conf, evaluate=False)}
        except Exception as E:
            if "Unexpected result" in str(E):
                logger.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}."
                )
                raise optuna.TrialPruned()
            else:
                logger.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E

    def custom_updates(self, trial, conf):
        # Get list of hyperparameters from the config
        hyperparameters = conf["optuna"]["parameters"]
        # Now update some via custom rules
        weights = []
        for ptype in ["rain_weight", "snow_weight", "sleet_weight", "frz_rain_weight"]:
            value = trial_suggest_loader(trial, hyperparameters[ptype])
            logger.info(f"Updated {ptype} with value {value}")
            weights.append(value)
        # Update the config based on optuna's suggestion
        conf["model"]["loss_weights"] = weights
        return conf


def trainer(conf, evaluate=True, data_seed=0, use_uncertainty=False):
    input_features = (
        conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
    )
    output_features = conf["ptypes"]
    metric = conf["metric"]
    data = load_ptype_data_day(conf, data_split=0, verbose=1)
    scaled_data, scalers = preprocess_data(
        data,
        input_features,
        output_features,
        scaler_type="standard",
        encoder_type="onehot",
    )
    callbacks = []
    if "ModelCheckpoint" in conf["callbacks"]:  # speed up echo
        callbacks.append(
            MetricsCallback(
                scaled_data["train_x"], scaled_data["train_y"], name="train"
            )
        )
    callbacks.append(
        MetricsCallback(scaled_data["val_x"], scaled_data["val_y"], name="val")
    )
    callbacks += get_callbacks(conf)
    mlp = DenseNeuralNetwork(**conf["model"], callbacks=callbacks)
    history = mlp.fit(scaled_data["train_x"], scaled_data["train_y"])

    if evaluate:
        for name in data.keys():
            x = scaled_data[f"{name}_x"]
            y_pred = mlp.predict(x)
            #             if use_uncertainty:
            #                 pred_probs, u = calc_prob_uncertainty(y_pred)
            #                 pred_probs = pred_probs.numpy()
            #                 u = u.numpy()
            #             else:
            #                 pred_probs = y_pred
            pred_probs = y_pred
            true_labels = np.argmax(data[name][output_features].to_numpy(), 1)
            pred_labels = np.argmax(pred_probs, 1)
            confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
            data[name]["true_label"] = true_labels
            data[name]["pred_label"] = pred_labels
            data[name]["pred_conf"] = confidences
            for k in range(pred_probs.shape[-1]):
                data[name][f"pred_conf{k+1}"] = pred_probs[:, k]
            # if use_uncertainty:
            #     data[name]["pred_sigma"] = u
            data[name].to_parquet(
                os.path.join(conf["save_loc"], f"{name}_{data_seed}.parquet")
            )
        return 1

    elif conf["direction"] == "max":  # Return metric to be used in ECHO
        return max(history.history[metric])
    else:
        return min(history.history[metric])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train_classifier_keras.py model.yml")
        sys.exit()

    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    trainer(conf)