import logging
from echo.src.base_objective import BaseObjective
from echo.src.trial_suggest import trial_suggest_loader
import yaml
import shutil
import os
import gc
import optuna
import pickle
import warnings
import numpy as np
from tensorflow.keras import backend as K
from argparse import ArgumentParser

from ptype.callbacks import get_callbacks, MetricsCallback
from ptype.models import DenseNeuralNetwork
from ptype.data import load_ptype_data_day, preprocess_data

from evml.keras.callbacks import ReportEpoch
from evml.keras.models import calc_prob_uncertainty
from bridgescaler import save_scaler


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
        if "rain_weight" in conf["optuna"]["parameters"]:
            conf = self.custom_updates(trial, conf)
        try:
            return {self.metric: trainer(conf, evaluate=False)}
        except Exception as E:
            if "Unexpected result" in str(E) or "CUDA" in str(E):
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


def trainer(conf, evaluate=True, data_seed=0):
    input_features = (
        conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
    )
    output_features = conf["ptypes"]
    metric = conf["metric"]
    # flag for using the evidential model
    if conf["model"]["loss"] == "dirichlet":
        use_uncertainty = True
    else:
        use_uncertainty = False
    data = load_ptype_data_day(conf, data_split=0, verbose=1)
    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf else conf["scale_groups"]
    groups = [conf[g] for g in scale_groups]
    leftovers = list(
        set(input_features)
        - set([row for group in scale_groups for row in conf[group]])
    )
    if len(leftovers):
        groups.append(leftovers)
    # scale the data
    scaled_data, scalers = preprocess_data(
        data,
        input_features,
        output_features,
        scaler_type="standard",
        encoder_type="onehot",
        groups=groups,
    )
    # Save the scalers when not using ECHO
    if evaluate:
        os.makedirs(os.path.join(conf["save_loc"], "scalers"), exist_ok=True)
        for scaler_name, scaler in scalers.items():
            fn = os.path.join(conf["save_loc"], "scalers", f"{scaler_name}.json")
            try:
                save_scaler(scaler, fn)
            except TypeError:
                with open(fn, "wb") as fid:
                    pickle.dump(scaler, fid)
    # set up callbacks
    callbacks = []
    if use_uncertainty:
        callbacks.append(ReportEpoch(conf["model"]["annealing_coeff"]))
    if "ModelCheckpoint" in conf["callbacks"]:  # speed up echo
        callbacks.append(
            MetricsCallback(
                scaled_data["train_x"],
                scaled_data["train_y"],
                name="train",
                use_uncertainty=use_uncertainty,
            )
        )
        callbacks.append(
            MetricsCallback(
                scaled_data["test_x"],
                scaled_data["test_y"],
                name="test",
                use_uncertainty=use_uncertainty,
            )
        )
    callbacks.append(
        MetricsCallback(
            scaled_data["val_x"],
            scaled_data["val_y"],
            name="val",
            use_uncertainty=use_uncertainty,
        )
    )
    callbacks += get_callbacks(conf)
    # initialize the model
    mlp = DenseNeuralNetwork(**conf["model"], callbacks=callbacks)
    # train the model
    history = mlp.fit(scaled_data["train_x"], scaled_data["train_y"])
    # Predict on the data splits
    if evaluate:
        # Save the best model when not using ECHO
        mlp.model.save(os.path.join(conf["save_loc"], "model"))
        for name in data.keys():
            x = scaled_data[f"{name}_x"]
            pred_probs = mlp.predict(x)
            if use_uncertainty:
                pred_probs, u, ale, epi = calc_prob_uncertainty(pred_probs)
                pred_probs = pred_probs.numpy()
                u = u.numpy()
                ale = ale.numpy()
                epi = epi.numpy()
            true_labels = np.argmax(data[name][output_features].to_numpy(), 1)
            pred_labels = np.argmax(pred_probs, 1)
            confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
            data[name]["true_label"] = true_labels
            data[name]["pred_label"] = pred_labels
            data[name]["pred_conf"] = confidences
            for k in range(pred_probs.shape[-1]):
                data[name][f"pred_conf{k+1}"] = pred_probs[:, k]
            if use_uncertainty:
                data[name]["evidential"] = u
                data[name]["aleatoric"] = np.take_along_axis(
                    ale, pred_labels[:, None], axis=1
                )
                data[name]["epistemic"] = np.take_along_axis(
                    epi, pred_labels[:, None], axis=1
                )
            data[name].to_parquet(os.path.join(conf["save_loc"], f"{name}.parquet"))
        return 1

    elif conf["direction"] == "max":  # Return metric to be used in ECHO
        return max(history.history[metric])
    else:
        return min(history.history[metric])


if __name__ == "__main__":

    description = "Usage: python train_mlp.py -c model.yml"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    trainer(conf)
