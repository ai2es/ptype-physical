import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from ptype.callbacks import MetricsCallback
from ptype.data import load_ptype_uq, preprocess_data
from mlguess.keras.callbacks import get_callbacks
from mlguess.keras.models import CategoricalDNN
from bridgescaler import save_scaler


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def trainer(conf, evaluate=True, data_split=0, mc_forward_passes=0):
    input_features = []
    for features in conf["input_features"]:
        input_features += conf[features]
    output_features = conf["output_features"]
    metric = conf["metric"]
    # flag for using the evidential model
    if conf["model"]["loss"] == "evidential":
        use_uncertainty = True
    else:
        use_uncertainty = False
    # load data using the split (see n_splits in config)
    data = load_ptype_uq(conf, data_split=data_split, verbose=1, drop_mixed=False)
    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf else conf["scale_groups"]
    groups = [list(conf[g]) for g in scale_groups]
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
        scaler_type=conf["scaler_type"],
        encoder_type="onehot",
        groups=groups,
    )
    # Save the scalers when not using ECHO
    if evaluate:
        os.makedirs(os.path.join(conf["save_loc"], "scalers"), exist_ok=True)
        for scaler_name, scaler in scalers.items():
            if conf["ensemble"]["n_splits"] == 1:
                fn = os.path.join(conf["save_loc"], "scalers", f"{scaler_name}.json")
            else:
                fn = os.path.join(
                    conf["save_loc"], "scalers", f"{scaler_name}_{data_split}.json"
                )
            try:
                save_scaler(scaler, fn)
            except TypeError:
                with open(fn, "wb") as fid:
                    pickle.dump(scaler, fid)
    # set up callbacks
    callbacks = []
    # if use_uncertainty:
    #     callbacks.append(ReportEpoch(conf["model"]["annealing_coeff"]))
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
    callbacks += get_callbacks(conf, path_extend="models")
    # initialize the model
    mlp = CategoricalDNN(**conf["model"], callbacks=callbacks)
    # train the model
    print(scaled_data["train_y"])
    raise
    history = mlp.fit(scaled_data["train_x"], scaled_data["train_y"])

    if conf["ensemble"]["n_splits"] > 1:
        pd_history = pd.DataFrame.from_dict(history.history)
        pd_history["split"] = data_split
        pd_history.to_csv(
            os.path.join(conf["save_loc"], "models", f"training_log_{data_split}.csv")
        )

    # Predict on the data splits
    if evaluate:
        # Save the best model when not using ECHO
        if conf["ensemble"]["n_splits"] == 1:
            mlp.model.save(os.path.join(conf["save_loc"], "models", "best.h5"))
        else:
            mlp.model.save(
                os.path.join(conf["save_loc"], "models", f"model_{data_split}.h5")
            )
        for name in data.keys():
            x = scaled_data[f"{name}_x"]
            if use_uncertainty:
                pred_probs, u, ale, epi = mlp.predict_uncertainty(x)
                pred_probs = pred_probs.numpy()
                u = u.numpy()
                ale = ale.numpy()
                epi = epi.numpy()
            elif mc_forward_passes > 0:  # Compute epistemic uncertainty with MC dropout
                pred_probs = mlp.predict(x)
                _, ale, epi, entropy, mutual_info = mlp.predict_monte_carlo(
                    x, mc_forward_passes=mc_forward_passes
                )
            else:
                pred_probs = mlp.predict(x)
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
                for k in range(pred_probs.shape[-1]):
                    data[name][f"aleatoric{k+1}"] = ale[:, k]
                    data[name][f"epistemic{k+1}"] = epi[:, k]
            elif mc_forward_passes > 0:
                data[name]["aleatoric"] = np.take_along_axis(
                    ale, pred_labels[:, None], axis=1
                )
                data[name]["epistemic"] = np.take_along_axis(
                    epi, pred_labels[:, None], axis=1
                )
                for k in range(pred_probs.shape[-1]):
                    data[name][f"aleatoric{k+1}"] = ale[:, k]
                    data[name][f"epistemic{k+1}"] = epi[:, k]
                data[name]["entropy"] = entropy
                data[name]["mutual_info"] = mutual_info

            if conf["ensemble"]["n_splits"] == 1:
                data[name].to_parquet(
                    os.path.join(conf["save_loc"], f"evaluate/{name}.parquet")
                )
            else:
                data[name].to_parquet(
                    os.path.join(
                        conf["save_loc"], f"evaluate/{name}_{data_split}.parquet"
                    )
                )
        return 1

    elif conf["direction"] == "max":  # Return metric to be used in ECHO
        return max(history.history[metric])
    else:
        return min(history.history[metric])
