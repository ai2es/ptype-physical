import os
import gc
import sys
import tqdm
import yaml
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd

from ptype.callbacks import get_callbacks, MetricsCallback
from ptype.models import DenseNeuralNetwork
from pathlib import Path
from ptype.data import load_ptype_data_day, preprocess_data

from evml.keras.callbacks import ReportEpoch
from evml.keras.models import calc_prob_uncertainty

from collections import defaultdict
from argparse import ArgumentParser
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf

import warnings

warnings.filterwarnings("always")
pd.options.mode.chained_assignment = None


def train(conf, data, mc_forward_passes=0):
    if conf["model"]["loss"] == "dirichlet":
        evidential_model = True
    else:
        evidential_model = False
    callbacks = []
    if evidential_model:
        callbacks.append(ReportEpoch(conf["model"]["annealing_coeff"]))
    callbacks.append(
        MetricsCallback(
            data["val_x"],
            data["val_y"],
            name="val",
            use_uncertainty=evidential_model,
        )
    )
    callbacks += get_callbacks(conf)
    mlp = DenseNeuralNetwork(**conf["model"], callbacks=callbacks)
    history = mlp.fit(
        data["train_x"], data["train_y"], validation_data=(data["val_x"], data["val_y"])
    )
    # Evaluate
    results_dict = defaultdict(dict)
    relevant_keys = set([k.strip("_y") for k in data.keys() if "_y" in k])
    for name in relevant_keys:
        x = data[f"{name}_x"]
        if evidential_model:  # Compute uncertainties
            pred_probs = mlp.predict(x)
            pred_probs, u, ale, epi = calc_prob_uncertainty(pred_probs)
            pred_probs = pred_probs.numpy()
            u = u.numpy()
            ale = ale.numpy()
            epi = epi.numpy()
        elif mc_forward_passes > 0:  # Compute epistemic uncertainty with MC dropout
            pred_probs = mlp.predict(x)
            _, epi, entropy, mutual_info = mlp.predict_dropout(
                x, mc_forward_passes=mc_forward_passes
            )
        else:  # Predict probabilities only when the random policy is being used
            pred_probs = mlp.predict(x)
        true_labels = np.argmax(data[f"{name}_y"], 1)
        pred_labels = np.argmax(pred_probs, 1)
        confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
        results_dict[name]["true_label"] = list(true_labels)
        results_dict[name]["pred_label"] = list(pred_labels)
        results_dict[name]["pred_conf"] = list(confidences[:, 0])
        for k in range(pred_probs.shape[-1]):
            results_dict[name][f"pred_conf{k+1}"] = list(pred_probs[:, k])
        if evidential_model:
            results_dict[name]["evidential"] = list(u[:, 0])
            results_dict[name]["aleatoric"] = list(
                np.take_along_axis(ale, pred_labels[:, None], axis=1)[:, 0]
            )
            results_dict[name]["epistemic"] = list(
                np.take_along_axis(epi, pred_labels[:, None], axis=1)[:, 0]
            )
        if mc_forward_passes > 0:
            results_dict[name]["mc-dropout"] = list(
                np.take_along_axis(epi, pred_labels[:, None], axis=1)[:, 0]
            )
            results_dict[name]["entropy"] = list(entropy)
            results_dict[name]["mutual-info"] = list(mutual_info)

    results_dict = {d: pd.DataFrame.from_dict(v) for d, v in results_dict.items()}
    training_log = pd.DataFrame.from_dict(history.history)

    del mlp
    tf.keras.backend.clear_session()
    gc.collect()

    return training_log, results_dict


def launch_pbs_jobs(
    config,
    nodes,
    gpu=1,
    save_path="./",
    policy="mc-dropout",
    iterations=20,
    mc_steps=100,
    cpus=8,
    mem=128,
    walltime="12:00:00",
):

    script_path = Path(__file__).absolute()
    if gpu > 0:
        args = f"""#PBS -l select=1:ncpus={cpus}:ngpus={gpu}:mem={mem}GB
        #PBS -l gpu_type=v100"""
    else:
        args = f"#PBS -l select=1:ncpus={cpus}:mem={mem}GB"

    for worker in range(nodes):
        script = f"""
        #!/bin/bash -l
        #PBS -N {policy}-{worker}
        {args}
        #PBS -l walltime={walltime}
        #PBS -A NAML0001
        #PBS -q casper
        #PBS -o {os.path.join(save_path, "out")}
        #PBS -e {os.path.join(save_path, "out")}

        source ~/.bashrc
        conda activate ptype
        python {script_path} -c {config} -p {policy} -i {iterations} -s {mc_steps} -n {nodes} -w {worker}
        """
        with open("launcher.sh", "w") as fid:
            fid.write(script)
        jobid = subprocess.Popen(
            "qsub launcher.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        print(jobid)

    os.remove("launcher.sh")


if __name__ == "__main__":

    description = "Run active training on p-type data sets. "
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-p",
        dest="policy",
        type=str,
        default="evidential",
        help="The active training policy to be used",
    )
    parser.add_argument(
        "-s",
        dest="steps",
        type=str,
        default=0,
        help="Number of MC iterations to use with policy = mc-dropout",
    )
    parser.add_argument(
        "-i",
        dest="iterations",
        type=int,
        default=20,
        help="The number of active training iterations to perform",
    )
    parser.add_argument(
        "-w",
        dest="worker",
        type=int,
        default=1,
        help="Worker number (to be used with parallelization)",
    )
    parser.add_argument(
        "-n",
        dest="nodes",
        type=int,
        default=1,
        help="The total number of nodes (GPUs, 1 model per GPU).",
    )
    parser.add_argument(
        "-g",
        dest="gpu",
        type=int,
        default=1,
        help="Use a GPU to train the model (boolean). Default = 1",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {nodes} workers to PBS. Run -m option once all workers finish.",
    )
    parser.add_argument(
        "-cpu",
        dest="cpu",
        type=int,
        default=8,
        help="Number of CPU cores to request from PBS. Default = 8",
    )
    parser.add_argument(
        "-mem",
        dest="mem",
        type=int,
        default=128,
        help="Number of GBS of RAM to request from PBS. Default = 128",
    )
    parser.add_argument(
        "-t",
        dest="walltime",
        type=str,
        default="12:00:00",
        help="Simulation wall time to request from PBS. Default = 12:00:00",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    policy = str(args_dict.pop("policy"))
    num_iterations = int(args_dict.pop("iterations"))
    worker = int(args_dict.pop("worker"))
    nodes = int(args_dict.pop("nodes"))
    gpu = int(args_dict.pop("gpu"))
    steps = int(args_dict.pop("steps"))
    launch = bool(int(args_dict.pop("launch")))
    cpus = int(args_dict.pop("cpu"))
    mem = int(args_dict.pop("mem"))
    walltime = str(args_dict.pop("walltime"))

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    seed = conf["seed"]
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(os.path.join(save_loc, "training_logs"), exist_ok = True)
    os.makedirs(os.path.join(save_loc, "active_logs"), exist_ok = True)
    os.makedirs(os.path.join(save_loc, "data_logs"), exist_ok = True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    if launch:
        logging.info(f"Launching {nodes} workers to PBS")
        launch_pbs_jobs(
            config_file,
            nodes,
            gpu,
            conf["save_loc"],
            policy,
            num_iterations,
            steps,
            cpus,
            mem,
            walltime,
        )
        sys.exit()

    # seed_everything(seed)
    # Load the data
    input_features = (
        conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
    )
    output_features = conf["ptypes"]
    metric = conf["metric"]
    data = load_ptype_data_day(conf, data_split=0, verbose=1)
    data["train"] = pd.concat([data["train"], data["val"]])
    del data["val"]
    
    # Set up a dataframe to save the split details after each active iteration
    data_log = defaultdict(list)
    data_log["id"] = [_ for v in data.values() for _ in v["id"]] 
    data_log["ptype"] = [_ for v in data.values() for _ in np.argmax(v[output_features].values, 1)]
    data_log = pd.DataFrame.from_dict(data_log)

    # check if we should scale the input data by groups
    scale_groups = [] if "scale_groups" not in conf else conf["scale_groups"]
    groups = [conf[g] for g in scale_groups]
    ungrouped = list(
        set(input_features)
        - set([row for group in scale_groups for row in conf[group]])
    )
    if len(ungrouped):
        groups.append(ungrouped)

    # set up data splits / iteration loops
    num_selected = int((1.0 / num_iterations) * data["train"].shape[0])
    data_splits = list(range(conf["n_splits"]))
    if nodes > 1:
        data_splits = np.array_split(data_splits, nodes)[worker]

    for sidx in data_splits:

        my_iter = tqdm.tqdm(range(num_iterations), total=num_iterations, leave=True)

        active_results = defaultdict(list)
        for iteration in my_iter:
            if iteration == 0:
                # Select random fraction on first pass
                train_data_ = data["train"].sample(n=num_selected, random_state=seed)
                left_overs = np.array(
                    list(set(data["train"]["id"]) - set(train_data_["id"]))
                )
                left_overs = data["train"][data["train"]["id"].isin(left_overs)].copy()
            else:
                # Select with a policy
                if policy == "random":
                    # selection = left_overs.sample(n = num_selected, random_state = seed)
                    selection = min(
                        data["train"].shape[0], (iteration + 1) * num_selected
                    )
                    train_data_ = data["train"].sample(n=selection, random_state=seed)
                else:
                    left_overs = left_overs.sort_values(policy, ascending=False)
                    if num_selected > left_overs.shape[0]:
                        selection = left_overs.copy()
                    else:
                        selection = left_overs.iloc[:num_selected].copy()
                    # Add to the training data and determine whats left over for the next iteration
                    train_data_ = pd.concat([train_data_, selection])

                left_overs = np.array(
                    list(set(data["train"]["id"]) - set(train_data_["id"]))
                )
                left_overs = data["train"][data["train"]["id"].isin(left_overs)].copy()
                if left_overs.shape[0] == 0:
                    break
                    
            # Save the ID and split to the data complexition dataframe 
            # test ~ 0, train ~ 1, left overs ~ 2
            data_log[f"iteration {iteration}"] = [-1 for _ in range(data_log.shape[0])] 
            data_log[f"iteration {iteration}"].iloc[left_overs["id"].values] = 2
            data_log[f"iteration {iteration}"].iloc[train_data_["id"].values] = 1
            data_log[f"iteration {iteration}"].iloc[data["test"]["id"].values] = 0
            data_log.to_csv(os.path.join(save_loc, "data_logs", f"data_complexion_{sidx}.csv"))
                    
            # Split the available training data into train/validation split
            splitter = GroupShuffleSplit(
                n_splits=conf["n_splits"],
                train_size=conf["train_size2"],
                random_state=seed,
            )
            this_train_idx, this_valid_idx = list(
                splitter.split(train_data_, groups=train_data_["day"])
            )[sidx]
            this_train_data, this_valid_data = (
                train_data_.iloc[this_train_idx],
                train_data_.iloc[this_valid_idx],
            )
            # Load data splits into a dictionary
            data_dict = {
                "train": this_train_data,
                "val": this_valid_data,
                "test": data["test"],
                "left_overs": left_overs,
            }

            # Preprocess the data
            scaled_data, scalers = preprocess_data(
                data_dict,
                input_features,
                output_features,
                scaler_type="standard",
                encoder_type="onehot",
                groups=groups,
            )
            
            # Train model and predict on holdouts
            if policy in ["mc-dropout", "entropy", "mutual-info"]:
                training_log, pred_df = train(
                    conf, scaled_data, mc_forward_passes=steps
                )
            else:
                training_log, pred_df = train(conf, scaled_data)

            # Save data
            if policy in pred_df["left_overs"]:
                left_overs[policy] = pred_df["left_overs"][policy]
            training_log.to_csv(
                os.path.join(save_loc, "training_logs", f"train_log_{iteration}_{sidx}.csv"), index=False
            )
            print_str = f"Iteration {iteration}"
            active_results["iteration"].append(iteration)
            active_results["ensemble"].append(sidx)

            # Compute some metrics
            for name in pred_df.keys():
                true_labels = pred_df[name]["true_label"]
                pred_labels = pred_df[name]["pred_label"]
                pred_probs = pred_df[name]["pred_conf"]
                prec, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average="macro"
                )
                active_results[f"{name}_ave_acc"].append(
                    balanced_accuracy_score(true_labels, pred_labels)
                )
                active_results[f"{name}_prec"].append(prec)
                active_results[f"{name}_recall"].append(recall)
                active_results[f"{name}_f1"].append(f1)
                pred_probs = pred_df[name][
                    [f"pred_conf{k+1}" for k in range(len(output_features))]
                ]
                try:
                    active_results[f"{name}_auc"].append(
                        roc_auc_score(
                            scaled_data[f"{name}_y"], pred_probs, multi_class="ovr"
                        )
                    )
                except Exception:
                    active_results[f"{name}_auc"].append(np.nan)
                # print_str += f" {_split}_{metric} {value:.4f}

            active_df = pd.DataFrame.from_dict(active_results)
            active_df.to_csv(os.path.join(save_loc, "active_logs", f"active_train_log_{sidx}.csv"))
            # print(print_str)
            # my_iter.set_description(print_str)
            # my_iter.refresh()