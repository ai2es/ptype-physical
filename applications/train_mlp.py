import os
import sys
import tqdm
import yaml
import pickle
import shutil
import logging
import warnings
import numpy as np
import pandas as pd
from ptype.callbacks import MetricsCallback
from ptype.data import load_ptype_uq, preprocess_data
from pytpe.trainer import trainer
from mlguess.keras.callbacks import get_callbacks
from mlguess.keras.models import CategoricalDNN
from mlguess.pbs import launch_pbs_jobs
from bridgescaler import save_scaler
from argparse import ArgumentParser

warnings.filterwarnings("ignore")


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
    parser.add_argument(
        "-l",
        dest="launch",
        type=bool,
        default=False,
        help="Launch n_splits number of qsub jobs.",
    )
    parser.add_argument(
        "-s",
        dest="serial",
        type=bool,
        default=False,
        help="Whether to parallelize the training over GPUs (default is 0)",
    )
    parser.add_argument(
        "-i",
        dest="split_id",
        type=int,
        default=0,
        help="Which split this node will run (ranges from 0 to n_splits-1)",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    launch = bool(int(args_dict.pop("launch")))
    this_split = int(args_dict.pop("split_id"))
    run_serially = bool(int(args_dict.pop("serial")))

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # If we are running the training and not launching
    n_splits = conf["ensemble"]["n_splits"]
    mc_steps = conf["ensemble"]["mc_steps"]

    assert this_split <= (
        n_splits - 1
    ), "The worker ID is larger than the number of cross-validation n_splits."

    # Create the save directory if does not exist
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(os.path.join(save_loc, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_loc, "evaluate"), exist_ok=True)

    # Copy the model config file to the new directory
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    if launch:
        from pathlib import Path

        script_path = Path(__file__).absolute()
        logging.info("Launching to PBS")
        if run_serially:
            # If we are running serially, launch only one job
            # set serial flag = True
            launch_pbs_jobs(config_file, script_path, args="-s 1")
        else:
            # Launch QSUB jobs and exit
            for split in range(n_splits):
                # launch_pbs_jobs
                launch_pbs_jobs(config_file, script_path, args=f"-i {split}")
        sys.exit()

    # Run in serial over the number of ensembles (one at a time)
    if run_serially:
        for split in tqdm.tqdm(range(n_splits)):
            trainer(conf, data_split=split, mc_forward_passes=mc_steps)

    # Run one ensemble
    else:
        trainer(conf, data_split=this_split, mc_forward_passes=mc_steps)