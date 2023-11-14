from echo.src.base_objective import BaseObjective
from echo.src.trial_suggest import trial_suggest_loader
from tensorflow.keras import backend as K
from pytpe.trainer import trainer
import logging 
import optuna
import gc


logger = logging.getLogger(__name__)


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric)

    def train(self, trial, conf):
        K.clear_session()
        gc.collect()
        conf["ensemble"]["n_splits"] = 1
        if "CSVLogger" in conf["callbacks"]:
            del conf["callbacks"]["CSVLogger"]
        if "ModelCheckpoint" in conf["callbacks"]:
            del conf["callbacks"]["ModelCheckpoint"]
        if "rain_weight" in conf["optuna"]["parameters"]:
            conf = self.custom_updates(trial, conf)
        try:
            return {self.metric: trainer(conf, evaluate=False)}
        except Exception as E:
            if (
                "Unexpected result" in str(E)
                or "CUDA" in str(E)
                or "FAILED_PRECONDITION" in str(E)
                or "CUDA_ERROR_ILLEGAL_ADDRESS" in str(E)
                or "ResourceExhaustedError" in str(E)
                or "Graph execution error" in str(E)
            ):
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