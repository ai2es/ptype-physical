from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
)
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from typing import List, Dict
import numpy as np
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)


def get_callbacks(config: Dict[str, str]) -> List[Callback]:
    callbacks = []
    if "callbacks" in config:
        save_loc = config["save_loc"]
        config = config["callbacks"]
    else:
        return []
    if "ModelCheckpoint" in config:
        config["ModelCheckpoint"]["filepath"] = os.path.join(
            save_loc, config["ModelCheckpoint"]["filepath"]
        )
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer")
    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")
    # LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer
    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")
    if "CSVLogger" in config:
        config["CSVLogger"]["filename"] = os.path.join(
            save_loc, config["CSVLogger"]["filename"]
        )
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")
    return callbacks


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = {}) -> None:
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x, y, name="val"):
        super(MetricsCallback, self).__init__()
        self.x = x
        self.y = y
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.x))
        logs[f"{self.name}_csi"] = self.mean_csi(y_pred)
        true_labels = np.argmax(self.y, 1)
        pred_labels = np.argmax(y_pred, 1)
        prec, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro"
        )
        logs[f"{self.name}_acc"] = balanced_accuracy_score(true_labels, pred_labels)
        logs[f"{self.name}_prec"] = prec
        logs[f"{self.name}_recall"] = recall
        logs[f"{self.name}_f1"] = f1
        return

    def mean_csi(self, pred_probs):
        pred_labels = np.argmax(pred_probs, 1)
        confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
        rocs = []
        for i in range(pred_probs.shape[1]):
            forecasts = confidences.copy()
            obs = np.where(np.argmax(self.y, 1) == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts[:, 0], obs)
            rocs.append(roc.max_csi())
        return np.mean(rocs)
