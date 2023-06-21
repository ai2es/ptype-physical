import yaml, glob
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from typing import List, Dict
import logging

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
   
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.xlim(-0.5, len(np.unique(y))-0.5)
    # plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax

logger = logging.getLogger(__name__)

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = {}) -> None:
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def get_callbacks(config: Dict[str, str]) -> List[Callback]:
        callbacks = []
        if "callbacks" in config:
            config = config["callbacks"]
        else:
            return []
        if "ModelCheckpoint" in config:
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
            callbacks.append(CSVLogger(**config["CSVLogger"]))
            logger.info("... loaded CSVLogger")
        return callbacks













