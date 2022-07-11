import yaml
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import logging, tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import optuna
from echo.src.base_objective import BaseObjective
from typing import List, Dict
import sys
import random
import os
import copy
import time, glob
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def torch_seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def torch_average_acc(labels, preds):
    ra = 0
    ra_tot = 0
    sn = 0
    sn_tot = 0
    pl = 0
    pl_tot = 0
    fzra = 0
    fzra_tot = 0
    preds = preds.cpu()
    labels = labels.cpu()
    preds = preds.numpy()
    labels = labels.numpy()
    for i in range(len(preds)):
        if labels[i] == 0:
            if preds[i] == 0:
                ra += 1
            ra_tot += 1
        if labels[i] == 1:
            if preds[i] == 1:
                sn += 1
            sn_tot += 1
        if labels[i] == 2:
            if preds[i] == 2:
                pl += 1
            pl_tot += 1
        if labels[i] == 3:
            if preds[i] == 3:
                fzra += 1
            fzra_tot += 1
    try:
        ra_acc = ra/ra_tot
    except ZeroDivisionError:
        ra_acc = np.nan
    try:
        sn_acc = sn/sn_tot
    except ZeroDivisionError:
        sn_acc = np.nan
    try:
        pl_acc = pl/pl_tot
    except ZeroDivisionError:
        pl_acc = np.nan
    try:
        fzra_acc = fzra/fzra_tot
    except ZeroDivisionError:
        fzra_acc = np.nan
        
    acc = [ra_acc, sn_acc, pl_acc, fzra_acc]
    return np.nanmean(acc, dtype=np.float64)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

# metric, metric function, and loss function definitions
metric = "torch_average_acc"
eval_metric = "valid_torch_average_acc"
metric_fn = torch_average_acc
criterion = edl_mse_loss

class Objective(BaseObjective):

    def __init__(self, config, metric=eval_metric, device="cuda"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            model, optimizer, training_results = trainer(conf)
        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
                raise optuna.TrialPruned()
            elif "reraise" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}.")
                raise optuna.TrialPruned()
            else:
                logging.warning(
                    f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E
        results_dict = {self.metric:max(training_results[self.metric])}
        return results_dict
    
def trainer(conf):
    # load data
    df = pd.read_parquet(conf['data_path'])
    
    # model config
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    n_splits = conf['trainer']['n_splits']
    train_size1 = conf['trainer']['train_size1'] # sets test size
    train_size2 = conf['trainer']['train_size2'] # sets valid size
    seed = conf['trainer']['seed']
    epochs = conf['trainer']['epochs']
    num_hidden_layers = conf['trainer']['num_hidden_layers']
    hidden_size = conf['trainer']['hidden_size']
    dropout_rate = conf['trainer']['dropout_rate']
    batch_size = conf['trainer']['batch_size']
    learning_rate = conf['trainer']['learning_rate']
    activation = conf['trainer']['activation']

    # set seed
    torch_seed_everything(seed)

    # split and preprocess the data
    df['day'] = df['datetime'].apply(lambda x: str(x).split(' ')[0])

    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size1, random_state=seed)
    train_idx, test_idx = list(splitter.split(df, groups=df['day']))[0]
    train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]

    splitter = GroupShuffleSplit(n_splits=n_splits, train_size=train_size2, random_state=seed)
    train_idx, valid_idx = list(splitter.split(train_data, groups=train_data['day']))[0]
    train_data, valid_data = train_data.iloc[train_idx], train_data.iloc[valid_idx]

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(train_data[features])
    x_valid = scaler_x.transform(valid_data[features])
    x_test = scaler_x.transform(test_data[features])
    y_train = np.argmax(train_data[outputs].to_numpy(), 1)
    y_valid = np.argmax(valid_data[outputs].to_numpy(), 1)
    y_test = np.argmax(test_data[outputs].to_numpy(), 1)

    # convert splits to torch tensor datasets
    train_split = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_split, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0)

    valid_split = TensorDataset(
        torch.from_numpy(x_valid).float(),
        torch.from_numpy(y_valid).long()
    )
    valid_loader = DataLoader(valid_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)

    test_split = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_split, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=0)

    dataloaders = {
        "train": train_loader,
        "val": valid_loader,
        "test": test_loader,
    }
    
    # model structure
    def load_mlp_model(input_size, hidden_size, output_size, num_hidden_layers, dropout_rate, activation):
        activation_dict = {'leaky':nn.LeakyReLU(), 'elu':nn.ELU(), 'relu':nn.ReLU() , 'selu':nn.SELU()}
        activation_fn = activation_dict[activation]

        model = nn.Sequential()
        model.append(nn.utils.spectral_norm(nn.Linear(input_size, hidden_size)))
        model.append(activation_fn)

        for i in range(num_hidden_layers):
            model.append(nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)))
            model.append(activation_fn)

        model.append(nn.utils.spectral_norm(nn.Linear(hidden_size, output_size)))
    
        return model

    model = load_mlp_model(len(features), hidden_size, len(outputs), num_hidden_layers, dropout_rate, activation)
    
    # model training
    def one_hot_embedding(labels, num_classes=10):
        # Convert to One Hot Encoding
        y = torch.eye(num_classes)
        return y[labels]

    def train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        metric_fn,
        num_classes=4,
        stopping_patience=4,
        minimize=False,
        scheduler=None,
        num_epochs=25,
        device=None,
        uncertainty=False,
        metric="acc"
    ):

        since = time.time()

        if not device:
            device = get_device()

        best_model_wts = copy.deepcopy(model.state_dict())
        if minimize:
            best_metric = 100.0
        else:  
            best_metric = 0.0

        stop_early = False
        training_results = defaultdict(list)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    #print("Training...")
                    model.train()  # Set model to training mode
                else:
                    #print("Validating...")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                correct = 0

                if verbose:
                    total = int(np.ceil(len(dataloaders[phase].dataset) / batch_size))
                    my_iter = tqdm.tqdm(enumerate(dataloaders[phase]),
                                    total = total,
                                    leave = True)
                else:
                    my_iter = enumerate(dataloaders[phase])

                # Iterate over data.
                results_dict = defaultdict(list)
                for i, (inputs, labels) in my_iter:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):

                        if uncertainty:
                            y = one_hot_embedding(labels, num_classes)
                            y = y.to(device)
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(
                                outputs, y.float(), epoch, num_classes, 10, device 
                            )

                            match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                            acc = torch.mean(match)
                            evidence = relu_evidence(outputs)
                            alpha = evidence + 1
                            u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                            total_evidence = torch.sum(evidence, 1, keepdim=True)
                            mean_evidence = torch.mean(total_evidence)
                            mean_evidence_succ = torch.sum(
                                torch.sum(evidence, 1, keepdim=True) * match
                            ) / torch.sum(match + 1e-20)
                            mean_evidence_fail = torch.sum(
                                torch.sum(evidence, 1, keepdim=True) * (1 - match)
                            ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    metric_value = metric_fn(labels, preds)
                    results_dict["loss"].append(loss.item())
                    results_dict[f"{metric}"].append(metric_value)

                    if verbose:
                        print_str = f"Epoch: {epoch} "
                        print_str += f'{phase}_loss: {np.mean(results_dict["loss"]):.4f} '
                        print_str += f'{phase}_{metric}: {np.mean(results_dict[f"{metric}"]):.4f}'
                        my_iter.set_description(print_str)
                        my_iter.refresh()

                epoch_loss = np.mean(results_dict["loss"])
                epoch_metric = np.mean(results_dict[f"{metric}"])

                if phase == "train":
                    training_results["train_loss"].append(epoch_loss)
                    training_results[f"train_{metric}"].append(epoch_metric)
                else:
                    training_results["val_loss"].append(epoch_loss)
                    training_results[f"val_{metric}"].append(epoch_metric)

                training_results["epoch"].append(epoch)

                if scheduler is not None:
                    if phase == "val":
                        scheduler.step(1-epoch_metric)

                # deep copy the model
                if phase == "val" and (epoch_metric < best_metric if minimize else epoch_metric > best_metric):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Stop training if we have not improved after X epochs
                if phase == "val":
                    best_epoch = [i for i,j in enumerate(
                        training_results[f"val_{metric}"]) if j == max(training_results[f"val_{metric}"])][0]
                    offset = epoch - best_epoch
                    if offset >= stopping_patience:
                        stop_early = True
                        break
            if stop_early:
                break

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val {}: {:4f}".format(metric, best_metric))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, optimizer, training_results
    
    device = get_device()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    verbose= True
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=2, 
        verbose=verbose,
        min_lr=1.0e-7
    )

    model, optimizer, training_results = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        metric_fn,
        scheduler=lr_scheduler,
        num_epochs=epochs,
        device=device,
        uncertainty=True,
        metric=metric
    )
    
    return model, optimizer, training_results
    
if __name__ == '__main__':
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    config = '~/winter-ptype/evidential_config/model.yml'
    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    model, optimizer, training_results = trainer(conf)
    print(max(training_results[metric]))