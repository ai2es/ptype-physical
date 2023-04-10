import logging, tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import copy
import time, yaml, glob
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy import crs as ccrs
from cartopy import feature as cfeature

from reliability import reliability_diagram, reliability_diagrams, compute_calibration
from plotting import plot_confusion_matrix
from losses import *
from seed import torch_seed_everything
from metrics import *

def trainer(conf):
    df = pd.read_parquet(conf['data_path'])

    # model config
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    upsampling = conf['upsampling']
    train_path = f"{conf['train_path']}{upsampling}.pt"
    val_path = f"{conf['val_path']}{upsampling}.pt"
    save_path = conf['save_path']
    model_name = f"{conf['model_name']}{upsampling}"
    
    seed = conf['trainer']['seed']
    epochs = conf['trainer']['epochs']
    num_hidden_layers = conf['trainer']['num_hidden_layers']
    hidden_size = conf['trainer']['hidden_size']
    dropout_rate = conf['trainer']['dropout_rate']
    batch_size = conf['trainer']['batch_size']
    learning_rate = conf['trainer']['learning_rate']
    activation = conf['trainer']['activation']
    criterion = conf['trainer']['criterion']
    metrics = conf['trainer']['metrics']

    # set seed
    torch_seed_everything(seed)
    
    # load the training, validation, and test splits
    train_split = torch.load(train_path)
    valid_split = torch.load(val_path)
    
    train_loader = DataLoader(train_split, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)
    
    valid_loader = DataLoader(valid_split, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          num_workers=0)
    
    dataloaders = {
    "train": train_loader,
    "val": valid_loader,
    }
    
    def load_mlp_model(input_size, hidden_size, output_size, num_hidden_layers, dropout_rate, activation):
        activation_dict = {'leaky':nn.LeakyReLU(), 'elu':nn.ELU(), 'relu':nn.ReLU() , 'selu':nn.SELU()}
        activation_fn = activation_dict[activation]

        model = nn.Sequential()
        model.append(nn.utils.spectral_norm(nn.Linear(input_size, hidden_size)))
        model.append(activation_fn)

        for i in range(num_hidden_layers):
            if num_hidden_layers == 1:
                model.append(nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)))
                model.append(activation_fn)
            else:
                model.append(nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)))
                model.append(activation_fn)
                model.append(nn.Dropout(dropout_rate))

        model.append(nn.utils.spectral_norm(nn.Linear(hidden_size, output_size)))
        # print(model)

        return model

    model = load_mlp_model(len(features), hidden_size, len(outputs), num_hidden_layers, dropout_rate, activation)
    
    def one_hot_embedding(labels, num_classes=10):
        # Convert to One Hot Encoding
        y = torch.eye(num_classes)
        return y[labels]
    
    def train_model(
        model,
        dataloaders,
        optimizer,
        num_classes=4,
        stopping_patience=4,
        minimize=False,
        scheduler=None,
        num_epochs=25,
        device=None,
        uncertainty=False,
        metrics=["torch_acc"],
        criterion="edl_mse_loss"
    ):
        criterion_dict = {'edl_mse_loss':edl_mse_loss, 'edl_log_loss':edl_log_loss, 
                          'edl_digamma_loss':edl_digamma_loss}
        criterion = criterion_dict[criterion]
        metric_dict = {'torch_acc':torch_acc, 'torch_average_acc':torch_average_acc, 
                      'torch_ece':torch_ece, 'torch_balanced_ece':torch_balanced_ece}
        metric_fns = [metric_dict[metric] for metric in metrics]
        
        since = time.time()

        if not device:
            device = get_device()

        best_model_wts = copy.deepcopy(model.state_dict())
        if minimize:
            best_metrics = [100.0]
        else:  
            best_metrics = [0.0]

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
                            uncertainties = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                            probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
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
                    for i, metric in enumerate(metrics):
                        if 'ece' in metric:
                            metric_value = metric_fns[i](labels, probs)
                        else:
                            metric_value = metric_fns[i](labels, preds)
                            
                        results_dict[f"{metric}"].append(metric_value)

                    results_dict["loss"].append(loss.item())

                    if verbose:
                        print_str = f"Epoch: {epoch} "
                        print_str += f'{phase}_loss: {np.mean(results_dict["loss"]):.4f} '
                        for i, metric in enumerate(metrics):
                            if i == (len(metrics)-1):
                                print_str += f'{phase}_{metric}: {np.mean(results_dict[f"{metric}"]):.4f}'
                            else:
                                print_str += f'{phase}_{metric}: {np.mean(results_dict[f"{metric}"]):.4f} '
                        my_iter.set_description(print_str)
                        my_iter.refresh()

                epoch_loss = np.mean(results_dict["loss"])
                epoch_metrics = [np.mean(results_dict[f"{metric}"]) for metric in metrics]

                if phase == "train":
                    training_results["train_loss"].append(epoch_loss)
                    for i, metric in enumerate(metrics):
                        training_results[f"train_{metric}"].append(epoch_metrics[i])
                else:
                    training_results["val_loss"].append(epoch_loss)
                    for i, metric in enumerate(metrics):
                        training_results[f"val_{metric}"].append(epoch_metrics[i])

                training_results["epoch"].append(epoch)

                if scheduler is not None:
                    if phase == "val":
                        scheduler.step(1-epoch_metrics[0])

                # deep copy the model
                if phase == "val" and (epoch_metrics[0] < best_metrics[0] if minimize else epoch_metrics[0] > best_metrics[0]):
                    best_metrics = [epoch_metric for epoch_metric in epoch_metrics]
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Stop training if we have not improved after X epochs
                if phase == "val":
                    best_epoch = [i for i,j in enumerate(
                        training_results[f"val_{metrics[0]}"]) if j == max(training_results[f"val_{metrics[0]}"])][0]  #only consider accuracy
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
        for i, metric in enumerate(metrics):
            print("Best val {}: {:4f}".format(metric, best_metrics[i]))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, optimizer, training_results, best_metrics
    
    device = get_device()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
    verbose= True
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience = 2, 
        verbose = verbose,
        min_lr = 1.0e-7
    )

    model, optimizer, training_results, best_metrics = train_model(
        model,
        dataloaders,
        optimizer,
        scheduler=lr_scheduler,
        num_epochs=epochs,
        device=device,
        uncertainty=True,
        metrics=metrics,
        criterion=criterion
    )
    
    torch.save(model.state_dict(), f'{save_path}{model_name}.pt')
    
    return model, optimizer, training_results, best_metrics
    

if __name__ == '__main__':
    config = f'evidential_config/asos072022_pl10fzra2.yml'
    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    model, optimizer, training_results, best_metrics = trainer(conf)