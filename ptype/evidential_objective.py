import yaml
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import logging
import tqdm
import optuna
from echo.src.base_objective import BaseObjective
import random
import os
import copy
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)
        
config1 = '/glade/u/home/jwillson/winter-ptype/code/evidential_config/hyper.yml'
with open(config1) as f:       
    conf1 = yaml.load(f, Loader=yaml.FullLoader)

eval_metrics = conf1['optuna']['metric']

def torch_seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device
    

def torch_acc(labels, preds):
    return torch.mean((preds == labels.data).float()).item()


def torch_average_acc(true_labels, pred_labels):
    accs = []
    for _label in true_labels.unique():
        c = torch.where(true_labels == _label)
        ave_acc = (true_labels[c] == pred_labels[c]).float().mean().item()
        accs.append(ave_acc)
    acc = np.mean(accs)
    return acc


def torch_ece(true_labels, pred_probs):
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = torch.max(pred_probs, 1)[0]
    predictions = torch.argmax(pred_probs, 1)
    accuracies = predictions.eq(true_labels)
    ece = torch.zeros(1, device=get_device())
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()*100 


def torch_balanced_ece(true_labels, pred_probs):
    class_ece = []
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confs = torch.max(pred_probs, 1)[0]
    preds = torch.argmax(pred_probs, 1)
    for _label in true_labels.unique():
        c = torch.where(true_labels == _label)
        confidences = confs[c]
        predictions = preds[c]
        accuracies = predictions.eq(true_labels[c])
        ece = torch.zeros(1, device=get_device())
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        class_ece.append(ece.item()*100)
    return np.mean(class_ece)


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

class Objective(BaseObjective):

    def __init__(self, config, metric=eval_metrics, device="cuda"):

        """Initialize the base class"""
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            model, optimizer, training_results, best_metrics = trainer(conf)
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
        results_dict = {metric:best_metrics[i] for i, metric in enumerate(self.metric)}
        return results_dict   
    
def trainer(conf):
    df = pd.read_parquet(conf['data_path'])

    # model config
    features = conf['tempvars'] + conf['tempdewvars'] + conf['ugrdvars'] + conf['vgrdvars']
    outputs = conf['outputvars']
    upsampling = conf['upsampling']
    train_path = f"{conf['train_path']}{upsampling}.pt"
    val_path = f"{conf['val_path']}{upsampling}.pt"
    save_path = conf['save_path']
    
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
    
    return model, optimizer, training_results, best_metrics
    
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
    
    config = '/glade/u/home/jwillson/winter-ptype/code/evidential_config/config.yml'
    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        
    model, optimizer, training_results, best_metrics = trainer(conf)