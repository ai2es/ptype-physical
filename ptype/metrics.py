import numpy as np
import pandas as pd
from collections import OrderedDict
import torch

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def acc(y_true, y_pred):
    return (y_true == y_pred).mean()

def torch_acc(labels, preds):
    return torch.mean((preds == labels.data).float()).item()

def average_acc(y_true, y_pred):
    pred_labels = np.argmax(y_pred, 1)
    true_labels = np.argmax(y_true, 1)
    accs = []
    for _label in np.unique(true_labels):
        c = np.where(true_labels == _label)
        ave_acc = (true_labels[c] == pred_labels[c]).mean()
        accs.append(ave_acc)
    acc = np.mean(accs)
    return acc

def torch_average_acc(true_labels, pred_labels):
    accs = []
    for _label in true_labels.unique():
        c = torch.where(true_labels == _label)
        ave_acc = (true_labels[c] == pred_labels[c]).float().mean().item()
        accs.append(ave_acc)
    acc = np.mean(accs)
    return acc

def ece(y_true, y_pred):
    """
    Calculates the expected calibration error of the
    neural network.
    """
    confidences = np.max(y_pred, 1)
    pred_labels = np.argmax(y_pred, 1)
    true_labels = np.argmax(y_true, 1)
    num_bins = 10
    
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    
    return ece*100

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

def balanced_ece(y_true, y_pred):
    """
    Calculates the balanced expected calibration error of the
    neural network.
    """
    probs = np.max(y_pred, 1)
    preds = np.argmax(y_pred, 1)
    labels = np.argmax(y_true, 1)
    num_bins = 10
    
    test_data = pd.DataFrame.from_dict(
        {"pred_labels": preds,
         "true_labels": labels, 
         "pred_conf": probs})
    
    cond0 = (test_data["true_labels"] == 0)
    cond1 = (test_data["true_labels"] == 1)
    cond2 = (test_data["true_labels"] == 2)
    cond3 = (test_data["true_labels"] == 3)
    results = OrderedDict()
    results['ra_percent'] = {
        "true_labels": test_data[cond0]["true_labels"].values,
        "pred_labels": test_data[cond0]["pred_labels"].values,
        "confidences": test_data[cond0]["pred_conf"].values
    }
    results['sn_percent'] = {
        "true_labels": test_data[cond1]["true_labels"].values,
        "pred_labels": test_data[cond1]["pred_labels"].values,
        "confidences": test_data[cond1]["pred_conf"].values
    }
    results['pl_percent'] = {
        "true_labels": test_data[cond2]["true_labels"].values,
        "pred_labels": test_data[cond2]["pred_labels"].values,
        "confidences": test_data[cond2]["pred_conf"].values
    }
    results['fzra_percent'] = {
        "true_labels": test_data[cond3]["true_labels"].values,
        "pred_labels": test_data[cond3]["pred_labels"].values,
        "confidences": test_data[cond3]["pred_conf"].values
    }
    
    ece_list = []

    for i, (name, data) in enumerate(results.items()):
        pred_labels = data["pred_labels"]
        true_labels = data["true_labels"]
        confidences = data["confidences"]
        
        assert(len(confidences) == len(pred_labels))
        assert(len(confidences) == len(true_labels))
        assert(num_bins > 0)
    
        bin_size = 1.0 / num_bins
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float64)
        bin_confidences = np.zeros(num_bins, dtype=np.float64)
        bin_counts = np.zeros(num_bins, dtype=np.int32)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
                bin_confidences[b] = np.mean(confidences[selected])
                bin_counts[b] = len(selected)

        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        ece_list.append(ece)
    
    balanced_ece = np.mean(ece_list)
    
    return balanced_ece*100

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