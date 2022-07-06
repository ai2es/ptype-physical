import numpy as np

def average_acc(y_true, y_pred):
    """
    Calculates the individual accuracy of each ptype class and 
    returns the average accuracy. When a class isn't present, it is
    ignored in the calculation of the average.
    """
    ra = 0
    ra_tot = 0
    sn = 0
    sn_tot = 0
    pl = 0
    pl_tot = 0
    fzra = 0
    fzra_tot = 0
    preds = np.argmax(y_pred, 1)
    labels = np.argmax(y_true, 1)
    for i in range(len(y_true)):
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