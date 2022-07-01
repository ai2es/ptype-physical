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