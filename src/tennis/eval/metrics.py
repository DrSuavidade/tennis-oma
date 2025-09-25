import numpy as np

def log_loss(y_true, p):
    eps = 1e-15
    p = np.clip(p, eps, 1-eps)
    return -(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean()

def brier(y_true, p):
    return ((p - y_true)**2).mean()

def ece(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    inds = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if mask.any():
            conf = p[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return ece
