# --- cell 1 ---
import numpy as np

def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):

    np.random.seed(seed)

    n = len(y)
    indices = np.random.permutation(n)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Baseline: predict training target mean
    baseline = np.mean(y[train_idx])

    test_predictions = np.full(len(test_idx), baseline)

    return test_predictions, train_idx, val_idx, test_idx