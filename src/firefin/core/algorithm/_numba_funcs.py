# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import numpy as np
from numba import njit


@njit
def _validate_pairwise(x, y):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("_validate_pairwise: Both inputs must be 2D arrays")
    n1, m1 = x.shape
    n2, m2 = y.shape
    if n1 != n2:
        raise ValueError("_validate_pairwise: Both inputs must have the same number of rows")
    if m1 != m2 and min(m1, m2) != 1:
        raise ValueError("_validate_pairwise: Both inputs must have the same number of columns or one column")


@njit
def _corr_pearson(x, y):
    assert len(x) == len(y)
    msk = np.isfinite(x) & np.isfinite(y)
    if msk.sum() <= 3:
        return np.nan
    elif msk.all():
        x_ = x
        y_ = y
    else:
        x_ = x[msk]
        y_ = y[msk]
    mean_x = np.mean(x_)
    mean_y = np.mean(y_)
    x_centered = x_ - mean_x
    y_centered = y_ - mean_y
    var_x = np.sum(x_centered**2)
    if var_x == 0:
        return np.nan
    var_y = np.sum(y_centered**2)
    if var_y == 0:
        return np.nan
    cov = np.sum(x_centered * y_centered)
    return cov / np.sqrt(var_x * var_y)


@njit
def corr(x, y, method="pearson"):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("corr: Both inputs must be 1D arrays")
    if x.shape != y.shape:
        raise ValueError("corr: Both inputs must have the same shape")
    if method == "pearson":
        return _corr_pearson(x, y)
    else:
        raise NotImplementedError("corr: Only Pearson correlation is supported")


@njit
def ts_corr(x, y, w, method="pearson"):
    _validate_pairwise(x, y)
    n, m1 = x.shape
    _, m2 = y.shape
    k = max(m1, m2)
    out = np.full((n, k), np.nan)
    for i in range(n):
        x_ = x[max(0, i - w + 1) : i + 1]
        y_ = y[max(0, i - w + 1) : i + 1]
        for j in range(k):
            x__ = x_[:, min(j, m1 - 1)]
            y__ = y_[:, min(j, m2 - 1)]
            out[i, j] = corr(x__, y__, method)
    return out
