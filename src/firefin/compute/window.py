# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import pandas as pd
import typing
from ..core.algorithm import _numba_funcs

__all__ = ["ts_corr"]


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int, method: typing.Literal["pearson", "kendall", "spearman"]):
    x, y = x.align(y, join="outer", copy=False)
    result = pd.DataFrame(
        _numba_funcs.ts_corr(x.values, y.values, n, method),
        index=x.index,
        columns=x.columns,
    )
    return result
