# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import typing
from functools import partial

import numpy as np
import pandas as pd

from ..common.const import MIN_BARTIMES

IndexType = typing.Literal["d", "day", "m", "min", "minute", "l1", "l2"]
MockType = typing.Literal["rand", "norm", "price", "volume", "return", "arange"]


def _get_l2_seconds():
    morning = pd.timedelta_range("09:15:00", "11:30:00", freq="3 s")
    afternoon = pd.timedelta_range("13:00:00", "15:30:00", freq="3 s")
    return morning.union(afternoon)


l2_seconds = _get_l2_seconds()


def _index_maker(n, index_type: IndexType = "day"):
    if index_type in ("d", "day"):
        return pd.date_range("2010/1/1", periods=n, name="trade_date").strftime("%Y-%m-%d")
    elif index_type in ("m", "min", "minute"):
        n_days, n_minute = divmod(n, len(MIN_BARTIMES))
        if n_minute > 0:
            n_days += 1
        day_part = _index_maker(n_days, index_type="day")
        total_index = pd.MultiIndex.from_product([day_part, MIN_BARTIMES], names=["trade_date", "bartime"])
        if n_minute == 0:
            return total_index
        else:
            return total_index[: -(len(MIN_BARTIMES) - n_minute)]
    elif index_type in ("l1", "l2"):
        n_days, n_sec = divmod(n, 5702)
        if n_sec > 0:
            n_days += 1
        day_part = _index_maker(n_days, index_type="day")

        total_index = pd.concat(
            [pd.Series(0, index=pd.DatetimeIndex([dt]).repeat(5702) + l2_seconds) for dt in day_part]
        ).index
        if n_sec == 0:
            return total_index
        else:
            return total_index[: -(5702 - n_sec)]
    raise NotImplementedError(f"index_type {index_type} not implemented")


def _nb_random(shape, mock):
    if mock == "rand":
        return np.random.random(shape)
    elif mock == "norm":
        return np.random.randn(*shape)
    elif mock == "return":
        return np.random.normal(0.0, 0.03, shape)
    elif mock == "price":
        rt = _nb_random(shape, mock="return")
        price = (rt + 1).cumprod().reshape((shape[0], -1))
        price *= np.exp(np.random.normal(3.5, 1.06, price.shape[-1]))
        return price.reshape(shape)
    elif mock == "volume":
        return np.exp(np.random.normal(14.26, 1.29, shape))
    elif mock == "arange":
        total = 1
        for s in shape:
            total *= s
        return np.arange(total, dtype=np.float64).reshape(shape)


def _value_maker(shape, fill_value=np.nan, mock: MockType = "rand"):
    if fill_value is np.nan:
        if mock in MockType.__args__:
            return _nb_random(shape, mock)
        else:
            raise ValueError(f"mock {mock} not implemented")
    else:
        return np.full(shape, fill_value)


def _generate_stock_code(i):
    c = f"{i:06}."
    if not c.startswith(("0", "3", "6")):
        c = ("0", "3", "6")[int(c[0]) % 3] + c[1:]

    if c.startswith(("0", "3")):
        c += "SZ"
    else:
        c += "SH"
    return c


def _columns_maker(n):
    return pd.Index(sorted(map(_generate_stock_code, range(n))), name="stock_code")


def gen_df(*shape, fill_value=np.nan, index: IndexType = "day", mock: MockType = "rand", **joblib_kwargs):
    """quickly generate stock like DataFrames for test"""
    if not shape:
        shape = (10, 3)
    shape = tuple(np.ravel(shape))

    index_maker = partial(_index_maker, index_type=index)
    value_maker = partial(_value_maker, fill_value=fill_value, mock=mock)

    if len(shape) == 1:
        container = pd.Series
        idx_col = {"index": index_maker(shape[0])}
    elif len(shape) == 2:
        container = pd.DataFrame
        idx_col = {"index": index_maker(shape[0]), "columns": _columns_maker(shape[1])}
    else:
        raise NotImplementedError(f"shape {shape} not implemented")

    out = container(value_maker(shape), **idx_col)
    return out
