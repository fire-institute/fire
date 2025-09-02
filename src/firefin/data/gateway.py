# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import numpy as np
import pandas as pd
from ..common.config import logger, DATA_MAPS
from .datainfo import load_AStock_info
from .fake import gen_df
from .file_reader import file_reader

def _get_clean_names(names) -> list:
    output = []

    def _add_from_str(s):
        for n in s.split(","):
            n = n.replace(" ", "")
            if n and n not in output:
                output.append(n)

    for name in np.ravel(names):
        if isinstance(name, str):
            _add_from_str(name)
        else:
            # we assume it's iterable of strings
            for _name in name:
                _add_from_str(_name)
    return output


def _get_clean_se(start=None, end=None, dates=None):
    """basic checks, no transformation for input ts"""
    if dates is None:
        if start is not None:
            assert np.ndim(start) == 0, f"start must be a scalar, got {start}"
        if end is not None:
            assert np.ndim(end) == 0, f"end must be a scalar, got {end}"

    else:
        if start is not None or end is not None:
            raise ValueError("start and end cannot be used with dates")
        else:
            if isinstance(dates, slice):
                dates = [dates.start, dates.stop]
            elif pd.api.types.is_list_like(dates):
                pass
            else:
                # str, datetime-like
                dates = np.atleast_1d(dates)
            start, end = dates[0], dates[-1]

    return start, end


def _parse_args(names, start_date, end_date, dates):
    """
    parse names, start_date, end_date

    Notes
    -----
    define name string as a str of a single data or a str of multiple data names separated by comma.
    `names` can be: single name sting, iterable of name sting, or an iterable containing name string
    and trailing datetime-like
    if `names` has trailing datetime-like, `start_date`, `end_date` and `dates` should be None

    Examples
    --------
    names can be:
        "close"
        "close, open"
        ["close", "open"]
        ["close, open"]
    tailing datetime-like can be:
        "2020/1/1"
        ["2020/1/1", "2020/1/2"]
        slice("2020/1/1", "2020/1/2")

    """

    def is_datetime_like(obj):
        try:
            pd.to_datetime(obj)
        except Exception:
            return False
        else:
            return True

    if dates is None:
        # datetime is list-like or slice
        if isinstance(n1 := names[-1], slice):
            names = names[:-1]
            start_date, end_date = _get_clean_se(start_date, end_date, dates=n1)
        elif pd.api.types.is_list_like(n1) and is_datetime_like(t := np.ravel(n1)):
            names = names[:-1]
            start_date, end_date = _get_clean_se(start_date, end_date, dates=t)
        else:
            if is_datetime_like(n1):
                end_date = n1
                names = names[:-1]
                if len(names) >= 2 and is_datetime_like(n2 := names[-1]):
                    start_date = n2
                    names = names[:-1]
                else:
                    start_date = end_date
            start_date, end_date = _get_clean_se(start_date, end_date)
    else:
        start_date, end_date = _get_clean_se(start_date, end_date, dates=dates)

    return _get_clean_names(names), start_date, end_date


def check_if_valid(names: list[str]) -> dict[str, bool]:
    return {n: n in DATA_MAPS.keys() for n in names}


def fetch_data(
    *args,
    names=None,
    start_date=None,
    end_date=None,
    dates=None,
    market_range="ALL",
) -> dict[str, pd.DataFrame]:
    if names is None:
        names = args
    elif args:
        raise ValueError("you may only use `names` or `*args` to specify the data to be queried")

    names, start_date, end_date = _parse_args(names, start_date, end_date, dates)

    results = {}
    if not names:
        return results

    valid = check_if_valid(names)

    for k, v in valid.items():
        if not v:
            columns, index = load_AStock_info()
            logger.warning(f"{k} is not a valid data name, mock with random data")
            results[k] = gen_df((len(index), len(columns)))
            names.remove(k)

    if len(names) == 0:
        return results

    # only support file reader for now
    file_reader_names = dict()

    for name in names:
        try:
            l, r = DATA_MAPS[name].split("::")  # noqa: E741
        except Exception as e:
            logger.error(f"cannot find data source for {name}, reason: {e}")
            continue

        if l == "file":
            if r not in file_reader_names:
                file_reader_names[r] = [name]
            else:
                file_reader_names[r].append(name)
        else:
            raise ValueError(f"{name} unsupported data source: {l}::{r}")

    # only support file reader for now
    results.update(file_reader(file_reader_names))
    return results
