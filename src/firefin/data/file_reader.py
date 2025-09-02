# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import os
import pandas as pd
from ..common.config import DATA_PATH

data_path = os.path.join(os.path.dirname(__file__), 'raw')

# TODO: support other file types
# TODO: support start and end date, only read the data in the range
def read_feather(names):
    try:
        result = {n : pd.read_feather(f"{DATA_PATH}/{n}.feather") for n in names}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}, please download data first")
    return result


def file_reader(info: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    # TODO: support other file types
    feather_reader_names = info['feather']
    return read_feather(feather_reader_names)