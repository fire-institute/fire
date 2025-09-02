# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import os
import pandas as pd
from ..common.config import DATA_PATH

def load_AStock_info() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        columns = pd.read_feather(os.path.join(DATA_PATH, "columns.feather"))
        index = pd.read_feather(os.path.join(DATA_PATH, "index.feather"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}, please download data first")
    return columns, index