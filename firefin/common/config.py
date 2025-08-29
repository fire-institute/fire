# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt
import os
import pathlib
import yaml
import json
from loguru import logger

# Load configuration from YAML file, windows with default Decode
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

# Load configuration from YAML file

# Define DATA_PATH based on configuration, expanding user and resolving path

if os.name == "posix":
    DATA_PATH = config.get("paths", {}).get("unix", "~/.fire/data/raw/")
else:
    DATA_PATH = config.get("paths", {}).get("windows", "D:\\Data\\AStockData\\raw")

# resolve ~ and envars
DATA_PATH = pathlib.Path(DATA_PATH).expanduser().resolve()

# load data maps from config
DATA_MAPS = config.get("data_maps", {})

json_files = list(DATA_PATH.glob("*.json"))
if json_files:
    for json_file in json_files:
        with open(json_file, "r") as f:
            DATA_MAPS.update(json.load(f))
else:
    import multiprocessing as _mp

    if _mp.current_process().name == "MainProcess":
        logger.info("No additional JSON files found in DATA_PATH, load default DATA_MAPS.")
