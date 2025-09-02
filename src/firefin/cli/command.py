# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import os
import sys
import click
import subprocess
from ..common.config import DATA_PATH, logger


@click.group()
def cli():
    pass


@click.command(help="Display help")
def help():
    click.echo("Help")


def _prepare_folder():
    # check if data directory exists
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Data directory is created.")
    else:
        logger.info("Data directory already exists. Skipping creating directory.")


# TODO: Add more data source
@click.command(help="Download data")
@click.option('--data_url', default=None, help='download from provided url')
def download(data_url):
    logger.info("Preparing Data for the first time ...")
    _prepare_folder()

    if data_url:
        if not data_url.startswith("http"):
            raise Exception("Please provide a valid url to download data from.")
        if not data_url.endswith(".tar.gz"):
            raise Exception("Please provide a valid url to download data from. The url should end with .tar.gz")
        data_file_name = data_url.split("/")[-1]
        raw_data_path = DATA_PATH / data_file_name
        request_url = data_url
    else:
        # download default package
        logger.info("No URL provided, will download default AStockData.tar.gz from GitHub.")
        raw_data_path = DATA_PATH / "AStockData.tar.gz"
        request_url = (
            "https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz"
        )

    # Check if gz file is exist
    if raw_data_path.exists():
        logger.info("Data already exists, Remove Data First...")
        # ensure the file is removed before downloading again
        raw_data_path.unlink(missing_ok=True)  # remove the file
        logger.info("Data Removed.")

    logger.info("Downloading data ...")
    # Download data from file server
    try:
        subprocess.run(f"wget {request_url} -O {raw_data_path}", shell=True, check=True)
        subprocess.run(
            f'tar -xvf {raw_data_path} -C {DATA_PATH} --strip-components=1',
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt (Ctrl+C), program terminated")
        sys.exit(0)
 

@click.command(help="Prepare data")
@click.argument("file_path", type=click.Path(exists=True))
def load(file_path: str = None):
    logger.info("Preparing Data for the first time ...")
    _prepare_folder()

    # tar unzip file, print progress
    try:
        subprocess.run(
            f'tar -xvf {file_path} -C {DATA_PATH} --strip-components=1',
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt (Ctrl+C), program terminated")
        sys.exit(0)

cli.add_command(help)
cli.add_command(download)
cli.add_command(load)