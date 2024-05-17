# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def donwload_dataset():
    dataset_folder = Path("data/raw")
    # log into kaggle
    api = KaggleApi()
    api.authenticate()
    
    # if the folder does not exist, create it
    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)
    
    # download the dataset if not already downlaoded
    if not (dataset_folder / 'zeroshot-llm4ts-benchmark').exists():
        api.dataset_download_files('vittoriorossi/zeroshot-llm4ts-benchmark', 
                               path=dataset_folder, 
                               unzip=True)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('Downloading dataset')
    donwload_dataset()
    logger.info('Dataset downloaded')

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
