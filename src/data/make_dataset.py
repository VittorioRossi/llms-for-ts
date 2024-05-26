import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os 

from kaggle.api.kaggle_api_extended import KaggleApi
import json
import shutil

def download_dataset(dataset_folder = "data/raw"):
    
    if isinstance(dataset_folder, str):
        dataset_folder = Path(dataset_folder)
    # log into kaggle
    api = KaggleApi()
    api.authenticate()
    
    # if the folder does not exist, create it
    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)
    
    dataset_ref = 'vittoriorossi/zeroshot-llm4ts-benchmark'
    
    # download the dataset if not already downlaoded
    if not (dataset_folder / 'zeroshot-llm4ts-benchmark').exists():
        api.dataset_download_files(dataset_ref, 
                               path=dataset_folder, 
                               unzip=True)
    else: 
        # check when it was downloaded
        last_downloaded = os.path.getmtime(dataset_folder / 'zeroshot-llm4ts-benchmark')

        # check if the dataset is updated
        metadata = json.loads(api.dataset_metadata(dataset_ref))
        last_updated = metadata['lastUpdated']
        
        if last_downloaded < last_updated:
            # clean the dataset_folder
            print(f"Dataset {dataset_ref} has been updated. Downloading the new version")
            shutil.rmtree(dataset_folder)
            dataset_folder.mkdir(parents=True)
            api.dataset_download_files(dataset_ref, 
                               path=dataset_folder, 
                               unzip=True,
                               force=True)


    return dataset_folder / 'zeroshot-llm4ts-benchmark'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default="data/raw")
@click.argument('output_filepath', type=click.Path(), default="data/processed")
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    logger.info('Downloading dataset')
    raw_dataset_folder = download_dataset(input_filepath)
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
