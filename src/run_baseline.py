#!/usr/bin/env python3
from evaluation import evaluate
from features import DATASET_LOADERS
from models.baselines import mean_forecast, naive_forecast, ses_forecast
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import click
import yaml

MODELS = {
    'mean': mean_forecast,
    'naive': naive_forecast,
    'ses': ses_forecast
}


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def run_experiment(model_name, dataset_name, window_size, target_size, batch_size=64, chunk_size=10, preds_path=None, limit_obs=None, stride=None):
    model_name_clean = model_name.split('/')[1] if '/' in model_name else model_name
    run_name = f'{model_name_clean}_{dataset_name}_baseline_{window_size}_{target_size}_{stride}'
    # check if dataset_name is in DATASET_LOADERS
    if dataset_name not in DATASET_LOADERS:
        logging.error(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
        return
        #raise ValueError(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
    

    logger.info(f'Running benchmark - {run_name}')

    logger.info('Loading dataset')

    dataset = DATASET_LOADERS[dataset_name]
    data_generator = dataset.process('base',
                                    window_size=window_size, 
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    chunksize=chunk_size,
                                    stride=stride)

    logger.info('Loading model')
    
    model = MODELS[model_name]

    logger.info('Running inference')
    preds = []
    true = []

    # observation is a batch contatinign (X, y) where X has size 64 x window_size and y has size 64 x target_size
    num_bateches = limit_obs//batch_size
    n_batches = 0
    for observation in tqdm(data_generator, total=num_bateches):
        cleaned_obs = [list(map(float, obs.strip().split())) for obs in observation[0]]
        prediction = [model(cl, target_size, seasonality=dataset.seasonality) for cl in cleaned_obs]

        preds.extend(prediction)
        true.extend(observation[1])

        if n_batches == num_bateches:
            break

        n_batches += 1

    preds = np.array(preds).reshape(-1, target_size).astype(float)
    true = np.array(true).reshape(-1, target_size).astype(float)

    if preds_path:
        logger.info(f'Saving predictions to {preds_path}')
        saving_path = Path(preds_path) / f'{run_name}.npy'
        saving_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(saving_path, preds)

    logger.info('Evaluating model')
    eval = evaluate(true, preds)
    logger.info(eval)

    logger.info(f'End of benchmark run - {run_name}')

    return eval



@click.command()
@click.option('--config_path', type=click.STRING, required=True, help='Path to configuration file')
def main(config_path):
    config = yaml.safe_load(open(config_path, 'r'))
    # take the config path title 
    config_name = Path(config_path).stem
    # create a directory to save the results
    results_dir = Path(config.get('results_path','experiments')) / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for experiment in config.get('experiments', []):
        dataset_name = experiment.get('dataset_name')
        window_size = experiment.get('window_size', 15)
        target_size = experiment.get('target_size', 1)
        batch_size = experiment.get('batch_size', 64)
        chunk_size = experiment.get('chunk_size', 10)
        limit_obs = experiment.get('limit_obs', 50_000)
        stride = experiment.get('stride', 1)

        if dataset_name not in DATASET_LOADERS:
            logging.error(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
            return

        for model_name in MODELS.keys():
            
            if not isinstance(window_size, list):
                window_size = [window_size]
            if not isinstance(target_size, list):
                target_size = [target_size]
            
            for ws in window_size:
                for ts in target_size:
                    run_name = f"{dataset_name}_{model_name}_{ws}_{ts}_{stride}"
                    evals = run_experiment(model_name,
                                        dataset_name,
                                        ws,
                                        ts,
                                        batch_size,
                                        chunk_size,
                                        results_dir,
                                        limit_obs,
                                        stride=stride)

                    saving_path = results_dir / ('baseline' + '.txt')
                    saving_path.parent.mkdir(parents=True, exist_ok=True)
                    string_to_write = f'{run_name}: {evals.__str__()}' + '\n'
                    with open(saving_path, 'a+') as f:
                        f.write(string_to_write)

if __name__=='__main__':
    main()