#!/usr/bin/env python

from evaluation import evaluate
from features import DATASET_LOADERS
from models.models import HuggingFaceLLM
from prompt.utils import get_available_templates
import logging
from tqdm import tqdm
import click
import numpy as np
from pathlib import Path
import yaml


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def run_experiment(model_name, dataset_name, prompt_name, window_size, target_size, batch_size=64, chunk_size=10, preds_path=None, univariate=False, limit_obs=None):
    run_name = f'{model_name}_{dataset_name}_{prompt_name}_{window_size}_{target_size}'
    # check if dataset_name is in DATASET_LOADERS
    if dataset_name not in DATASET_LOADERS:
        logging.error(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
        return
        #raise ValueError(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
    
    if prompt_name not in get_available_templates():
        logging.error(f'Prompt {prompt_name} not found. Available prompts are: default, random, template')
        #raise ValueError(f'Prompt {prompt_name} not found. Available prompts are: default, random, template')
        return

    logger.info(f'Running benchmark - {run_name}')

    logger.info('Loading dataset')

    data_generator = DATASET_LOADERS[dataset_name].process(prompt_name, 
                                                            window_size=window_size, 
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            chunksize=chunk_size)

    logger.info('Loading model')
    try:
        model = HuggingFaceLLM(model_name)
    except Exception as e:
        logger.error(f'Model {model_name} not found. Please check the model name and try again.')
        return
        #raise ValueError(f'Model {model_name} not found. Please check the model name and try again.')

    logger.info('Running inference')
    preds = []
    true = []

    # observation is a batch contatinign (X, y) where X has size 64 x window_size and y has size 64 x target_size
    num_bateches = limit_obs//batch_size
    n_batches = 0
    for observation in tqdm(data_generator, total=num_bateches):
        preds.extend(model.generate(observation[0]))
        true.extend(observation[1])
        
        n_batches += 1
        if n_batches == num_bateches:
            break


    preds = np.array(preds).reshape(-1, target_size).astype(float)
    true = np.array(true).reshape(-1, target_size).astype(float)

    if preds_path:
        logger.info(f'Saving predictions to {preds_path}')
        saving_path = Path(preds_path) / f'{run_name}.npy'
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
    

    for experiment in config['experiments']:
        model_name = experiment['model_name']
        dataset_name = experiment['dataset_name']
        prompt_name = experiment['prompt_name']
        window_size = experiment.get('window_size', 15)
        target_size = experiment.get('target_size', 1)
        batch_size = experiment.get('batch_size', 64)
        chunk_size = experiment.get('chunk_size', 10)
        univariate = experiment.get('univariate', False)
        limit_obs = experiment.get('limit_obs', 50_000)

    
        model_name_clean = model_name.split('/')[1]
        run_name = f"{model_name_clean}_{dataset_name}_{prompt_name}_{window_size}_{target_size}"
        if dataset_name == 'all':
            for dataset_name in DATASET_LOADERS.keys():
                evals = run_experiment(model_name,
                                       dataset_name,
                                       prompt_name,
                                       window_size,
                                       target_size,
                                       batch_size,
                                       chunk_size,
                                       results_dir,
                                       univariate,
                                       limit_obs)
                saving_path = results_dir / (run_name + '.txt')
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                with open(saving_path, 'a+') as f:
                    f.write(evals.__str__())

        else:
            evals = run_experiment(model_name,
                                   dataset_name,
                                   prompt_name,
                                   window_size,
                                   target_size,
                                   batch_size,
                                   chunk_size,
                                   results_dir,
                                   univariate,
                                   limit_obs)

            saving_path = results_dir / (run_name + '.txt')
            saving_path.parent.mkdir(parents=True, exist_ok=True)
            with open(saving_path, 'a+') as f:
                f.write(evals.__str__())



if __name__=='__main__':
    main()