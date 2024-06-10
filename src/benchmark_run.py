#!/usr/bin/env python

import logging
import warnings
import time
from pathlib import Path

import click
import numpy as np
import yaml
from tqdm import tqdm
from transformers import set_seed

from evaluation import evaluate
from features import DATASET_LOADERS
from models.models import HuggingFaceLLM, HuggingFaceLLMChat
from prompt.utils import get_available_templates

# Set seed and filter warnings
set_seed(42)
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name, example_output, target_size, is_chat_model=True, **kwargs):
    max_token_multiplier = kwargs.get('max_token_multiplier', 3)
    try:
        model_cls = HuggingFaceLLMChat if is_chat_model else HuggingFaceLLM
        model = model_cls(model_name, 
                          target_size=target_size, 
                          example_output=example_output, 
                          max_token_multiplier=max_token_multiplier,
                          skip_special_tokens=kwargs.get('skip_special_tokens', True))
        logger.info(f'Model {model_name} loaded')
        return model
    except Exception as e:
        logger.error(f'Model {model_name} not found. Please check the model name and try again.')
        logger.error(e)
        return None

def save_results(results_dir, run_name, eval_result, preds):
    logger.info(f'Saving predictions to {results_dir}')
    saving_path = Path(results_dir) / f'{run_name}.npy'
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(saving_path, preds)

    saving_path = Path(results_dir) / f'{run_name}.txt'
    with open(saving_path, 'a+') as f:
        f.write(str(eval_result) + '\n')

def run_experiment(model_name, dataset_name, prompt_name, window_size, target_size, stride, batch_size=64, chunk_size=10, limit_rows=100, results_dir=None, is_chat_model=False, **kwargs):
    model_name_clean = model_name.split('/')[1] if '/' in model_name else model_name
    run_name = f'{model_name_clean}_{dataset_name}_{prompt_name}_{window_size}_{target_size}_{stride}'

    # Check if dataset_name is in DATASET_LOADERS
    if dataset_name not in DATASET_LOADERS:
        logger.error(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
        return

    if prompt_name not in get_available_templates():
        logger.error(f'Prompt {prompt_name} not found. Available prompts are: default, random, template')
        return

    logger.info(f'Running benchmark - {run_name}')
    logger.info('Loading dataset')

    dataset = DATASET_LOADERS[dataset_name]
    data_generator = dataset.process(prompt_name, window_size=window_size, target_size=target_size, batch_size=batch_size, stride=stride, chunksize=chunk_size, limit_rows=limit_rows)

    logger.info('Loading model')
    model = load_model(model_name, 
                       target_size=target_size, 
                       example_output=dataset.example_output, 
                       is_chat_model=is_chat_model, **kwargs)

    if not model:
        return

    logger.info('Running inference')
    preds, true = [], []
    timeout = time.time() + 60 * 30  # 30 minutes from now
    iters = 0
    limit_iterations = kwargs.get('limit_iterations', None)

    for observation in tqdm(data_generator):
        if limit_iterations and iters > limit_iterations:
            logger.info('Limit iterations reached. Stopping the benchmark')
            break

        prediction = model.generate(observation[0])
        preds.extend(prediction)
        true.extend(observation[1])

        iters += 1
        if time.time() > timeout and not limit_iterations:
            logger.info('Timeout reached. Stopping the benchmark')
            break

    preds = np.array(preds).reshape(-1, target_size).astype(float)
    true = np.array(true).reshape(-1, target_size).astype(float)

    if results_dir:
        save_results(results_dir, run_name, evaluate(true, preds), preds)

    logger.info('Evaluating model')
    eval_result = evaluate(true, preds)
    logger.info(eval_result)

    logger.info(f'End of benchmark run - {run_name}')
    return eval_result

@click.command()
@click.option('--config_path', type=click.STRING, required=True, help='Path to configuration file')
def main(config_path):
    config = yaml.safe_load(open(config_path, 'r'))
    config_name = Path(config_path).stem
    results_dir = Path(config.get('results_path', 'experiments')) / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for experiment in config['experiments']:
        common_params = {
            'model_name': experiment['model_name'],
            'prompt_name': experiment['prompt_name'],
            'window_size': experiment.get('window_size', 15),
            'target_size': experiment.get('target_size', 1),
            'batch_size': experiment.get('batch_size', 64),
            'chunk_size': experiment.get('chunk_size', 10),
            'is_chat_model': experiment.get('is_chat_model', True),
            'max_token_multiplier': experiment.get('max_token_multiplier', 1),
            'stride': experiment.get('stride', 1),
            'limit_rows': experiment.get('limit_rows', 100),
            'limit_iterations': experiment.get('limit_iterations', None),
            'results_dir': results_dir,
            'skip_special_tokens': experiment.get('skip_special_tokens', True),
        }

        dataset_name = experiment['dataset_name']
        #model_name_clean = common_params['model_name'].split('/')[1] if '/' in common_params['model_name'] else common_params['model_name']
        #run_name = f"{model_name_clean}_{dataset_name}_{common_params['prompt_name']}_{common_params['window_size']}_{common_params['target_size']}"

        if dataset_name == 'all':
            for dataset_name in DATASET_LOADERS.keys():
                common_params['dataset_name'] = dataset_name
                eval_result = run_experiment(**common_params)
        else:
            common_params['dataset_name'] = dataset_name
            eval_result = run_experiment(**common_params)

if __name__ == '__main__':
    main()