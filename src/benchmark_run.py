#!/usr/bin/env python

from evaluation import evaluate
from features import DATASET_LOADERS
from models.models import HuggingFaceLLM, HuggingFaceLLMChat
from prompt.utils import get_available_templates
import logging
from tqdm import tqdm
import click
import numpy as np
from pathlib import Path
import yaml
import warnings
from transformers import set_seed
import time
set_seed(42)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def load_model(model_name, example_output,target_size, is_chat_model=True, **kwargs):
    max_token_mutliplier = kwargs.get('max_token_mutliplier', 3)
    if is_chat_model:
        try:
            model = HuggingFaceLLMChat(model_name, target_size=target_size, example_output=example_output, max_token_mutliplier=max_token_mutliplier)
            logger.info(f'Chat model {model_name} loaded')
            return model
        except Exception as e:
            logger.error(f'Model {model_name} not found. Please check the model name and try again.')
            logger.error(e)
            return
    else:
        try:
            model = HuggingFaceLLM(model_name, target_size=target_size, example_output=example_output, max_token_mutliplier=max_token_mutliplier)
            logger.info(f'Model {model_name} loaded')
            return model
        except Exception as e:
            logger.error(f'Model {model_name} not found. Please check the model name and try again.')
            logger.error(e)
            return

def run_experiment(model_name, 
                   dataset_name, 
                   prompt_name, 
                   window_size, 
                   target_size, 
                   stride, 
                   batch_size=64, 
                   chunk_size=10, 
                   limit_rows=100,
                   preds_path=None,
                   is_chat_model=False, **kwargs):
    model_name_clean = model_name.split('/')[1] if '/' in model_name else model_name
    run_name = f'{model_name_clean}_{dataset_name}_{prompt_name}_{window_size}_{target_size}_{stride}'
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

    dataset = DATASET_LOADERS[dataset_name]
    data_generator = dataset.process(prompt_name, 
                                    window_size=window_size, 
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    stride=stride,
                                    chunksize=chunk_size,
                                    limit_rows=limit_rows)


    logger.info('Loading model')

        
    model = load_model(model_name, target_size=target_size, example_output=dataset.example_output, is_chat_model=is_chat_model, **kwargs)
        
        

    logger.info('Running inference')
    preds = []
    true = []

    # observation is a batch contatinign (X, y) where X has size 64 x window_size and y has size 64 x target_size



    timeout = time.time() + 60*30   # 30 minutes from now
    for observation in tqdm(data_generator):
        prediction = model.generate(observation[0])
        preds.extend(prediction)
        true.extend(observation[1])

        if time.time() > timeout:
            logger.info('Timeout reached. Stopping the benchmark')
            break


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
    

    for experiment in config['experiments']:
        model_name = experiment['model_name']
        dataset_name = experiment['dataset_name']
        prompt_name = experiment['prompt_name']
        window_size = experiment.get('window_size', 15)
        target_size = experiment.get('target_size', 1)
        batch_size = experiment.get('batch_size', 64)
        chunk_size = experiment.get('chunk_size', 10)
        is_chat_model = experiment.get('is_chat_model', True)
        max_token_mutliplier = experiment.get('max_token_mutliplier', 1)
        stride = experiment.get('stride', 1)
        limit_rows = experiment.get('limit_rows', 100)

    
        model_name_clean = model_name.split('/')[1] if '/' in model_name else model_name
        run_name = f"{model_name_clean}_{dataset_name}_{prompt_name}_{window_size}_{target_size}"
        if dataset_name == 'all':
            for dataset_name in DATASET_LOADERS.keys():
                evals = run_experiment(model_name=model_name,
                                       dataset_name=dataset_name,
                                       prompt_name=prompt_name,
                                       window_size=window_size,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       chunk_size=chunk_size,
                                       results_dir=results_dir,
                                       limit_rows=limit_rows,
                                       is_chat_model=is_chat_model, 
                                       max_token_mutliplier=max_token_mutliplier,
                                       stride=stride)
                saving_path = results_dir / (run_name + '.txt')
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                with open(saving_path, 'a+') as f:
                    f.write(evals.__str__() + '\n')

        else:
            evals = run_experiment(model_name,
                                   dataset_name,
                                   prompt_name,
                                   window_size,
                                   target_size,
                                   batch_size,
                                   chunk_size,
                                   results_dir,
                                   limit_rows=limit_rows,
                                   is_chat_model=is_chat_model,
                                   max_token_mutliplier=max_token_mutliplier)

            saving_path = results_dir / (run_name + '.txt')
            saving_path.parent.mkdir(parents=True, exist_ok=True)
            with open(saving_path, 'a+') as f:
                f.write(evals.__str__() + '\n')



if __name__=='__main__':
    main()