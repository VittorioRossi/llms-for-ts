#!/usr/bin/env python

from evaluation import evaluate
from features import DATASET_LOADERS
from models.models import HuggingFaceLLM
from prompt.utils import get_available_templates
import logging
from tqdm import tqdm
import click

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model_name', type=click.STRING, required=True, help='Model identifier on huggingface.co')
@click.option('--dataset_name', type=click.STRING, required=True, help='Name of the dataset')
@click.option('--prompt_name', type=click.STRING, required=True, help='Name of the prompt setting')
@click.option('--window_size', type=click.INT, help='Window size for the input features', default=24)
@click.option('--target_size', type=click.INT, help='Target size', default=1)
@click.option('--batch_size', type=click.INT, help='Batch size', default=64)
@click.option('--chunk_size', type=click.INT, help='Chunk_size', default=10)
def main(model_name, dataset_name, prompt_name, window_size, target_size, batch_size=64, chunk_size=10):

    # check if dataset_name is in DATASET_LOADERS
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f'Dataset {dataset_name} not found. Available datasets are: {list(DATASET_LOADERS.keys())}')
    
    if prompt_name not in get_available_templates():
        raise ValueError(f'Prompt {prompt_name} not found. Available prompts are: default, random, template')

    logger.info('Running benchmark')

    logger.info('Loading dataset')

    data_generator = DATASET_LOADERS[dataset_name].process(prompt_name, 
                                                            window_size=window_size, 
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            chunk_size=chunk_size)

    logger.info('Loading model')
    try:
        model = HuggingFaceLLM(model_name)
    except Exception as e:
        raise ValueError(f'Model {model_name} not found. Please check the model name and try again.')

    logger.info('Running inference')
    preds = []
    true = []
    for observation in tqdm(dataset):
        preds.append(model.generate(observation[0]))
        true.append(observation[1])

    logger.info('Evaluating model')
    eval = evaluate(true, preds)
    logger.info(eval)


if __name__=='__main__':
    main()