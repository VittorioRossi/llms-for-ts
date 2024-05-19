"""
In this script we are going to create the function that takes in a prompt and is able 
to transform a raw dataset into a processed one ready to be passed to the model.

Since we have very different datasets we have to handle evey dataset in a different way.
"""
from jinja2 import Template
import pandas as pd
import numpy as np

from src.prompt.utils import load_template

def _create_observation(series:np.ndarray, window_size:int = 24, target_size:np.ndarray = 1):
    X = np.array([series.iloc[i:i + window_size].values for i in range(len(series) - window_size - target_size)])
    y = np.array([series.iloc[i + window_size: i+window_size+target_size].values for i in range(len(series) - window_size - target_size)])
    return X, y.flatten()

def _create_observation_row_wise(df: pd.DataFrame, window_size: int = 24, target_size: int = 1):
    X, y = [], []
    for _, row in df.iterrows():
        _x, _y = _create_observation(row, window_size, target_size)
        X.append(_x)
        y.append(_y)
    X = np.array(X)
    y = np.array(y)
    return X, y
        

def process_dataset(dataset: pd.DataFrame,
                    prompt_name: str, 
                    window_size:int, 
                    target_size: int,
                    row_wise:bool = False,
                    target:str = ''
                    ) -> pd.DataFrame:
    """
    This function takes in a dataset, a prompt name, window size, target size, and optional parameters
    and returns a processed dataset ready to be passed to the model.
    
    Args:
    dataset (pd.DataFrame): The dataset to be processed.
    prompt_name (str): The name of the prompt to be used to process the dataset.
    window_size (int): The size of the sliding window used to create observations.
    target_size (int): The size of the target window.
    row_wise (bool, optional): If True, create observations row-wise. Defaults to False.
    target (str, optional): The name of the target column. Defaults to ''.
    
    Returns:
    pd.DataFrame: The processed dataset.
    """
    # load the prompt
    prompt: Template = load_template(prompt_name)

    # transform the dataset into many X, y pairs.
    if not row_wise and (target not in dataset.columns):
        raise ValueError(f"The target column {target} is not in the dataset")

    X, y = None, None

    if row_wise:
        X, y = _create_observation_row_wise(dataset, window_size, target_size)
    else:
        X, y = _create_observation(dataset[target], window_size, target_size)

    
    # process the dataset
    processed_X = [prompt.render(data=x) for x in X]
    

    return pd.DataFrame({"X": processed_X, "y": y})