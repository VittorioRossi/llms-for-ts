from jinja2 import Template
import pandas as pd
import numpy as np

from pydantic import BaseModel 
from typing import List, Tuple

from prompt import load_template

class Observation(BaseModel):
    """
    Represents an observation in a time series dataset.

    Args:
        X (dict[str, np.ndarray]): The input data for the model.
        y (np.ndarray): The target data for the model.
        target_name (str, optional): The name of the target variable. Defaults to 'target'.
        metadata (dict, optional): Additional metadata associated with the observation. Defaults to {}.
    """

    X: dict[str, list]
    y: list
    target_name: str
    metadata: dict = {}


    def window_size(self):
        """
        Returns the window size of the observation.

        Returns:
            int: The window size.
        """
        return len(self.X[self.target_name])
    
    def target_size(self):
        """
        Returns the target size of the observation.

        Returns:
            int: The target size.
        """
        return len(self.y)
    
    def ts_features(self):
        """
        Returns the list of time series features in the observation.

        Returns:
            list: The list of time series features.
        """
        return list(self.X[0].keys())

    def metadatas(self):
        """
        Returns the keys of the metadata(i.e. the vars that are constant across the observation)

        Returns:
            list: The keys of the metadata.
        """
        return self.metadata.keys()

    def as_tuple(self):
        """
        Returns the observation as a tuple.

        Returns:
            Tuple: The observation as a tuple.
        """
        return (self.X, self.y)

    def render(self, prompt:Template) -> Tuple[str, np.ndarray]:
        """
        Render the observation using the prompt.

        Args:
            prompt (Template): The prompt to use to render the observation.

        Returns:
            Tuple[str, np.ndarray]: The rendered observation.
        """
        return prompt.render(data=self.X, metadata=self.metadata)

def render_series(series: list, prompt_name, **kwargs) -> str:
    """
    Render a list of time series values using the prompt.

    Args:
        series (list): The list of time series values.
        prompt (Template): The prompt to use to render the time series values.

    Returns:
        str: The rendered time series values.
    """
    prompt: Template = load_template(prompt_name)

    return prompt.render(data={'target':series}, **kwargs)


def _create_observations_w_ft_and_meta(df: pd.DataFrame, prompt:Template, target: str, ts_features: List[str], metadata: List[str], window_size: int = 24, target_size: int = 1, stride=1) -> List[Observation]:
    """
    Create an observation object with time series features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series features and target variable.
        target (str): The name of the target variable column in the DataFrame.
        ts_features (List[str]): A list of column names for the time series features in the DataFrame.
        metadata (List[str]): A list of column names for the metadata in the DataFrame.
        window_size (int, optional): The size of the sliding window used to create the observation. Defaults to 24.
        target_size (int, optional): The number of future time steps to predict. Defaults to 1.

    Returns:
        List[Observation]: A list of observation objects containing the input features (X) and target variable (y).
    """
    # Remove leading and trailing NaNs from the target and adjust the dataset accordingly

    # Define the range for slicing
    slice_end = len(df) - window_size - target_size

    # Extend the time series features to include the target column
    ts_features = ts_features + [target]
    
    # Pre-compute metadata values
    mt = {meta: df[meta].iloc[0] for meta in metadata}
    
    # Create the list of observations
    X_fin, y_fin = [], []
    for i in range(0, slice_end, stride):  # Adjust the range to use the stride
        y = df[target].iloc[i + window_size: i + window_size + target_size].values.flatten()
        if np.isnan(y).any():
            continue

        X = {ft_name: df[ft_name].iloc[i:i + window_size].values for ft_name in ts_features}
        X_fin.append(prompt.render(data=X, metadata=mt, target_size=target_size))
        y_fin.append(list(np.array(y, dtype=float).flatten()))
    
    return X_fin, y_fin

def process_dataset(dataset: pd.DataFrame,
                    prompt_name: str, 
                    window_size:int, 
                    target_size: int,
                    target:str,
                    ts_features: List[str] = [],
                    metadata: List[str] = [],
                    stride:int = 1,
                    *args,
                    **kwargs) -> List[Tuple[str, np.ndarray]]:
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

    X, y = [], []
    if len(metadata) != 0:
        assert isinstance(metadata, list), "metadata must be a list of strings"
        assert all([meta in dataset.columns for meta in metadata]), "metadata must be a list of columns in the dataset"
        #the metadata specifies the object 
        for group in dataset.groupby(metadata):
            X_group, y_group = _create_observations_w_ft_and_meta(group[1], 
                                                                prompt,
                                                                target, 
                                                                ts_features=ts_features, 
                                                                metadata=metadata, 
                                                                window_size=window_size, 
                                                                target_size=target_size,
                                                                stride = stride)
            X.extend(X_group)
            y.extend(y_group)
    else:
        X, y = _create_observations_w_ft_and_meta(dataset, 
                                                     prompt,
                                                     target, 
                                                    ts_features=ts_features, 
                                                    metadata=[], 
                                                    window_size=window_size, 
                                                    target_size=target_size,
                                                    stride = stride)
    
    return X, y

def create_batches(X, y, batch_size):
    """Yield successive batches from a list."""
    if isinstance(X, list):
        for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size]
    else:
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]
