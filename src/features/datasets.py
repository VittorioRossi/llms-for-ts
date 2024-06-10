from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import ast

from . import utils
from pathlib import Path


#### CACHING ####

def build_cache_path(cache_folder, window_size, target_size, stride, prompt_name='', **kwargs):
    return Path(cache_folder) / f'{prompt_name}_{window_size}_{target_size}_{stride}.csv'

def cache_dataset(X, y, cache_path, **kwargs):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'a') as f:  # Use 'a' to append to the file
        for xi, yi in zip(X,y):
            f.write(f'{repr(xi)};{repr(yi)}\n')  # Write yi as a quoted string

def load_cached_data(cache_path, batch_size=64):
    with open(cache_path, 'r') as f:
        data = f.readlines()

    X, y = zip(*[line.strip().split(';') for line in data])
    X = list(X)

    y = [ast.literal_eval(item) for item in y]
    y = np.array(y).astype(float)

    return utils.create_batches(X, y, batch_size)

#### LOADING ####

def load_dataset_file(dataset_name, **kwargs):
    ...


class Dataset(ABC):
    @abstractmethod
    def process(self, prompt_name):
        pass

class CTDataset(Dataset):
    def __init__(self, path:str = 'data/raw/CT', cache_folder = 'data/processed/CT'):
        self.path = path
        self.cache_folder = cache_folder
        self.example_output = "000"
        self.seasonalities = []

    def process(self, prompt_name:str, batch_size:int, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            return load_cached_data(cache_path, batch_size=batch_size)

        X = open(self.path + '/minimal/val_x_prompt.txt', 'r').read().splitlines()
        y = open(self.path + '/minimal/val_y_prompt.txt', 'r').read().splitlines()
        X = [utils.render_series(x.replace(',', '').split(), prompt_name, **kwargs) for x in X]
        cache_dataset(X, y, cache_path, **kwargs)
        batches = utils.create_batches(X, y, batch_size)
        return batches # this is a generator

class SGDataset(Dataset):
    def __init__(self, path:str = 'data/raw/SG', cache_folder = 'data/processed/SG'):
        self.path = path
        self.cache_folder = cache_folder
        self.example_output = "000"
        self.seasonalities = [7]

    def process(self, prompt_name, batch_size, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            return load_cached_data(cache_path, batch_size=batch_size)

        X = open(self.path + '/minimal/val_x_prompt.txt', 'r').read().splitlines()
        y = open(self.path + '/minimal/val_y_prompt.txt', 'r').read().splitlines()
        X = [utils.render_series(x.replace(',', '').split(), prompt_name, **kwargs) for x in X]
        cache_dataset(X, y, cache_path, prompt_name=prompt_name, **kwargs)
        batches = utils.create_batches(X, y, batch_size)
        return batches # this is a generator
    

class ETTHDataset(Dataset):
    """
    ETTHDataset is a dataset class that takes in a path to the ETTH dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/ETTh1', cache_folder = 'data/processed/ETTh1'):
        self.path = path
        self.cache_folder = cache_folder
        self.example_output = "00"
        self.seasonalities = [24, 168] #daily and weekly
    
    def process(self, prompt_name:str,batch_size:int, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            return load_cached_data(cache_path, batch_size=batch_size)
        
        df = pd.read_csv(self.path).round(kwargs.get('round', 2))

        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', []),
            'metadata': kwargs.get('metadata', []),
        }
        df = df.rename(columns={'OT': 'target'})
        X, y = utils.process_dataset(df, prompt_name, **config)
        cache_dataset(X, y, cache_path, prompt_name=prompt_name, **config)
        return utils.create_batches(X, y, batch_size)

class M4Dataset(Dataset):
    """
    M4Dataset is a dataset class that takes in a path to the M4 dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/m4', cache_folder = 'data/processed/m4', name = 'weekly', train = True, **kwargs):
        if not name in ['month', 'week', 'quarter']:
            raise ValueError("Name must be one of 'month', 'quarter', 'week'")

        self.path = path
        self.df_path = f'{self.path}/' + ('train.csv' if train else 'test.csv')
        self.cache_folder = cache_folder
        self.example_output = "00000.00"
        self.seasonalities = self._compute_seasonality(name)

    def _compute_seasonality(self, name):
        if name == 'week':
            return [52]
        elif name == 'quarter':
            return [4]
        elif name == 'month':
            return [12]
    
    def process(self, prompt_name:str, batch_size, chunksize = 1000, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            for batch in load_cached_data(cache_path, batch_size=batch_size):
                yield batch
            return 

        limit_rows = kwargs.get('limit_rows', None)
        if limit_rows:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize, nrows=limit_rows)
        else:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize)

        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': [],
            'metadata': ["V1"],
            'stride': kwargs.get('stride', 1)
        }

        for chunk in chunks:
            chunk = chunk.melt(id_vars=['V1'], var_name='d', value_name='target')
            chunk['target'] = chunk['target'].round(2)
            X,y = utils.process_dataset(chunk, prompt_name, **config)
            cache_dataset(X, y, cache_path,prompt_name=prompt_name, **config)
            for batch in utils.create_batches(X, y, batch_size):
                yield batch

class M5Dataset(Dataset):
    """
    M5Dataset is a dataset class that takes in a path to the M5 dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/m5', train = True, cache_folder = 'data/processed/m5', **kwargs):
        self.path = path
        self.cache_folder = cache_folder
        self.example_output = "000"
        self.seasonalities = [7, 28]
    
        self.df_path = f'{self.path}/' + ('sales_train_validation.csv' if train else 'sales_train_evaluation.csv')

        self.feature_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.ts_vars = ["date",
                        "weekday",
                        "wday",
                        "month",
                        "year",
                        "d",
                        "event_name_1",
                        "event_type_1",
                        "event_name_2",
                        "event_type_2",
                        "snap_CA",
                        "snap_TX",
                        "snap_WI",
                        "price"]

        
    
    def _merge_metadata(self, df):
        df = df.merge(self.calendar, how='left', on='d')
        df = df.merge(self.prices, how='left', on="wm_yr_wk")
        return df
    
    def process(self, prompt_name:str, batch_size, chunksize=100, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            for batch in load_cached_data(cache_path, batch_size=batch_size):
                yield batch
            return 

        limit_rows = kwargs.get('limit_rows', None)
        if limit_rows:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize, nrows=limit_rows)
        else:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize)
        
        merge_data = kwargs.get('merge', False)
        if merge_data:
            self.calendar = pd.read_csv(f'{self.path}/calendar.csv').astype({'d': 'str'})
            self.prices = pd.read_csv(f'{self.path}/sell_prices.csv')


        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', ['d']),
            'metadata':  kwargs.get('metadata', ["id", 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']),
        }

        metadata = config.get('metadata')
        for chunk in chunks:
            chunk = chunk.melt(id_vars=metadata, var_name='d', value_name='target')
            chunk = chunk.astype({'target':"float"})
            if merge_data:
                chunk = self._merge_metadata(chunk)
            
            X, y = utils.process_dataset(chunk, prompt_name, **config)
            cache_dataset(X, y, cache_path,prompt_name=prompt_name, **config)

            for batch in  utils.create_batches(X, y, batch_size):
                yield batch

class GWTDataset(Dataset):
    """
    GWTDataset is a dataset class that takes in a path to the GWT dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/gwt', cache_folder = 'data/processed/gwt'):
        self.path = path
        self.df_path = f'{self.path}/train.csv'
        self.cache_folder = cache_folder
        self.example_output = "0000"
        self.seasonalities = [7]
    
    def process(self, prompt_name:str, batch_size, chunksize = 1000, **kwargs):
        cache_path = build_cache_path(self.cache_folder, prompt_name=prompt_name, **kwargs)
        if self.cache_folder and cache_path.exists():
            for batch in load_cached_data(cache_path, batch_size=batch_size):
                yield batch
            return 
        
        limit_rows = kwargs.get('limit_rows', None)
        if limit_rows:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize, nrows=limit_rows)
        else:
            chunks = pd.read_csv(self.df_path, chunksize=chunksize)

        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', ['d']),
            'metadata':  kwargs.get('metadata', ["Page"]),
        }
        for chunk in chunks:
            chunk = chunk.melt(id_vars=['Page'], var_name='d', value_name='target').astype({'d': 'str', 'target':"float"})
            X, y = utils.process_dataset(chunk, prompt_name, **config)
            cache_dataset(X, y, cache_path,prompt_name=prompt_name, **config)
            for batch in utils.create_batches(X,y, batch_size):
                yield batch