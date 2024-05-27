from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from features.utils import process_dataset

class Dataset(ABC):
    @abstractmethod
    def process(self):
        pass

class CTDataset(Dataset):
    def __init__(self, path:str = '/data/raw/CT'):
        self.path = path

    def process(self, **kwargs):
        X = open(self.path + '/minimal/val_x_prompt.txt', 'r').read().splitlines('\n')
        y = open(self.path + '/minimal/val_y_prompt.txt', 'r').read().splitlines('\n')
        
        for obs in [(xi, yi) for xi, yi in zip(X, y)]:
            yield obs


class SGFDataset(Dataset):
    def __init__(self, path:str = '/data/raw/SG'):
        self.path = path

    def process(self, **kwargs):
        X = open(self.path + '/minimal/val_x_prompt.txt', 'r').read().splitlines('\n')
        y = open(self.path + '/minimal/val_y_prompt.txt', 'r').read().splitlines('\n')
        
        for obs in [(xi, yi) for xi, yi in zip(X, y)]:
            yield obs

class ETTHDataset(Dataset):
    """
    ETTHDataset is a dataset class that takes in a path to the ETTH dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/etth'):
        self.path = path
    
    def process(self, promt_name:str, **kwargs):
        df = pd.read_csv(self.path)
        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', []),
            'metadata': kwargs.get('metadata', []),
        }
        df = df.rename(columns={'OT': 'target'})
        generator = process_dataset(df, promt_name, **config)
        for observation in generator:
            yield observation

class M4Dataset(Dataset):
    """
    M4Dataset is a dataset class that takes in a path to the M4 dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/m4', train = True, **kwargs):
        self.path = path
        self.df_path = f'{self.path}/' + ('train.csv' if train else 'test.csv')

    
    def process(self, promt_name:str, chunksize = 1000, **kwargs):
        chunks = pd.read_csv(self.df_path, chunksize=chunksize)

        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': [],
            'metadata': ["V1"],
        }

        for chunk in chunks:
            chunk = chunk.melt(id_vars=['V1'], var_name='d', value_name='target')
            generator = process_dataset(chunk, promt_name, **config)
            for observation in generator:
                yield observation
    


class M5Dataset(Dataset):
    """
    M5Dataset is a dataset class that takes in a path to the M5 dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/m5', train = True):
        self.path = path
    
        self.calendar = pd.read_csv(f'{self.path}/calendar.csv').astype({'d': 'str'})
        self.prices = pd.read_csv(f'{self.path}/sell_prices.csv')
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
    

    def process(self, promt_name:str, chunksize = 100, **kwargs):
        chunks = pd.read_csv(self.df_path, chunksize=chunksize)

        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', ['d']),
            'metadata':  kwargs.get('metadata', ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']),
        }

        metadata = config.get('metadata')
        for chunk in chunks:
            chunk = chunk.melt(id_vars=metadata, var_name='d', value_name='target')
            chunk = self._merge_metadata(chunk)
            generator = process_dataset(chunk, promt_name, **config)
            for observation in generator:
                yield observation

class GWTDataset(Dataset):
    """
    GWTDataset is a dataset class that takes in a path to the GWT dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/gwt'):
        self.path = path
        self.df_path = f'{self.path}/train.csv'
    
    def process(self, promt_name:str, chunksize = 1000, **kwargs):
        chunks = pd.read_csv(self.df_path, chunksize=chunksize)
        config = {
            'target': 'target',
            'target_size': kwargs.get('target_size', 1),
            'window_size': kwargs.get('window_size', 24),
            'ts_features': kwargs.get('ts_features', ['d']),
            'metadata':  kwargs.get('metadata', ["Page"]),
        }
        for chunk in chunks:
            chunk = chunk.melt(id_vars=['Page'], var_name='d', value_name='target')
            generator = process_dataset(chunk, promt_name, **config)
            for observation in generator:
                yield observation

class PEMSDataset(Dataset):
    def __init__(self, path='data/raw/PEMS4') -> None:
        self.path = path
        self.distances = pd.read_csv(f'{self.path}/distances.csv')
    
    def process(self, promt_name:str, chunksize = 1000, **kwargs):
        data = np.load(f'{self.path}/PEMS04.npz')['X']
        ...