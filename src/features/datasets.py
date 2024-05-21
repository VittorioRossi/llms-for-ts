from abc import ABC, abstractmethod
import pandas as pd

from src.features.utils import process_dataset


class Dataset(ABC):
    @abstractmethod
    def process(self):
        pass



class M4Dataset(Dataset):
    """
    M4Dataset is a dataset class that takes in a path to the M4 dataset and processes it.
    """
    def __init__(self, path:str = '/data/raw/m4', train = True):
        self.path = path
        self.df_path = f'{self.path}/' + ('train.csv' if train else 'test.csv')
    
    def process(self, promt_name:str, chunksize = 1000, **kwargs):
        chunks = pd.read_csv(self.df_path, chunksize=chunksize)
        for chunk in chunks:
            yield process_dataset(chunk, promt_name, **kwargs)
    

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
        self.id_vars = ['store_id', 'item_id', 'wm_yr_wk']
        
    def _merge_metadata(self, df):
        df = df.merge(self.calendar, how='left', on='d')
        df = df.merge(self.prices, how='left', on=self.id_vars)
        return df
    

    def process(self, promt_name:str, chunksize = 1000, metadata = None, **kwargs):
        chunks = pd.read_csv(self.df_path, chunksize=chunksize)
        
        if metadata is not None and isinstance(metadata, list):
            for chunk in chunks:
                chunk = chunk.drop(columns=self.feature_cols, axis=1)
                yield process_dataset(chunk, promt_name, row_wise=True, **kwargs)
        else:
            for chunk in chunks:
                chunk = chunk.melt(id_vars=self.feature_cols, var_name='d', value_name='sales')
                print(chunk.columns)
                chunk = self._merge_metadata(chunk)
                yield process_dataset(chunk, promt_name, target='sales', **kwargs)