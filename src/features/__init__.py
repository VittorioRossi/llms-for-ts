from .datasets import *
import os

cache_folder = os.environ.get('CACHE_FOLDER','data/processed/')
DATASET_LOADERS = {
    'CT': CTDataset(cache_folder=cache_folder + 'CT'),
    'SG': SGDataset(cache_folder=cache_folder + 'SG'),
    'ETTh1': ETTHDataset(path = 'data/raw/ETTh1/train.csv', cache_folder=cache_folder + 'ETTh1'),
    'ETTh2': ETTHDataset(path = 'data/raw/ETTh2/train.csv', cache_folder=cache_folder + 'ETTh2'),
    'ETTm1': ETTHDataset(path = 'data/raw/ETTm1/train.csv', cache_folder=cache_folder + 'ETTm1'),
    'ETTm2': ETTHDataset(path = 'data/raw/ETTm2/train.csv', cache_folder=cache_folder + 'ETTm2'),
    'M4-month': M4Dataset(path='data/raw/m4-monthly', cache_folder=cache_folder + 'm4-monthly', name='month'),
    'M4-week': M4Dataset(path='data/raw/m4-weekly', cache_folder=cache_folder + 'm4-weekly',name='week'),
    'M4-quarter': M4Dataset("data/raw/m4-quarterly", cache_folder=cache_folder + 'm4-quarterly', name='quarter'),
    'M5': M5Dataset(path='data/raw/m5', cache_folder=cache_folder + 'm5'),
    'GWT': GWTDataset(path='data/raw/gwt', cache_folder=cache_folder + 'gwt'),
}