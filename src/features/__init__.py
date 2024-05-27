from .datasets import *

DATASET_LOADERS = {
    'CT': CTDataset(),
    'SG': SGFDataset(),
    'ETTh1': ETTHDataset(path = '/data/raw/etth/ETTh1.csv'),
    'ETTh2': ETTHDataset(path = '/data/raw/etth/ETTh2.csv'),
    'ETTm1': ETTHDataset(path = '/data/raw/etth/ETTm1.csv'),
    'ETTm2': ETTHDataset(path = '/data/raw/etth/ETTm2.csv'),
    'M4-month': M4Dataset(path='data/raw/m4-monthly'),
    'M4-week': M4Dataset(path='data/raw/m4-weekly'),
    'M4-quarter': M4Dataset("data/raw/m4-quarterly"),
    'M5': M5Dataset(path='data/raw/m5'),
    'GWT': GWTDataset(path='data/raw/gwt'),
}
