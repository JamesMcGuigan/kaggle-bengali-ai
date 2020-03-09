import gc
from typing import List, Callable, AnyStr, Dict, Tuple

import glob2
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


class DatasetDF():
    def __init__(self,
                 split: float=0.2,
                 apply_X: List[Callable]=[],
                 apply_Y: List[Callable]=[],
                 data_dir='./input/bengaliai-cv19',
                 image_shape=(137,236),
                 data_id=0
    ):
        self.image_shape: Tuple = image_shape

        self.files: Dict[AnyStr, List[AnyStr]] = {
            "train": sorted(glob2.glob(f'{data_dir}/train_image_data_{data_id}.parquet')),
            "test":  sorted(glob2.glob(f'{data_dir}/test_image_data_{data_id}.parquet')),
        }
        self.raw: Dict[AnyStr, DataFrame] = {
            key: pd.concat([ pd.read_parquet(file) for file in self.files[key] ])
            for key in self.files
        }
        self.raw['train'], self.raw['valid'] = train_test_split(self.raw['train'], test_size=split)

        self.X: Dict[AnyStr, Series] = {}
        self.Y: Dict[AnyStr, Series] = {}
        for key in reversed(['train','valid','test']):
            self.X[key] = (
                self.raw[key]
                    .drop(columns='image_id', errors='ignore')
                    .values.astype('uint8')
                    .reshape(-1, *self.image_shape, 1)
            )
            self.Y[key] = self.raw[key]['image_id'] if 'image_id' in self.raw[key].columns else None
            del self.raw[key]; gc.collect()  # Free up memory

        for function in apply_X:
            for key in self.X:
                self.X[key] = self.X[key].apply(function)

        for function in apply_Y:
            for key in self.Y:
                self.X[key] = self.X[key].apply(function)


if __name__ == '__main__':
    # NOTE: loading all datasets exceeds 12GB RAM and crashes Python (on 16GB RAM machine)
    for data_id in range(0,4):
        dataset = DatasetDF(data_id='*')
        print(f"{data_id} dataset.files", dataset.files)
        print(f"{data_id} dataset.raw",   { key: df.shape for key, df in dataset.raw.items() })
        print(f"{data_id} dataset.X",     { key: df.shape for key, df in dataset.X.items()   })
        print(f"{data_id} dataset.Y",     { key: df.shape for key, df in dataset.Y.items()   })
        print(f"{data_id} dataset.image_shape", dataset.image_shape)