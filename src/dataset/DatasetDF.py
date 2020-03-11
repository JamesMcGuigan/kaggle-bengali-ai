import os
from typing import AnyStr, Dict, List, Union

import glob2
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.settings import settings


class DatasetDF():
    def __init__(self,
                 test_train   = 'train',
                 data_id: Union[str,int] = '0',
                 fraction     = 1,
                 Y_field      = None,
                 shuffle      = True,
                 split: float = 0.1,
        ):
        self.test_train = test_train
        self.data_id    = data_id
        self.Y_field    = Y_field
        self.split      = split    if self.test_train is 'train' else 0
        self.shuffle    = shuffle  if self.test_train is 'train' else False
        self.fraction   = fraction if self.test_train is 'train' else 1

        self.csv_filename         = f"{settings['dir']['data']}/train.csv"
        self.csv_data             = pd.read_csv(self.csv_filename).set_index('image_id', drop=True).astype('category')
        self.csv_data['grapheme'] = self.csv_data['grapheme'].cat.codes.astype('category')

        self.image_filenames = sorted(glob2.glob(f"{settings['dir']['data']}/{test_train}_image_data_{data_id}.parquet"))

        X = { "train": [], "valid": [] }
        Y = { "train": [], "valid": [] }
        for filename in self.image_filenames:
            train, valid = pd.read_parquet(filename), pd.DataFrame()
            if self.fraction < 1:
                train, discard = train_test_split(train, train_size=self.fraction, shuffle=self.shuffle, random_state=0)
            if self.split != 0:
                train, valid   = train_test_split(train, test_size=self.split,     shuffle=self.shuffle, random_state=0)

            X['train'].append( self.transform_X(train) )
            X['valid'].append( self.transform_X(valid) )
            Y['train'].append( self.transform_Y(train) )
            Y['valid'].append( self.transform_Y(valid) )

        self.X: Dict[AnyStr, np.ndarray] = { key: np.concatenate(X[key]) for key in X.keys() }
        self.Y: Dict[AnyStr, np.ndarray] = { key: np.concatenate(Y[key]) for key in Y.keys() }


    # noinspection PyArgumentList
    def transform_X(self, df: DataFrame) -> np.ndarray:
        output = (
            df.drop(columns='image_id', errors='ignore')
              .values.astype('uint8')
              .reshape(-1, 137, 236, 1)
        )
        return output


    def transform_Y(self, df: DataFrame) -> Union[DataFrame,List]:
        if self.test_train == 'test': return []

        labels = df['image_id'].values
        output = self.csv_data.loc[labels]
        if self.Y_field:
            output = output[self.Y_field]
        output = pd.get_dummies( output )  # `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape
        return output


    def input_shape(self):
        return self.X['train'].shape[1:]

    def output_shape(self):
        return self.Y['train'].shape[-1]

    def epoch_size(self):
        return self.X['train'].shape[0]


if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    # NOTE: loading all datasets exceeds 12GB RAM and crashes Python (on 16GB RAM machine)
    for data_id in range(0,4):
        for test_train in ['test', 'train']:
            dataset = DatasetDF(test_train=test_train, data_id=data_id, fraction=1)
            print(f"{test_train}:{data_id} dataset.image_filenames", dataset.image_filenames)
            print(f"{test_train}:{data_id} dataset.X",               { key: df.shape for key, df in dataset.X.items() })
            print(f"{test_train}:{data_id} dataset.Y",               { key: df.shape for key, df in dataset.Y.items() })
            print(f"{test_train}:{data_id} dataset.input_shape()",   dataset.input_shape())
            print(f"{test_train}:{data_id} dataset.output_shape()",  dataset.output_shape())
            print(f"{test_train}:{data_id} dataset.epoch_size()",    dataset.epoch_size())
