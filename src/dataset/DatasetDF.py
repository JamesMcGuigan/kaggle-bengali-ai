import gc
import os
from typing import AnyStr, Dict, Union

import glob2
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from src.settings import settings


class DatasetDF():
    csv_filename         = f"{settings['dir']['data']}/train.csv"
    csv_data             = pd.read_csv(csv_filename).set_index('image_id', drop=True).astype('category')
    csv_data['grapheme'] = csv_data['grapheme'].cat.codes.astype('category')

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

        self.image_filenames = sorted(glob2.glob(f"{settings['dir']['data']}/{test_train}_image_data_{data_id}.parquet"))

        self.X:  Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        self.Y:  Dict[AnyStr, Union[pd.DataFrame,Dict]] = { "train": pd.DataFrame(),   "valid": pd.DataFrame()   }
        self.ID: Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        for filename in self.image_filenames:
            raw = {
                'train': pd.read_parquet(filename),
                'valid': None
            }
            if self.fraction < 1:
                raw['train'], discard      = train_test_split(raw['train'], train_size=self.fraction, shuffle=self.shuffle)
                del discard
            if self.split != 0:
                raw['train'], raw['valid'] = train_test_split(raw['train'], test_size=self.split,     shuffle=self.shuffle, random_state=0)
            if raw['valid'] is None:
                raw['valid'] = pd.DataFrame(columns=raw['train'].columns)

            # Attempt to save memory by doing transform_X() within the loop
            # X can be transformed before np.concatenate, but multi-output Y must be done after pd.concat()
            for key, value in raw.items():
                X = self.transform_X(value)
                if len(self.X[key]) == 0: self.X[key] = X
                else:                     self.X[key] = np.concatenate([ self.X[key],  self.transform_X(value)  ])
                self.Y[key]  = pd.concat([      self.Y[key],  value[['image_id']]      ])
                self.ID[key] = np.concatenate([ self.ID[key], value['image_id'].values ])
            del X, raw; gc.collect()

        self.Y = { key: self.transform_Y(value) for key,value in self.Y.items() }
        pass


    # noinspection PyArgumentList
    def transform_X(self, df: DataFrame) -> np.ndarray:
        output = (
            df.drop(columns='image_id', errors='ignore')
              .values.astype('float16')
              .reshape(-1, 137, 236, 1)
              / 255.0                    # normalization caused localhost 16Gb RAM to be exceeded without float16
        )
        return output


    def transform_Y(self, df: DataFrame) -> Union[DataFrame,Dict[AnyStr,DataFrame]]:
        if self.test_train == 'test': return pd.DataFrame()

        labels = df['image_id'].values
        output_df = self.csv_data.drop(columns='image_id', errors='ignore').loc[labels]
        output_df = output_df[self.Y_field] if self.Y_field else output_df

        if isinstance(output_df, Series) or len(output_df.columns) == 1:
            # single model output
            output = pd.get_dummies( output_df )
        else:
            # multi model output
            output = {
                column: pd.get_dummies( output_df[column] )
                for column in output_df.columns
            }
        return output


    def epoch_size(self):
        return self.X['train'].shape[0]


    def input_shape(self):
        return self.X['train'].shape[1:]  # == (137, 236, 1) / 2


    # def output_shape(self):
    #     return self.__class__.output_shape(self.Y_field)

    @classmethod
    def output_shape(cls, Y_field=None):
        if isinstance(Y_field, str):
            return cls.csv_data[Y_field].nunique()

        csv_data     = cls.csv_data[Y_field] if Y_field else cls.csv_data
        output_shape = (csv_data.drop(columns='image_id', errors='ignore')
                                .nunique()
                                .to_dict())
        return output_shape




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
