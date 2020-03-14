import math
import os
from time import sleep
from typing import AnyStr, Dict, Union

import gc
import glob2
import numpy as np
import pandas as pd
import skimage.measure
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
            del raw; gc.collect()

        self.Y = { key: self.transform_Y(value) for key,value in self.Y.items() }
        pass


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


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/edit/run/29865909
    # noinspection PyArgumentList
    @classmethod
    def transform_X(cls, train: DataFrame, resize=2, denoise=True, normalize=True, center=True, invert=True) -> np.ndarray:
        train = (train.drop(columns='image_id', errors='ignore')
                 .values.astype('uint8')                   # unit8 for initial data processing
                 .reshape(-1, 137, 236)                    # 2D arrays for inline image processing
        )
        gc.collect(); sleep(1)

        if invert:                                         # Colors | 0 = black      | 255 = white
            train = (255-train)                            # invert | 0 = background | 255 = line

        if denoise:                                        # Set small pixel values to background 0
            if invert: train *= (train >= 25)              #   0 = background | 255 = line  | np.mean() == 12
            else:      train += (255-train)*(train >= 230) # 255 = background |   0 = line  | np.mean() == 244

        if isinstance(resize, bool) and resize == True:
            resize = 2
        if resize and resize != 1:                  # Reduce image size by 2x
            # NOTEBOOK: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
            # Out of the different resize functions:
            # - np.mean(dtype=uint8) produces fragmented images (needs float16 to work properly - but RAM intensive)
            # - np.median() produces the most accurate downsampling
            # - np.max() produces an enhanced image with thicker lines (maybe slightly easier to read)
            # - np.min() produces a  dehanced image with thiner lines (harder to read)
            resize_fn = np.max if invert else np.min
            cval      = 0      if invert else 255
            train     = skimage.measure.block_reduce(train, (1, resize,resize), cval=cval, func=resize_fn)

        if center:
            # NOTE: cls.crop_center_image assumes inverted
            if not invert: train = (255-train)
            train = np.array([
                cls.crop_center_image(train[i,:,:])
                for i in range(train.shape[0])
            ])
            if not invert: train = (255-train)

        if normalize:
            train = train.astype('float16') / 255.0   # prevent division cast: int -> float64

        train = train.reshape(*train.shape, 1)        # 4D ndarray for tensorflow CNN

        gc.collect(); sleep(1)
        return train


    # DOCS: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # NOTE: assumes inverted
    @classmethod
    def crop_center_image(cls, img, tol=0):
        org_shape   = img.shape
        img_cropped = cls.crop_image(img)
        pad_x       = (org_shape[0] - img_cropped.shape[0])/2
        pad_y       = (org_shape[1] - img_cropped.shape[1])/2
        padding     = (
            (math.floor(pad_x), math.ceil(pad_x)),
            (math.floor(pad_y), math.ceil(pad_y))
        )
        img_center = np.pad(img_cropped, padding, 'constant', constant_values=0)
        return img_center


    # Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # This is the fast method that simply remove all empty rows/columns
    # NOTE: assumes inverted
    @classmethod
    def crop_image(cls, img, tol=0):
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]


    def epoch_size(self):
        return self.X['train'].shape[0]


    def input_shape(self):
        return self.X['train'].shape[1:]  # == (137, 236, 1) / 2


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
