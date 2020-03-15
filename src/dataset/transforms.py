import gc
import math
from time import sleep
from typing import AnyStr, Dict, Union, List

import numpy as np
import pandas as pd
import skimage.measure
from pandas import DataFrame, Series

from src.settings import settings


class Transforms():
    csv_filename         = f"{settings['dir']['data']}/train.csv"
    csv_data             = pd.read_csv(csv_filename).set_index('image_id', drop=True).astype('category')
    csv_data['grapheme'] = csv_data['grapheme'].cat.codes.astype('category')


    @classmethod
    def transform_Y(cls, df: DataFrame, Y_field: Union[List[str],str] = None) -> Union[DataFrame,Dict[AnyStr,DataFrame]]:
        try:
            labels = df['image_id'].values
            output_df = (cls.csv_data
                            .set_index('image_id', drop=True)
                            .loc[labels])
            output_df = output_df[Y_field] if Y_field else output_df
        except KeyError as exception:
            return pd.DataFrame()  # This will be the case for the test dataset

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


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
    # noinspection PyArgumentList
    @classmethod
    def transform_X(cls, train: DataFrame, resize=2, denoise=True, normalize=True, center=True, invert=True) -> np.ndarray:
        train = (train.drop(columns='image_id', errors='ignore')
                 .values.astype('uint8')                   # unit8 for initial data processing
                 .reshape(-1, 137, 236)                    # 2D arrays for inline image processing
                 )
        gc.collect(); sleep(1)


        # Invert for processing
        # Colors   |   0 = black      | 255 = white
        # invert   |   0 = background | 255 = line
        # original | 255 = background |   0 = line
        train = (255-train)

        if denoise:
            # Rescale lines to maximum brightness, and set background values (less than 2x mean()) to 0
            train = np.array([ train[i] + (255-train[i].max())              for i in range(train.shape[0]) ])
            train = np.array([ train[i] * (train[i] >= np.mean(train[i])*2) for i in range(train.shape[0]) ])

        if isinstance(resize, bool) and resize == True:
            resize = 2
        if resize and resize != 1:                  # Reduce image size by 2x
            # NOTEBOOK: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
            # Out of the different resize functions:
            # - np.mean(dtype=uint8) produces fragmented images (needs float16 to work properly - but RAM intensive)
            # - np.median() produces the most accurate downsampling
            # - np.max() produces an image with thicker lines - occasionally produces bounding boxes
            # - np.min() produces a  image with thiner  lines (harder to read)
            resize_fn = np.median  # np.max if invert else np.min

            # BUGFIX: np.array([ for in row ]) uses less peak memory than running block_reduce() once on entire train df
            train = np.array([
                skimage.measure.block_reduce(train[i,:,:], (resize,resize), cval=0, func=resize_fn)
                for i in range(train.shape[0])
            ])

        if center:
            # NOTE: cls.crop_center_image assumes inverted
            train = np.array([
                cls.crop_center_image(train[i,:,:], cval=0)
                for i in range(train.shape[0])
            ])

        # Un-invert if invert==False
        if not invert: train = (255-train)

        if normalize:
            train = train.astype('float16') / 255.0   # prevent division cast: int -> float64


        train = train.reshape(*train.shape, 1)        # 4D ndarray for tensorflow CNN

        gc.collect(); sleep(1)
        return train


    # DOCS: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # NOTE: assumes inverted
    @classmethod
    def crop_center_image(cls, img, cval=0, tol=0):
        org_shape   = img.shape
        img_cropped = cls.crop_image(img, tol=tol)
        pad_x       = (org_shape[0] - img_cropped.shape[0])/2
        pad_y       = (org_shape[1] - img_cropped.shape[1])/2
        padding     = (
            (math.floor(pad_x), math.ceil(pad_x)),
            (math.floor(pad_y), math.ceil(pad_y))
        )
        img_center = np.pad(img_cropped, padding, 'constant', constant_values=cval)
        return img_center


    # Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # This is the fast method that simply remove all empty rows/columns
    # NOTE: assumes inverted
    @classmethod
    def crop_image(cls, img, tol=0):
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
