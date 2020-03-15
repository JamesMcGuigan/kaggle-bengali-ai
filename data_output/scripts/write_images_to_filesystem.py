#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/preprocessing/write_images_to_filesystem.py --commit
##### 
##### 2020-03-15 19:07:26+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 5c8da87 [ahead 3] kaggle_compile.py | ./data_output/scripts/write_images_to_filesystem.py
##### 
##### 5c8da873b06614fd5aa786c4a8de708fcd988c48
##### 

#####
##### START src/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

settings = {}

settings['hparam_defaults'] = {
    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "min_lr":        0.001,
    "split":         0.2,
    "batch_size":    128,
    "fraction":      1.0,
    "patience": {
        'Localhost':    5,
        'Interactive':  0,
        'Batch':        5,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],
    "loops": {
        'Localhost':   1,
        'Interactive': 1,
        'Batch':       1,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],

    # Timeout = 120 minutes | allow 30 minutes for testing submit | TODO: unsure of KAGGLE_KERNEL_RUN_TYPE on Submit
    "timeout": "5m" if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == "Interactive" else "90m"
}

settings['verbose'] = {
    "tensorboard": {
        {
            'Localhost':   True,
            'Interactive': False,
            'Batch':       False,
        }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
    },
    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       2,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./models",
        "submissions": "./",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./data_output/models",
        "submissions": "./data_output/submissions",
        "logs":        "./logs",
    }
for dirname in settings['dir'].values(): os.makedirs(dirname, exist_ok=True)



#####
##### END   src/settings.py
#####

#####
##### START src/dataset/DatasetDF.py
#####

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

# from src.settings import settings



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
        img_cropped = cls.crop_image(img)
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


#####
##### END   src/dataset/DatasetDF.py
#####

#####
##### START src/util/argparse.py
#####

import argparse
import copy
from typing import Dict, List



def argparse_from_dicts(configs: List[Dict], inplace=False) -> List[Dict]:
    parser = argparse.ArgumentParser()
    for config in list(configs):
        for key, value in config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f'{key} (default: %(default)s)')
            else:
                parser.add_argument(f'--{key}', type=type(value),    default=value, help=f'{key} (default: %(default)s)')


    args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle

    outputs = configs if inplace else copy.deepcopy(configs)
    for index, output in enumerate(outputs):
        for key, value in outputs[index].items():
            outputs[index][key] = getattr(args, key)

    return outputs


def argparse_from_dict(config: Dict, inplace=False):
    return argparse_from_dicts([config], inplace)[0]


#####
##### END   src/util/argparse.py
#####

#####
##### START src/preprocessing/write_images_to_filesystem.py
#####

#!/usr/bin/env python
import copy
import os
import time

import glob2
import matplotlib
import matplotlib.image
import pandas as pd
from pyarrow.parquet import ParquetFile

# from src.dataset.DatasetDF import DatasetDF
# from src.settings import settings
# from src.util.argparse import argparse_from_dicts



# Entries into the Bengali AI Competition often suffer from out of memory errors when reading from a dataframe
# Quick and dirty solution is to write data as images to a directory and use ImageDataGenerator.flow_from_directory()
def write_images_to_filesystem( data_dir, feature_dir, ext='png', only=None, verbose=False, force=False, transform_args={} ):
    transform_defaults = { 'resize': 2, 'denoise': True, 'center': True, 'invert': True, 'normalize': False }
    transform_args     = { **transform_defaults, **transform_args }

    time_start = time.time()
    filename_groups = {
        "test":  sorted(glob2.glob(f"{settings['dir']['data']}/test_image_data_*.parquet")),
        "train": sorted(glob2.glob(f"{settings['dir']['data']}/train_image_data_*.parquet")),
    }
    if only:
        for test_train_valid, parquet_filenames in list(filename_groups.items()):
            if only not in test_train_valid:
                del filename_groups[test_train_valid]

    image_count = 0
    for test_train_valid, parquet_filenames in filename_groups.items():
        image_dir = f'{feature_dir}/{test_train_valid}'
        os.makedirs(image_dir, exist_ok=True)

        # Skip image creation if all images have already been extracted
        if not force:
            expected_images = sum([ ParquetFile(file).metadata.num_rows for file in parquet_filenames ])
            existing_images = len(glob2.glob(f'{image_dir}/*.{ext}'))
            if existing_images == expected_images: continue

        for parquet_filename in parquet_filenames:
            if verbose >= 2:
                print(f'write_images_to_filesystem({only or ""}) - reading:  ', parquet_filename)

            dataframe  = pd.read_parquet(parquet_filename)
            image_ids  = dataframe['image_id'].tolist()
            image_data = DatasetDF.transform_X(dataframe, **transform_args )

            for index, image_id in enumerate(image_ids):
                image_filename = f'{image_dir}/{image_id}.{ext}'
                if not force and os.path.exists(image_filename):
                    print(f'write_images_to_filesystem({only or ""}) - skipping: ', image_filename)
                    continue

                matplotlib.image.imsave(image_filename, image_data[index].squeeze(), cmap='gray')
                image_count += 1
                if verbose:
                    print(f'write_images_to_filesystem({only or ""}) - wrote:    ', image_filename)

    if verbose >= 1:
        print( f'write_images_to_filesystem({only or ""}) - wrote: {image_count} files in: {round(time.time() - time_start,2)}s')



if __name__ == '__main__':
    args = {
        'data_dir':    settings['dir']['data'],
        'feature_dir': settings['dir']['features'],
        'ext':         'png',
        'verbose':      2,
        'force':        False,
    }
    transform_args = {
        'resize':    2,
        'denoise':   1,
        'center':    1,
        'invert':    1,
        'normalize': 0
    }
    argparse_from_dicts([args, transform_args], inplace=True)

    test_args = copy.deepcopy(args)
    test_args['force'] = True

    write_images_to_filesystem(only='test',  transform_args=transform_args, **test_args )
    write_images_to_filesystem(only='train', transform_args=transform_args, **args )


#####
##### END   src/preprocessing/write_images_to_filesystem.py
#####

##### 
##### ./kaggle_compile.py src/preprocessing/write_images_to_filesystem.py --commit
##### 
##### 2020-03-15 19:07:26+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 5c8da87 [ahead 3] kaggle_compile.py | ./data_output/scripts/write_images_to_filesystem.py
##### 
##### 5c8da873b06614fd5aa786c4a8de708fcd988c48
##### 