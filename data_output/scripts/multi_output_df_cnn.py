#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/experiments/multi_output_df_cnn.py --save
##### 
##### 2020-03-14 18:18:46+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 3883707 DatasetDF | give fraction<1 a random shuffle
##### 
##### 388370722a16c50c4cb82df4e2c21b1a3f1170c7
##### 
##### Wrote: ./data_output/scripts/multi_output_df_cnn.py

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
    "timeout": {
        'Localhost':   "100m",
        'Interactive': "5m",
        'Batch':       "100m",  # Timeout = 120 minutes | allow 20 minutes for submit
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
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
        "models":      "./models",
        "submissions": "./",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input/bengaliai-cv19",
        "models":      "./data_output/models",
        "submissions": "./data_output/submissions",
        "logs":        "./logs",
    }
for dirname in settings['dir'].values(): os.makedirs(dirname, exist_ok=True)



#####
##### END   src/settings.py
#####

#####
##### START src/callbacks/KaggleTimeoutCallback.py
#####

import math
import re
import time
from typing import Union

import tensorflow as tf


class KaggleTimeoutCallback(tf.keras.callbacks.Callback):
    start_python = time.time()


    def __init__(self, timeout: Union[int, float, str], from_now=False, verbose=False):
        super().__init__()
        self.verbose           = verbose
        self.from_now          = from_now
        self.start_time        = self.start_python if not self.from_now else time.time()
        self.timeout_seconds   = self.parse_seconds(timeout)

        self.last_epoch_start  = time.time()
        self.last_epoch_end    = time.time()
        self.last_epoch_time   = self.last_epoch_end - self.last_epoch_start
        self.current_runtime   = self.last_epoch_end - self.start_time


    def on_train_begin(self, logs=None):
        self.check_timeout()  # timeout before first epoch if model.fit() is called again


    def on_epoch_begin(self, epoch, logs=None):
        self.last_epoch_start = time.time()


    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch_end  = time.time()
        self.last_epoch_time = self.last_epoch_end - self.last_epoch_start
        self.check_timeout()


    def check_timeout(self):
        self.current_runtime = self.last_epoch_end - self.start_time
        if self.verbose:
            print(f'\nKaggleTimeoutCallback({self.format(self.timeout_seconds)}) runtime {self.format(self.current_runtime)}')

        # Give timeout leeway of 2 * last_epoch_time
        if (self.current_runtime + self.last_epoch_time*2) >= self.timeout_seconds:
            print(f"\nKaggleTimeoutCallback({self.format(self.timeout_seconds)}) stopped after {self.format(self.current_runtime)}")
            self.model.stop_training = True


    @staticmethod
    def parse_seconds(timeout) -> int:
        if isinstance(timeout, (float,int)): return int(timeout)
        seconds = 0
        for (number, unit) in re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])?", str(timeout)):
            if   unit == 'd': seconds += float(number) * 60 * 60 * 24
            elif unit == 'h': seconds += float(number) * 60 * 60
            elif unit == 'm': seconds += float(number) * 60
            else:             seconds += float(number)
        return int(seconds)


    @staticmethod
    def format(seconds: Union[int,float]) -> str:
        runtime = {
            "d":   math.floor(seconds / (60*60*24) ),
            "h":   math.floor(seconds / (60*60)    ) % 24,
            "m":   math.floor(seconds / (60)       ) % 60,
            "s":   math.floor(seconds              ) % 60,
        }
        return " ".join([ f"{runtime[unit]}{unit}" for unit in ["h", "m", "s"] if runtime[unit] != 0 ])


#####
##### END   src/callbacks/KaggleTimeoutCallback.py
#####

#####
##### START src/dataset/DatasetDF.py
#####

import math
from time import sleep

import gc
import os
from typing import AnyStr, Dict, Union
import skimage.measure
import glob2
import numpy as np
import pandas as pd
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


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/edit/run/29865909
    # noinspection PyArgumentList
    @classmethod
    def transform_X(cls, train: DataFrame, resize=2, denoise=True, normalize=True, center=True, invert=True) -> np.ndarray:
        train = (train.drop(columns='image_id', errors='ignore')
                 .values.astype('uint8')            # unit8 for initial data processing
                 .reshape(-1, 137, 236)             # 2D array for inline image processing
        )
        gc.collect(); sleep(1)

        if isinstance(resize, bool) and resize == True:
            resize = 2
        if resize and resize != 1:                  # Reduce image size by 2x
            # skimage.measure.block_reduce() default cast: int -> float64
            # Use: np.array([train[i] for in]) helps reduce peak memory requirements
            mean = lambda x, axis: np.mean(x, axis=axis, dtype=np.uint8)  # func_kwargs={} arg not in pip version
            train = skimage.measure.block_reduce(train, (1, resize,resize), cval=0, func=mean)

        if invert:                                         # Colors | 0 = black      | 255 = white
            train = (255-train)                            # invert | 0 = background | 255 = line
        if denoise:                                        # Set small pixel values to background 0
            if invert: train *= (train >= 25)              #   0 = background | 255 = line  | np.mean() == 12
            else:      train += (255-train)*(train >= 230) # 255 = background |   0 = line  | np.mean() == 244
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


#####
##### END   src/dataset/DatasetDF.py
#####

#####
##### START vendor/CLR/clr_callback.py
#####

from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


#####
##### END   vendor/CLR/clr_callback.py
#####

#####
##### START src/models/MultiOutputCNN.py
#####

import inspect
import types
from typing import cast, Union, List, Dict

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    GlobalMaxPooling2D


# noinspection DuplicatedCode
def MultiOutputCNN(
        input_shape,
        output_shape: Union[List, Dict],
        cnns_per_maxpool=1,
        maxpool_layers=1,
        dense_layers=1,
        dense_units=64,
        dropout=0.25,
        regularization=False,
        global_maxpool=False,
        name='',
)  -> Model:
    function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    model_name    = f"{function_name}-{name}" if name else function_name
    # model_name  = seq([ function_name, name ]).filter(lambda x: x).make_string("-")  # remove dependency on pyfunctional - not in Kaggle repo without internet

    inputs = Input(shape=input_shape)
    x      = inputs

    for cnn1 in range(1,maxpool_layers+1):
        for cnn2 in range(1, cnns_per_maxpool+1):
            x = Conv2D( 32 * cnn1, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    if global_maxpool:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for nn1 in range(0,dense_layers):
        if regularization:
            x = Dense(dense_units, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.01))(x)
        else:
            x = Dense(dense_units, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    x = Flatten(name='output')(x)

    if isinstance(output_shape, dict):
        outputs = [
            Dense(output_shape, activation='softmax', name=key)(x)
            for key, output_shape in output_shape.items()
        ]
    else:
        outputs = [
            Dense(output_shape, activation='softmax', name=f'output_{index}')(x)
            for index, output_shape in enumerate(output_shape)
        ]

    model = Model(inputs, outputs, name=model_name)
    # plot_model(model, to_file=os.path.join(os.path.dirname(__file__), f"{name}.png"))
    return model


#####
##### END   src/models/MultiOutputCNN.py
#####

#####
##### START src/util/argparse.py
#####

import argparse
from typing import List, Dict


def argparse_from_dicts(configs: List[Dict]):
    parser = argparse.ArgumentParser()
    for config in list(configs):
        for key, value in config.items():
            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key} (default: %(default)s)')

    args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle
    for config in list(configs):
        for key, value in config.items():
            config[key] = getattr(args, key)


#####
##### END   src/util/argparse.py
#####

#####
##### START src/util/csv.py
#####

import os

from pandas import DataFrame


def df_to_submission(df: DataFrame) -> DataFrame:
    output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
    submission = DataFrame(columns=['row_id', 'target'])
    for index, row in df.iterrows():
        for output_field in output_fields:
            index = f"Test_{index}" if not str(index).startswith('T') else index
            submission = submission.append({
                'row_id': f"{index}_{output_field}",
                'target': df[output_field].loc[index],
            }, ignore_index=True)
    return submission


def df_to_submission_csv(df: DataFrame, filename: str):
    submission = df_to_submission(df)
    submission.to_csv(filename, index=False)
    print("wrote:", filename, submission.shape)

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        submission.to_csv('submission.csv', index=False)
        print("wrote:", 'submission.csv', submission.shape)

#####
##### END   src/util/csv.py
#####

#####
##### START src/util/hparam.py
#####

import math
import os
import re
import time
from typing import Dict, AnyStr, Union

import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, \
    ModelCheckpoint

# from src.callbacks.KaggleTimeoutCallback import KaggleTimeoutCallback
# from src.dataset.DatasetDF import DatasetDF
# from src.settings import settings
# from vendor.CLR.clr_callback import CyclicLR


def min_lr(hparams):
    # tensorboard --logdir logs/convergence_search/min_lr-optimized_scheduler-random-scheduler/ --reload_multifile=true
    # There is a high degree of randomness in this parameter, so it is hard to distinguish from statistical noise
    # Lower min_lr values for CycleCR tend to train slower
    hparams = { **settings['hparam_defaults'], **hparams }
    if 'min_lr'  in hparams:              return hparams['min_lr']
    if hparams["optimizer"] == "SGD":     return 1e05  # preferred by SGD
    else:                                 return 1e03  # fastest, least overfitting and most accidental high-scores


# DOCS: https://ruder.io/optimizing-gradient-descent/index.html
def scheduler(hparams: dict, dataset: DatasetDF, verbose=False):
    hparams = { **settings['hparam_defaults'], **hparams }
    if hparams['scheduler'] is 'constant':
        return LearningRateScheduler(lambda epocs: hparams['learning_rate'], verbose=False)

    if hparams['scheduler'] is 'linear_decay':
        return LearningRateScheduler(
            lambda epocs: max(
                hparams['learning_rate'] * (10. / (10. + epocs)),
                min_lr(hparams)
            ),
            verbose=verbose
        )

    if hparams['scheduler'].startswith('CyclicLR') \
            or hparams['scheduler'] in ["triangular", "triangular2", "exp_range"]:
        # DOCS: https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
        # CyclicLR_triangular, CyclicLR_triangular2, CyclicLR_exp_range
        mode = re.sub(r'^CyclicLR_', '', hparams['scheduler'])

        # step_size should be epoc multiple between 2 and 8, but multiple of 2 (= full up/down cycle)
        if   hparams['patience'] <=  6: whole_cycles = 1   #  1/2   = 0.5  | 6/2    = 3
        elif hparams['patience'] <= 12: whole_cycles = 2   #  8/4   = 2    | 12/4   = 3
        elif hparams['patience'] <= 24: whole_cycles = 3   # 14/6   = 2.3  | 24/6   = 4
        elif hparams['patience'] <= 36: whole_cycles = 4   # 26/8   = 3.25 | 36/8   = 4.5
        elif hparams['patience'] <= 48: whole_cycles = 5   # 28/10  = 2.8  | 48/10  = 4.8
        elif hparams['patience'] <= 72: whole_cycles = 6   # 50/12  = 4.2  | 72/12  = 6
        elif hparams['patience'] <= 96: whole_cycles = 8   # 74/16  = 4.6  | 96/16  = 6
        else:                           whole_cycles = 12  # 100/24 = 4.2  | 192/24 = 8

        return CyclicLR(
            mode      = mode,
            step_size =dataset.epoch_size() * (hparams['patience'] / (2.0 * whole_cycles)),
            base_lr   = min_lr(hparams),
            max_lr    = hparams['learning_rate']
        )

    if hparams['scheduler'].startswith('plateau'):
        factor = int(( re.findall(r'\d+', hparams['scheduler']) + [10] )[0])            # plateau2      || plateau10 (default)
        if 'sqrt' in hparams['scheduler']:  patience = math.sqrt(hparams['patience'])  # plateau2_sqrt || plateau10__sqrt
        else:                               patience = hparams['patience'] / 2.0

        return ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 1 / factor,
            patience = math.floor(patience),
            min_lr   = 0,   # min_lr(train_hparams),
            verbose  = verbose,
        )

    print("Unknown scheduler: ", hparams)


def losses(output_shape):
    if   isinstance(output_shape, list): losses = [ categorical_crossentropy      for n   in output_shape        ]
    elif isinstance(output_shape, dict): losses = { key: categorical_crossentropy for key in output_shape.keys() }
    else:                                losses = categorical_crossentropy
    return losses


def loss_weights(output_shape):
    # unique = dataset.apply(lambda col: col.nunique()); unique
    # grapheme_root           168   | sqrt = 12.9 / 54.9 = 0.24
    # vowel_diacritic          11   | sqrt =  3.3 / 54.9 = 0.06
    # consonant_diacritic       7   | sqrt =  2.6 / 54.9 = 0.05
    # grapheme               1295   | sqrt = 35.9 / 54.9 = 0.65
    if not isinstance(output_shape, dict): return None
    norm    = sum(map(math.sqrt, output_shape.values()))
    weights = {
        key: math.sqrt(value)/norm
        for key,value in output_shape.items()
    }
    return weights



def callbacks(hparams, dataset, model_file=None, log_dir=None, best_only=True, verbose=False ):
    schedule  = scheduler(hparams, dataset, verbose=verbose)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=verbose,
            patience=hparams.get('patience', hparams['patience']),
            restore_best_weights=best_only
        ),
        schedule,
        KaggleTimeoutCallback( hparams["timeout"], verbose=False ),
    ]
    if model_file:
        callbacks += [
            ModelCheckpoint(
                model_file,
                monitor='val_loss',
                verbose=False,
                save_best_only=best_only,
                save_weights_only=False,
                mode='auto',
            )
        ]
    if log_dir and settings['verbose']['tensorboard'] and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        callbacks += [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            KerasCallback(log_dir, hparams)                                     # log train_hparams
        ]
    return callbacks


def model_compile_fit(
        hparams:      Dict,
        model:        tf.keras.models.Model,
        dataset:      DatasetDF,
        epochs      = 999,
        output_shape: Union[None, int, Dict] = None,
        model_file:   AnyStr = None,
        log_dir:      AnyStr = None,
        best_only   = True,
        verbose     = settings['verbose']['fit'],
):
    hparams   = { **settings['hparam_defaults'], **hparams }
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    callback  = callbacks(hparams, dataset, model_file, log_dir, best_only, verbose)
    loss      = losses(output_shape)
    weights   = loss_weights(output_shape)

    timer_start = time.time()
    model.compile(
        loss=loss,
        loss_weights=weights,
        optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.001)),
        metrics=['accuracy']
    )
    history = model.fit(
        dataset.X["train"], dataset.Y["train"],
        batch_size=hparams.get("batch_size", 128),
        epochs=epochs,
        verbose=verbose,
        validation_data=(dataset.X["valid"], dataset.Y["valid"]),
        callbacks=callback
    )
    timer_seconds = int(time.time() - timer_start)

    if 'val_loss' in history.history:
        best_epoch            = history.history['val_loss'].index(min( history.history['val_loss'] )) if best_only else -1
        model_stats           = { key: value[best_epoch] for key, value in history.history.items() }
        model_stats['time']   = timer_seconds
        model_stats['epochs'] = len(history.history['loss'])
    else:
        model_stats = None
    return model_stats


#####
##### END   src/util/hparam.py
#####

#####
##### START src/experiments/multi_output_df_cnn.py
#####

import os

import numpy as np
import pandas as pd

# from src.dataset.DatasetDF import DatasetDF
# from src.models.MultiOutputCNN import MultiOutputCNN
# from src.settings import settings
# from src.util.argparse import argparse_from_dicts
# from src.util.csv import df_to_submission_csv
# from src.util.hparam import model_compile_fit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging

# NOTE: This line doesn't work on Kaggle
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
# [ tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU') ]


def hparam_key(hparams):
    return "-".join( f"{key}={value}" for key,value in hparams.items() ).replace(' ','')


def multi_output_df_cnn(train_hparams, model_hparams, pipeline_name):
    print("pipeline_name", pipeline_name)
    print("train_hparams", train_hparams)
    print("model_hparams", model_hparams)

    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)

    # csv_data    = pd.read_csv(f"{settings['dir']['data']}/train.csv")
    model_file  = f"{settings['dir']['models']}/{pipeline_name}/{pipeline_name}-{model_hparams_key}.hdf5"
    log_dir     = f"{settings['dir']['logs']}/{pipeline_name}/{model_hparams_key}/{train_hparams_key}"

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(log_dir,                     exist_ok=True)

    # output_shape = csv_data.drop(columns='image_id').nunique().to_dict()
    input_shape  = DatasetDF(test_train='train', fraction=0.0001, data_id=0).input_shape()
    output_shape = DatasetDF.output_shape()
    model = MultiOutputCNN(
        input_shape=input_shape,
        output_shape=output_shape,
        **model_hparams,
    )
    if os.path.exists( model_file ):
        model.load_weights( model_file )
    model.summary()

    model_stats = []
    for loop in range(train_hparams['loops']):
        for data_id in range(0,4):
            print("------------------------------")
            print(f"Training | data_id: {data_id}")
            print(f"model_hparams: {model_hparams}")
            print(f"train_hparams: {train_hparams}")
            print("------------------------------")

            dataset = DatasetDF(
                test_train='train',
                data_id=data_id,
                split=train_hparams['split'],
                fraction=train_hparams['fraction'],
            )

            stats = model_compile_fit(
                hparams      = {**train_hparams, **model_hparams},
                model        = model,
                dataset      = dataset,
                output_shape = output_shape,
                model_file   = model_file,
                log_dir      = log_dir,
                best_only    = True,
                verbose      = 2,
            )
            if stats is None: break  # KaggleTimeoutCallback() triggered on_train_begin()
            model_stats.append(stats)
        else: continue
        break                        # KaggleTimeoutCallback() triggered on_train_begin()
    return model, model_stats, output_shape



### Log Stats Results
def log_stats_results(model_stats, logfilename):
    with open(logfilename, 'w') as file:
        output = [
            "------------------------------",
            f"Completed",
            f"model_hparams: {model_hparams}",
            f"train_hparams: {train_hparams}"
        ]
        output += list(map(str, model_stats))
        output += [
            "------------------------------"
        ]
        output = "\n".join(output)
        print(      output )
        file.write( output )
        print("wrote:", logfilename)


### Predict Output Submssion
def submission_df(model, output_shape):
    submission = pd.DataFrame(columns=output_shape.keys())
    for data_id in range(0,4):
        test_dataset = DatasetDF(test_train='test', data_id=data_id)  # large datasets on submit, so loop
        predictions  = model.predict(test_dataset.X['train'])
        # noinspection PyTypeChecker
        submission = submission.append(
            pd.DataFrame({
                key: np.argmax( predictions[index], axis=-1 )
                for index, key in enumerate(output_shape.keys())
            }, index=test_dataset.ID['train'])
        )
    return submission



if __name__ == '__main__':
    # model_hparams = {
    #     "cnns_per_maxpool": 2,
    #     "maxpool_layers":   6,
    #     "dense_layers":     4,
    #     "dense_units":    128,
    #     "fraction":       0.1,
    # }
    model_hparams = {
        "cnns_per_maxpool":   3,
        "maxpool_layers":     4,
        "dense_layers":       2,
        "dense_units":      256,
        "regularization": False,
        "global_maxpool": False,
    }
    train_hparams = {
        "optimizer":     "RMSprop",
        "scheduler":     "constant",
        "learning_rate": 0.001,

        # "optimizer": "Adagrad",
        # "scheduler": "plateau2",
        # "learning_rate": 1,

        # "min_lr":        0.001,
        # "split":         0.2,
        # "batch_size":    128,
        # "fraction":      1.0,
        "patience":      10,
        "loops":         2,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['loops']    = 1
    train_hparams = { **settings['hparam_defaults'], **train_hparams }

    argparse_from_dicts([train_hparams, model_hparams])


    pipeline_name     = "multi_output_df_cnn"
    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)
    logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.log"
    csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.csv"

    model, model_stats, output_shape = multi_output_df_cnn(train_hparams, model_hparams, pipeline_name)

    submission = submission_df(model, output_shape)
    df_to_submission_csv( submission, csv_filename )



#####
##### END   src/experiments/multi_output_df_cnn.py
#####

##### 
##### ./kaggle_compile.py src/experiments/multi_output_df_cnn.py --save
##### 
##### 2020-03-14 18:18:46+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 3883707 DatasetDF | give fraction<1 a random shuffle
##### 
##### 388370722a16c50c4cb82df4e2c21b1a3f1170c7
##### 
##### Wrote: ./data_output/scripts/multi_output_df_cnn.py