#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/experiments/simple_triple_df_cnn.py --save
##### 
##### 2020-03-12 19:26:12+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 6ad3198 [ahead 2] KaggleTimeoutCallback.py | implement 115m timeout for Kaggle
##### 
##### 6ad3198785e4776e0e42633ffc55def28893e84c
##### 
##### Wrote: ./data_output/scripts/simple_triple_df_cnn.py

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
        'Localhost':   "110m",
        'Interactive': "1m",
        'Batch':       "110m",  # Timeout = 120 minutes | Submit exceeds timeout when using 115m
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

import os
from typing import List, Union

import glob2
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# from src.settings import settings


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


#####
##### END   src/dataset/DatasetDF.py
#####

#####
##### START vendor/CLR/clr_callback.py
#####

from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K


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
##### START src/models/SingleOutputCNN.py
#####

import inspect
import types
from typing import cast

from tensorflow_core.python.keras import Input, Model, regularizers
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    GlobalMaxPooling2D


def SingleOutputCNN(
        input_shape,
        output_shape,
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

    for cnn1 in range(0,maxpool_layers):
        for cnn2 in range(1, cnns_per_maxpool+1):
            x = Conv2D( 32 * cnn2, kernel_size=(3, 3), padding='same', activation='relu')(x)
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

    x = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs, x, name=model_name)
    # plot_model(model, to_file=os.path.join(os.path.dirname(__file__), f"{name}.png"))
    return model


#####
##### END   src/models/SingleOutputCNN.py
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

from pandas import DataFrame


def df_to_submission(df: DataFrame) -> DataFrame:
    output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
    submission = DataFrame(columns=['row_id', 'target'])
    for index, row in df.iterrows():
        for output_field in output_fields:
            submission = submission.append({
                'row_id': f"Test_{index}_{output_field}",
                'target': df[output_field].iloc[index],
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
import re
import time
from typing import Dict, AnyStr

import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
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


def model_compile_fit(
        hparams:    Dict,
        model:      tf.keras.models.Model,
        dataset:    DatasetDF,
        model_file: AnyStr = None,
        log_dir:    AnyStr = None,
        best_only   = True,
        verbose     = settings['verbose']['fit'],
):
    hparams   = { **settings['hparam_defaults'], **hparams }
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
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
        # ProgbarLogger(count_mode='samples', stateful_metrics=None)
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
    if log_dir and settings['verbose']['tensorboard']:
        callbacks += [  
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            KerasCallback(log_dir, hparams)                                     # log train_hparams
        ]

    timer_start = time.time()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.0001)),
        metrics=['accuracy']
    )
    history = model.fit(
        dataset.X["train"], dataset.Y["train"],
        batch_size=hparams.get("batch_size", 128),
        epochs=999,
        verbose=verbose,
        validation_data=(dataset.X["valid"], dataset.Y["valid"]),
        callbacks=callbacks
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
##### START src/experiments/simple_triple_df_cnn.py
#####

# This is a first pass, simplest thing that could possibly work attempt
# We train three separate MINST style CNNs for each label, then combine the results
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging

import numpy as np
import pandas as pd
import tensorflow as tf

# from src.dataset.DatasetDF import DatasetDF
# from src.models.SingleOutputCNN import SingleOutputCNN
# from src.util.argparse import argparse_from_dicts
# from src.util.csv import df_to_submission_csv
# from src.util.hparam import model_compile_fit
# from src.settings import settings

# https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
config  = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)



def simple_triple_df_cnn(train_hparams, model_hparams):
    model_hparams_key = "-".join( f"{key}={value}" for key,value in model_hparams.items() ).replace(' ','')
    print("train_hparams", train_hparams)
    print("model_hparams", model_hparams)


    csv_data    = pd.read_csv(f"{settings['dir']['data']}/train.csv")
    models      = {}
    model_files = {}
    model_stats = {}
    output_fields = [ "consonant_diacritic", "grapheme_root", "vowel_diacritic" ]
    for output_field in output_fields:
        model_stats[output_field] = []
        model_files[output_field] = f"{settings['dir']['models']}/simple_triple_df_cnn/SingleOutputCNN-{model_hparams_key}-{output_field}.hdf5"
        os.makedirs(os.path.dirname(model_files[output_field]), exist_ok=True)


    for output_field in output_fields:
        # # Release GPU memory
        # cuda.select_device(0)
        # cuda.close()
        # gc.collect()

        output_shape         = csv_data[output_field].nunique()
        models[output_field] = SingleOutputCNN(
            input_shape=(137,236, 1),
            output_shape=output_shape,
            name=output_field,
            **model_hparams,
        )

        if os.path.exists( model_files[output_field] ):
            models[output_field].load_weights( model_files[output_field] )

        models[output_field].summary()

        for loop in range(train_hparams['loops']):
            for data_id in range(0,4):
                print("------------------------------")
                print(f"Training | {output_field} | data_id: {data_id}")
                print(f"model_hparams: {model_hparams}")
                print(f"train_hparams: {train_hparams}")
                print("------------------------------")

                dataset = DatasetDF(
                    test_train='train',
                    data_id=data_id,
                    Y_field=output_field,
                    split=train_hparams['split'],
                    fraction=train_hparams['fraction'],
                )

                stats = model_compile_fit(
                    hparams    = {**train_hparams, **model_hparams},
                    model      = models[output_field],
                    dataset    = dataset,
                    model_file = model_files[output_field],
                    log_dir    = f"{settings['dir']['logs']}/simple_triple_df_cnn/{output_field}/",
                    best_only  = True,
                )
                if stats is None: break  # KaggleTimeoutCallback() triggered on_train_begin()
                model_stats[output_field].append(stats)
            else: continue
            break                        # KaggleTimeoutCallback() triggered on_train_begin()

        print("------------------------------")
        print(f"Completed | {output_field}")
        print(f"model_hparams: {model_hparams}")
        print(f"train_hparams: {train_hparams}")
        for stats in model_stats[output_field]: print(stats)
        print("------------------------------")


    ### Log Stats Results
    logfilename = f"{settings['dir']['submissions']}/SingleOutputCNN-{model_hparams_key}-submission.log"
    with open(logfilename, 'w') as file:
        output = []
        output.append("------------------------------")
        for output_field in output_fields:
            output.append(f"Completed | {output_field}")
            output.append(f"model_hparams: {model_hparams}")
            output.append(f"train_hparams: {train_hparams}")
            for stats in model_stats[output_field]:
                output.append(str(stats))
            output.append("------------------------------")
        print(      "\n".join(output) )
        file.write( "\n".join(output) )
        print("wrote:", logfilename)


    ### Output Predictions to CSV
    test_dataset = DatasetDF(test_train='test', data_id='*')  # contains all test data
    predictions  = pd.DataFrame()
    for output_field in output_fields:
        prediction = models[output_field].predict(test_dataset.X['train'])
        prediction = np.argmax( prediction, axis=-1 )
        predictions[output_field] = prediction


    df_to_submission_csv(
        predictions,
        f"{settings['dir']['submissions']}/SingleOutputCNN-{model_hparams_key}-submission.csv"
    )



if __name__ == '__main__':
    # model_hparams = {
    #     "cnns_per_maxpool": 2,
    #     "maxpool_layers":   6,
    #     "dense_layers":     4,
    #     "dense_units":    128,
    #     "fraction":       0.1,
    # }
    model_hparams = {
        "cnns_per_maxpool":   1,
        "maxpool_layers":     5,
        "dense_layers":       1,
        "dense_units":       64,
        "regularization": False,
        "global_maxpool": False,
    }
    train_hparams = {
        "optimizer":     "RMSprop",
        "scheduler":     "constant",
        "learning_rate": 0.001,
        # "min_lr":        0.001,
        # "split":         0.2,
        # "batch_size":    128,
        # "patience":      10,
        # "fraction":      1.0,
        # "loops":         2,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['loops']    = 1

    train_hparams = { **settings['hparam_defaults'], **train_hparams }
    argparse_from_dicts([train_hparams, model_hparams])

    simple_triple_df_cnn(train_hparams, model_hparams)

#####
##### END   src/experiments/simple_triple_df_cnn.py
#####

##### 
##### ./kaggle_compile.py src/experiments/simple_triple_df_cnn.py --save
##### 
##### 2020-03-12 19:26:12+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 6ad3198 [ahead 2] KaggleTimeoutCallback.py | implement 115m timeout for Kaggle
##### 
##### 6ad3198785e4776e0e42633ffc55def28893e84c
##### 
##### Wrote: ./data_output/scripts/simple_triple_df_cnn.py