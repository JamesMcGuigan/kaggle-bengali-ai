#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/experiments/multi_output_df_cnn.py --save
##### 
##### 2020-03-13 04:09:23+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 4086a2c [ahead 2] ./kaggle_compile.py | src/experiments/multi_output_df_cnn.py
##### 
##### 4086a2c7192388dd362105bf3e53d2f170a95aa0
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

import gc
import os
from typing import AnyStr, Dict, Union

import glob2
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
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

        self.X:  Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        self.Y:  Dict[AnyStr, Union[pd.DataFrame,Dict]] = { "train": pd.DataFrame(),   "valid": pd.DataFrame()   }
        self.ID: Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        for filename in self.image_filenames:
            raw = {}
            raw['train'], raw['valid'] = pd.read_parquet(filename), None
            if self.fraction < 1:
                raw['train'], discard      = train_test_split(raw['train'], train_size=self.fraction, shuffle=self.shuffle, random_state=0)
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
##### START src/models/MultiOutputCNN.py
#####

import inspect
import types
from typing import cast, Union, List, Dict

from tensorflow_core.python.keras import Input, Model, regularizers
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    GlobalMaxPooling2D


def MultiOutputCNN(
        input_shape,
        output_shapes: Union[List, Dict],
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

    if isinstance(output_shapes, dict):
        outputs = [
            Dense(output_shape, activation='softmax', name=key)(x)
            for key, output_shape in output_shapes.items()
        ]
    else:
        outputs = [
            Dense(output_shape, activation='softmax', name=f'output_{index}')(x)
            for index, output_shape in enumerate(output_shapes)
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
##### START src/experiments/multi_output_df_cnn.py
#####

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# from src.callbacks.KaggleTimeoutCallback import KaggleTimeoutCallback
# from src.dataset.DatasetDF import DatasetDF
# from src.models.MultiOutputCNN import MultiOutputCNN
# from src.util.hparam import model_compile_fit
# from src.settings import settings
# from src.util.argparse import argparse_from_dicts
# from src.util.csv import df_to_submission_csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging

# NOTE: This line doesn't work on Kaggle
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
# [ tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU') ]



def multi_output_df_cnn(train_hparams, model_hparams):
    pipeline_name     = "multi_output_df_cnn"
    model_hparams_key = "-".join( f"{key}={value}" for key,value in model_hparams.items() ).replace(' ','')
    print("train_hparams", train_hparams)
    print("model_hparams", model_hparams)

    csv_data    = pd.read_csv(f"{settings['dir']['data']}/train.csv")
    model_file  = f"{settings['dir']['models']}/{pipeline_name}/{pipeline_name}-{model_hparams_key}.hdf5"
    log_dir     = f"{settings['dir']['logs']}/{pipeline_name}"

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(log_dir,                     exist_ok=True)

    output_shapes = csv_data.drop(columns='image_id').nunique().to_dict()
    model = MultiOutputCNN(
        input_shape=(137,236, 1),
        output_shapes=output_shapes,
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

            hparams   = { **settings['hparam_defaults'], **train_hparams }
            optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])

            timer_start = time.time()
            model.compile(
                loss={
                    key: tf.keras.losses.categorical_crossentropy
                    for key in output_shapes.keys()
                },
                optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.0001)),
                metrics=['accuracy']
            )
            history = model.fit(
                dataset.X["train"], dataset.Y["train"],
                batch_size=hparams.get("batch_size"),
                epochs=999,
                verbose=2,
                validation_data=(dataset.X["valid"], dataset.Y["valid"]),
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        verbose=True,
                        patience=hparams.get('patience'),
                        restore_best_weights=True
                    ),
                    KaggleTimeoutCallback( hparams["timeout"], verbose=False ),
                ]
            )
            timer_seconds = int(time.time() - timer_start)

            if 'val_loss' in history.history:
                best_epoch      = history.history['val_loss'].index(min( history.history['val_loss'] ))
                stats           = { key: value[best_epoch] for key, value in history.history.items() }
                stats['time']   = timer_seconds
                stats['epochs'] = len(history.history['loss'])
                model_stats.append(stats)


    ### Log Stats Results
    logfilename = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.log"
    with open(logfilename, 'w') as file:
        output = []
        output.append("------------------------------")
        output.append(f"Completed")
        output.append(f"model_hparams: {model_hparams}")
        output.append(f"train_hparams: {train_hparams}")
        for stats in model_stats:
            output.append(str(stats))
        output.append("------------------------------")
        print(      "\n".join(output) )
        file.write( "\n".join(output) )
        print("wrote:", logfilename)


    ### Output Predictions to CSV
    submission = pd.DataFrame(columns=output_shapes.keys())
    for data_id in range(0,4):
        test_dataset = DatasetDF(test_train='test', data_id=data_id)  # contains all test data
        predictions  = model.predict(test_dataset.X['train'])
        # noinspection PyTypeChecker
        submission = submission.append(
            pd.DataFrame({
                key: np.argmax( predictions[index], axis=-1 )
                for index, key in enumerate(output_shapes.keys())
            }, index=test_dataset.ID['train'])
        )

    df_to_submission_csv(
        submission,
        f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.csv"
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
        # "fraction":      1.0,
        "patience":      10,
        "loops":         2,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['loops']    = 1

    train_hparams = { **settings['hparam_defaults'], **train_hparams }
    argparse_from_dicts([train_hparams, model_hparams])

    multi_output_df_cnn(train_hparams, model_hparams)


#####
##### END   src/experiments/multi_output_df_cnn.py
#####

##### 
##### ./kaggle_compile.py src/experiments/multi_output_df_cnn.py --save
##### 
##### 2020-03-13 04:09:23+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master 4086a2c [ahead 2] ./kaggle_compile.py | src/experiments/multi_output_df_cnn.py
##### 
##### 4086a2c7192388dd362105bf3e53d2f170a95aa0
##### 
##### Wrote: ./data_output/scripts/multi_output_df_cnn.py