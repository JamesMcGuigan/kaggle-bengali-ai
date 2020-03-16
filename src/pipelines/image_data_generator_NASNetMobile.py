#!/usr/bin/env python
import os
import time

import glob2
from pyarrow.parquet import ParquetFile

from src.dataset.DatasetDF import DatasetDF
from src.dataset.ParquetImageDataGenerator import ParquetImageDataGenerator
from src.dataset.Transforms import Transforms
from src.models.MultiOutputApplication import MultiOutputApplication
from src.settings import settings
from src.util.argparse import argparse_from_dicts
from src.util.csv import df_to_submission_csv, submission_df
from src.util.hparam import hparam_key, model_compile, model_stats_from_history, callbacks
from src.util.logs import log_model_stats


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging


def image_data_generator_application(train_hparams, model_hparams, pipeline_name):
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

    dataset_rows = ParquetFile(f"{settings['dir']['data']}/train_image_data_0.parquet").metadata.num_rows
    dataset      = DatasetDF(size=1)
    input_shape  = dataset.input_shape()
    output_shape = dataset.output_shape()
    model = MultiOutputApplication(
        input_shape=input_shape,
        output_shape=output_shape,
        **model_hparams,
    )
    model_compile(model_hparams, model, output_shape)

    # Load Pre-existing weights
    if os.path.exists( model_file ):
        try:
            model.load_weights( model_file )
            print('Loaded Weights: ', model_file)
        except Exception as exception: print('exception', exception)

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        load_models = glob2.glob(f'../input/**/{model_file}')
        for load_model in load_models:
            try:
                model.load_weights( load_model )
                print('Loaded Weights: ', load_model)
                break
            except Exception as exception: print('exception', exception)

    model.summary()


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing
    datagen_args = {
        "rescale":            1./255,
        "zoom_range":         0.2,
        "width_shift_range":  0.1,    # we already have centering
        "height_shift_range": 0.1,    # we already have centering
        "rotation_range":     45/2,
        "shear_range":        45/2,
        # "brightness_range":   0.5,  # Prebrightness normalized
        "fill_mode":         'constant',
        "cval": 0,
        # "featurewise_center": True,             # No visible effect in plt.imgshow()
        # "samplewise_center": True,              # No visible effect in plt.imgshow()
        # "featurewise_std_normalization": True,  # No visible effect in plt.imgshow() | requires .fit()
        # "samplewise_std_normalization": True,   # No visible effect in plt.imgshow() | requires .fit()
        # "zca_whitening": True,                   # Kaggle, insufficent memory
    }
    flow_args = {}
    flow_args['train'] = {
        "transform_X":      Transforms.transform_X,
        "transform_X_args": { "normalize": False },
        "transform_Y":      Transforms.transform_Y,
        "batch_size":       train_hparams['batch_size'],
        "reads_per_file":   3,
        "resamples":        1,
        "shuffle":          True,
        "infinite":         True,
    }
    flow_args['valid'] = {
        **flow_args['train'],
        "resamples":  1,
    }
    flow_args['test'] = {
        **flow_args['train'],
        "resamples":  1,
        "shuffle":    False,
        "infinite":   False,
        "test":       True,
    }

    datagens = {
        "train": ParquetImageDataGenerator(**datagen_args),
        "valid": ParquetImageDataGenerator(**datagen_args),
        "test":  ParquetImageDataGenerator(rescale=1./255),
    }
    # [ datagens[key].fit(train_batch) for key in datagens.keys() ]  # Not required
    generators = {
        "train": datagens['train'].flow_from_parquet(f"{settings['dir']['data']}/train_image_data_[123].parquet", **flow_args['train' ]),
        "valid": datagens['valid'].flow_from_parquet(f"{settings['dir']['data']}/train_image_data_0.parquet",     **flow_args['valid']),
        "test":  datagens['test' ].flow_from_parquet(f"{settings['dir']['data']}/test_image_data_*.parquet",      **flow_args['test']),
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        # For the Kaggle Submission, train on all available data and rely on Kaggle Timeout
        generators["train"] = datagens['train'].flow_from_parquet(f"{settings['dir']['data']}/train_image_data_*.parquet", **flow_args['train' ])

    callback = callbacks(train_hparams, dataset, model_file, log_dir, best_only=True, verbose=1)


    timer_start = time.time()
    # train == 1 parquet file | valid = 1 file read
    steps_per_epoch  = int(dataset_rows / flow_args['train']['batch_size'] * flow_args['train']['resamples'])
    validation_steps = int(dataset_rows / flow_args['valid']['batch_size'] / flow_args['train']['reads_per_file'])
    history = model.fit(
        generators['train'],
        validation_data = generators['valid'],
        epochs           = 30,
        steps_per_epoch  = steps_per_epoch,
        validation_steps = validation_steps,
        verbose          = 2,
        callbacks        = callback
    )
    timer_seconds = int(time.time() - timer_start)
    model_stats   = model_stats_from_history(history, timer_seconds, best_only=True)

    return model, model_stats, output_shape




if __name__ == '__main__':
    model_hparams = {
        "application":  "NASNetMobile",
        "pooling":      'avg',  # None, 'avg', 'max',
        "dense_units":   512,   # != (1295+168+11+7),
        "dense_layers":  2,
    }
    train_hparams = {
        "optimizer":     "RMSprop",
        "scheduler":     "constant",
        "learning_rate": 0.001,
        "best_only":     True,
        "batch_size":    32,    # Too small and the GPU is waiting on the CPU - too big and GPU runs out of RAM
        "patience":      10,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['loops']    = 1
    train_hparams = { **settings['hparam_defaults'], **train_hparams }

    argparse_from_dicts([train_hparams, model_hparams], inplace=True)


    pipeline_name     = "image_data_generator_application_NASNetMobile"
    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)
    logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.log"
    csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.csv"

    model, model_stats, output_shape = image_data_generator_application(train_hparams, model_hparams, pipeline_name)

    log_model_stats(model_stats, logfilename, model_hparams, train_hparams)

    submission = submission_df(model, output_shape)
    df_to_submission_csv( submission, csv_filename )