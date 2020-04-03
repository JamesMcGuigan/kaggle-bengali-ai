#!/usr/bin/env python
import os
import time
from collections import ChainMap
from typing import Dict

import glob2
import numpy as np
from pyarrow.parquet import ParquetFile

from src.dataset.DatasetDF import DatasetDF
from src.dataset.ParquetImageDataGenerator import ParquetImageDataGenerator
from src.dataset.Transforms import Transforms
from src.models.MultiOutputCNN import MultiOutputCNN
from src.settings import settings
from src.util.argparse import argparse_from_dicts
from src.util.csv import df_to_submission_csv, submission_df_generator
from src.util.hparam import callbacks, hparam_key, model_compile, model_stats_from_history
from src.util.logs import log_model_stats


def image_data_generator_cnn(
        train_hparams:    Dict,
        model_hparams:    Dict,
        transform_X_args: Dict,
        transform_Y_args: Dict,
        datagen_args:     Dict,
        pipeline_name  = 'image_data_generator_cnn',
        model_file     = None,
        log_dir        = None,
        verbose        = 2,
        load_weights   = True
):
    combined_hparams = { **model_hparams, **train_hparams, **transform_X_args, **transform_Y_args, **datagen_args }
    train_hparams    = { **settings['hparam_defaults'], **train_hparams }
    if verbose:
        print('-----')
        print("pipeline_name",    pipeline_name)
        print("train_hparams",    train_hparams)
        print("transform_X_args", transform_X_args)
        print("transform_Y_args", transform_Y_args)
        print("datagen_args",     datagen_args)
        print("model_file",       model_file)
        print("log_dir",          log_dir)
        print("load_weights",     load_weights)
        print('-----')

    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)
    transform_key     = hparam_key(ChainMap(*[ transform_X_args, transform_Y_args, datagen_args ]))

    # csv_data    = pd.read_csv(f"{settings['dir']['data']}/train.csv")
    model_file  = model_file or f"{settings['dir']['models']}/{pipeline_name}/{pipeline_name}-{model_hparams_key}.hdf5"
    log_dir     = log_dir    or f"{settings['dir']['logs']}/{pipeline_name}/{transform_key}/"

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(log_dir,                     exist_ok=True)

    dataset_rows = ParquetFile(f"{settings['dir']['data']}/train_image_data_0.parquet").metadata.num_rows
    dataset      = DatasetDF(
        size=1,
        transform_X_args=transform_X_args,
        transform_Y_args=transform_Y_args,
    )
    input_shape  = dataset.input_shape()
    output_shape = dataset.output_shape()
    model = MultiOutputCNN(
        input_shape=input_shape,
        output_shape=output_shape,
        **model_hparams,
    )
    model_compile(model_hparams, model, output_shape)

    # Load Pre-existing weights
    if load_weights:
        if os.path.exists( model_file ):
            try:
                model.load_weights( model_file )
                print('Loaded Weights: ', model_file)
            except Exception as exception: print('exception', exception)

        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
            load_models = (glob2.glob(f'../input/**/{os.path.basename(model_file)}')
                        +  glob2.glob(f'../input/**/{os.path.basename(model_file)}'.replace('=','')))  # Kaggle Dataset Upload removes '='
            for load_model in load_models:
                try:
                    model.load_weights( load_model )
                    print('Loaded Weights: ', load_model)
                    # break
                except Exception as exception: print('exception', exception)

    if verbose:
        model.summary()

    flow_args = {}
    flow_args['train'] = {
        "transform_X":      Transforms.transform_X,
        "transform_Y":      Transforms.transform_Y,
        "transform_X_args": transform_X_args,
        "transform_Y_args": transform_Y_args,
        "batch_size":       train_hparams['batch_size'],
        "reads_per_file":   2,
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
        "valid": ParquetImageDataGenerator(),
        "test":  ParquetImageDataGenerator(),
    }
    # [ datagens[key].fit(train_batch) for key in datagens.keys() ]  # Not required
    fileglobs = {
        "train": f"{settings['dir']['data']}/train_image_data_[123].parquet",
        "valid": f"{settings['dir']['data']}/train_image_data_0.parquet",
        "test":  f"{settings['dir']['data']}/test_image_data_*.parquet",
    }
    ### Preserve test/train split for Kaggle
    # if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    #     # For the Kaggle Submission, train on all available data and rely on Kaggle Timeout
    #     fileglobs["train"] = f"{settings['dir']['data']}/train_image_data_*.parquet"

    generators = {
        key: datagens[key].flow_from_parquet(value, **flow_args[key])
        for key,value in fileglobs.items()
    }
    dataset_rows_per_file = {
        key: np.mean([ ParquetFile(filename).metadata.num_rows for filename in glob2.glob(fileglobs[key]) ])
        for key in fileglobs.keys()
    }
    dataset_rows_total = {
        key: sum([ ParquetFile(filename).metadata.num_rows for filename in glob2.glob(fileglobs[key]) ])
        for key in fileglobs.keys()
    }

    ### Epoch: train == one whole parquet files | valid = 1 filesystem read
    steps_per_epoch  = int(dataset_rows_per_file['train'] / flow_args['train']['batch_size'] * flow_args['train']['resamples'] )
    validation_steps = int(dataset_rows_per_file['valid'] / flow_args['valid']['batch_size'] / flow_args['train']['reads_per_file'] )
    callback         = callbacks(combined_hparams, dataset, model_file, log_dir, best_only=True, verbose=1)

    timer_start = time.time()
    history = model.fit(
        generators['train'],
        validation_data  = generators['valid'],
        epochs           = train_hparams['epochs'],
        steps_per_epoch  = steps_per_epoch,
        validation_steps = validation_steps,
        verbose          = verbose,
        callbacks        = callback
    )
    timer_seconds = int(time.time() - timer_start)
    model_stats   = model_stats_from_history(history, timer_seconds, best_only=True)

    return model, model_stats, output_shape




if __name__ == '__main__':
    # Fastest with high score
    # - maxpool_layers=5 | cnns_per_maxpool=3 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=False
    #
    # Shortlist:
    # - maxpool_layers=5 | cnns_per_maxpool=3 | dense_layers=1 | dense_units=512 | global_maxpool=True  | regularization=False
    # - maxpool_layers=4 | cnns_per_maxpool=4 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=False
    # - maxpool_layers=4 | cnns_per_maxpool=4 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=True
    hparams = {}
    hparams['model'] = {
        "cnns_per_maxpool":   3,
        "maxpool_layers":     5,
        "cnn_units":         32,
        "cnn_kernel":         3,
        "cnn_strides":        1,
        "dense_layers":       1,
        "dense_units":      256,
        # "regularization": False,  # Produces worse results
        # "global_maxpool": False,  #
        "activation":    'relu',  # 'relu' | 'crelu' | 'leaky_relu' | 'relu6' | 'softmax' | 'tanh' | 'hard_sigmoid' | 'sigmoid'
        "dropout":         0.25,
    }
    hparams['transform_X'] = {
        "resize":       2,
        # "invert":    True,
        "rescale":   True,
        "denoise":   True,
        "center":    True,
        # "normalize": True,
    }
    hparams['transform_Y'] = {
    }

    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing
    hparams['datagen'] = {
        # "rescale":          1./255,  # "normalize": True is default in Transforms
        "zoom_range":         0.2,
        "width_shift_range":  0.1,     # we already have centering
        "height_shift_range": 0.1,     # we already have centering
        "rotation_range":     45/2,
        "shear_range":        45/2,
        # "brightness_range":   0.5,   # Prebrightness normalized
        "fill_mode":         'constant',
        "cval": 0,
        # "featurewise_center": True,             # No visible effect in plt.imgshow()
        # "samplewise_center": True,              # No visible effect in plt.imgshow()
        # "featurewise_std_normalization": True,  # No visible effect in plt.imgshow() | requires .fit()
        # "samplewise_std_normalization": True,   # No visible effect in plt.imgshow() | requires .fit()
        # "zca_whitening": True,                   # Kaggle, insufficent memory
    }
    hparams['train'] = {
        "optimizer":     "Adadelta",
        "scheduler":     "plateau10",
        "learning_rate": 1,
        "patience":      20,
        "best_only":     True,
        "batch_size":    128,     # Too small and the GPU is waiting on the CPU - too big and GPU runs out of RAM - keep it small for kaggle
        "epochs":        999,
        "loss_weights":  False,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        hparams['train']['patience'] = 0
        hparams['train']['epochs']   = 1
    hparams['train'] = { **settings['hparam_defaults'], **hparams['train'] }

    argparse_from_dicts(list(hparams.values()), inplace=True)


    pipeline_name         = "image_data_generator_cnn"
    hparams_model_key     = hparam_key(hparams['model'])
    hparams_transform_key = hparam_key(ChainMap(*[ hparams['transform_X'], hparams['transform_Y'], hparams['datagen'] ]))
    logfilename           = f"{settings['dir']['submissions']}/{pipeline_name}/{hparams_transform_key}/{hparams_model_key}-submission.log"
    csv_filename          = f"{settings['dir']['submissions']}/{pipeline_name}/{hparams_transform_key}/{hparams_model_key}-submission.csv"

    model, model_stats, output_shape = image_data_generator_cnn(
        train_hparams    = hparams['train'],
        model_hparams    = hparams['model'],
        transform_X_args = hparams['transform_X'],
        transform_Y_args = hparams['transform_Y'],
        datagen_args     = hparams['datagen'],
        pipeline_name    = pipeline_name,
        load_weights     = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
    )

    log_model_stats(model_stats, logfilename, hparams)

    submission = submission_df_generator(model, output_shape)
    df_to_submission_csv( submission, csv_filename )
