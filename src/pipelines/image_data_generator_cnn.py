#!/usr/bin/env python
import os
import time

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


def image_data_generator_cnn(train_hparams, model_hparams, pipeline_name):
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
    model = MultiOutputCNN(
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
    flow_args = {}
    flow_args['train'] = {
        "transform_X":      Transforms.transform_X,
        "transform_X_args": {},  #  "normalize": True is default in Transforms
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
        "valid": ParquetImageDataGenerator(),
        "test":  ParquetImageDataGenerator(),
    }
    # [ datagens[key].fit(train_batch) for key in datagens.keys() ]  # Not required
    fileglobs = {
        "train": f"{settings['dir']['data']}/train_image_data_[123].parquet",
        "valid": f"{settings['dir']['data']}/train_image_data_0.parquet",
        "test":  f"{settings['dir']['data']}/test_image_data_*.parquet",
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        # For the Kaggle Submission, train on all available data and rely on Kaggle Timeout
        fileglobs["train"] = f"{settings['dir']['data']}/train_image_data_*.parquet"

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
    callback         = callbacks(train_hparams, dataset, model_file, log_dir, best_only=True, verbose=1)

    timer_start = time.time()
    history = model.fit(
        generators['train'],
        validation_data  = generators['valid'],
        epochs           = train_hparams['epochs'],
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
        "best_only":     True,
        "batch_size":    128,     # Too small and the GPU is waiting on the CPU - too big and GPU runs out of RAM - keep it small for kaggle
        "patience":      10,
        "epochs":        99,
        "loss_weights":  False,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['epochs']   = 0
    train_hparams = { **settings['hparam_defaults'], **train_hparams }

    argparse_from_dicts([train_hparams, model_hparams], inplace=True)


    pipeline_name     = "image_data_generator_cnn"
    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)
    logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.log"
    csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.csv"

    model, model_stats, output_shape = image_data_generator_cnn(train_hparams, model_hparams, pipeline_name)

    log_model_stats(model_stats, logfilename, model_hparams, train_hparams)

    submission = submission_df_generator(model, output_shape)
    df_to_submission_csv( submission, csv_filename )
