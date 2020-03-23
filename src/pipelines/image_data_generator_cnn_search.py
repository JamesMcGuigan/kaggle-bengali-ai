#!/usr/bin/env python

import atexit
import os
import shutil

import tensorflow as tf

from src.pipelines.image_data_generator_cnn import image_data_generator_cnn
from src.settings import settings
from src.util.argparse import argparse_from_dict
from src.util.hparam import hparam_key
from src.util.hparam_search import hparam_combninations, hparam_logdir, hparam_run_name
from src.util.logs import log_model_stats


# remove logs and models for incomplete trainings
def onexit(outputs: list):
    print('Ctrl-C KeyboardInterrupt')
    for output in outputs:
        if os.path.exists(output):
            if os.path.isdir(output):
                shutil.rmtree(output)
                print(f'rm -rf {output}')
            else:
                os.remove(output)
                print(f'rm -f {output}')


def image_data_generator_cnn_search(
        debug=0,
        verbose=0,
):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging
    tf.keras.backend.set_floatx('float32')    # BUGFIX: Nan in summary histogram

    model_hparam_options = {
        "cnns_per_maxpool":   3,
        "maxpool_layers":     4,
        "dense_layers":       2,
        "dense_units":      256,
        "regularization": False,
        "global_maxpool": False,

        # "cnns_per_maxpool":   [2,3,4],
        # "maxpool_layers":     [4,5],
        # "dense_layers":       [1,2,3],
        # "dense_units":        [256],
        # "regularization":     [False], # [True,False],
        # "global_maxpool":     [False], # [True,False],
    }
    train_hparams_search = {
        "optimized_scheduler": {
            "Adagrad_triangular": { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "triangular"  },
            "Adagrad_plateau":    { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "plateau2"    },
            "Adam_triangular2":   { "learning_rate": 0.01,   "optimizer": "Adam",     "scheduler": "triangular2" },
            "Nadam_plateau":      { "learning_rate": 0.01,   "optimizer": "Nadam",    "scheduler": "plateau10"   },
            "Adadelta_plateau":   { "learning_rate": 1.0,    "optimizer": "Adadelta", "scheduler": "plateau10"   },
            "SGD_triangular2":    { "learning_rate": 1.0,    "optimizer": "SGD",      "scheduler": "triangular2" },
            "RMSprop_constant":   { "learning_rate": 0.001,  "optimizer": "RMSprop",  "scheduler": "constant"    },
        },
        # "optimizer":     "RMSprop",
        # "scheduler":     "constant",
        # "learning_rate": 0.001,
        # "best_only":     True,
        # "batch_size":    128,     # Too small and the GPU is waiting on the CPU - too big and GPU runs out of RAM - keep it small for kaggle
        # "patience":      10,
        # "epochs":        99,
        # "loss_weights":  False,
        # "timeout":       "6h"
    }
    model_combninations = hparam_combninations(model_hparam_options)
    train_combninations = hparam_combninations(train_hparams_search)
    combninations_count = len(model_combninations)*len(train_combninations)
    stats_history   = []

    if   len(train_combninations) >= 2 and len(model_combninations) == 1:
        pipeline_name = "image_data_generator_cnn_search_train"
    elif len(train_combninations) == 1 and len(model_combninations) >= 2:
        pipeline_name = "image_data_generator_cnn_search_model"
    else:
        pipeline_name = "image_data_generator_cnn_search_train_model"


    print(f"--- Testing {combninations_count} combinations")
    print(f"--- model_hparam_options: ", model_hparam_options)

    index = 0
    for model_hparams in model_combninations:
        for train_hparams in train_combninations:
            index += 1
            run_name = hparam_run_name(model_hparams, model_hparam_options)
            logdir   = hparam_logdir(model_hparams, model_hparam_options, settings['dir']['logs'])

            print("")
            print(f"--- Starting trial {index}/{combninations_count}: {logdir.split('/')[-2]} | {run_name}")
            print(model_hparams)
            print(train_hparams)

            model_hparams_key = hparam_key(model_hparams)
            train_hparams_key = hparam_key(train_hparams)
            logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}/{model_hparams_key}-{train_hparams_key}-submission.log"
            csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}/{model_hparams_key}-{train_hparams_key}-submission.csv"
            model_file        = f"{settings['dir']['models']}/{pipeline_name}/{model_hparams_key}/{train_hparams_key}/{model_hparams_key}-{train_hparams_key}.hdf5"
            log_dir           = f"{settings['dir']['logs']}/{pipeline_name}/{model_hparams_key}/{train_hparams_key}"

            for dirname in [ log_dir ] + list(map(os.path.dirname, [logfilename, csv_filename, model_file])):
                os.makedirs(dirname, exist_ok=True)

            if os.path.exists(model_file):
                print('Exists: skipping')
                continue
            if debug: continue
            atexit.register(onexit, [logfilename, csv_filename, model_file, log_dir])

            model, model_stats, output_shape = image_data_generator_cnn(
                train_hparams = train_hparams,
                model_hparams = model_hparams,
                pipeline_name = pipeline_name,
                model_file    = model_file,
                log_dir       = log_dir,
                verbose       = verbose,
            )

            log_model_stats(model_stats, logfilename, model_hparams, train_hparams)
            # submission = submission_df_generator(model, output_shape)
            # df_to_submission_csv( submission, csv_filename )
            stats_history += model_stats
            print(model_stats)

            atexit.unregister(onexit)

    print("")
    print("--- Stats History")
    print(stats_history)
    print("--- Finished")

    return stats_history

if __name__ == '__main__':
    argv = argparse_from_dict({ "debug": 0, "verbose": 0 })
    image_data_generator_cnn_search(**argv)