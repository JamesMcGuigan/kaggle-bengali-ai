#!/usr/bin/env python

import atexit
import os
import shutil

from src.pipelines.image_data_generator_cnn import image_data_generator_cnn
from src.settings import settings
from src.util.csv import submission_df_generator, df_to_submission_csv
from src.util.hparam import hparam_key
from src.util.hparam_search import hparam_combninations, hparam_logdir, hparam_run_name
from src.util.logs import log_model_stats


def onexit(log_dir):
    print('Ctrl-C KeyboardInterrupt')
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)  # remove logs for incomplete trainings
        print(f'rm -rf {log_dir}')


def image_data_generator_cnn_search(
        debug=False,
):
    model_hparam_options = {
        "cnns_per_maxpool":   [2,3,4],
        "maxpool_layers":     [4,5],
        "dense_layers":       [1,2,3],
        "dense_units":        [256],
        "regularization":     [False], # [True,False],
        "global_maxpool":     [False], # [True,False],
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
    combninations   = hparam_combninations(model_hparam_options)
    logdir          = hparam_logdir(combninations[0], model_hparam_options, settings['dir']['logs'])
    stats_history   = []

    print(f"--- Testing {len(combninations)} combinations in {logdir}")
    print(f"--- model_hparam_options: ", model_hparam_options)
    for index, model_hparams in enumerate(combninations):
        run_name = hparam_run_name(model_hparams, model_hparam_options)
        logdir   = hparam_logdir(model_hparams, model_hparam_options, settings['dir']['logs'])

        print("")
        print(f"--- Starting trial {index+1}/{len(combninations)}: {logdir.split('/')[-2]} | {run_name}")
        print(model_hparams)
        if os.path.exists(logdir):
            print('Exists: skipping')
            continue
        if debug: continue

        atexit.register(onexit, logdir)

        pipeline_name     = "image_data_generator_cnn"
        model_hparams_key = hparam_key(model_hparams)
        train_hparams_key = hparam_key(train_hparams)
        logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.log"
        csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}-{model_hparams_key}-submission.csv"

        model, model_stats, output_shape = image_data_generator_cnn(train_hparams, model_hparams, pipeline_name)

        log_model_stats(model_stats, logfilename, model_hparams, train_hparams)
        submission = submission_df_generator(model, output_shape)
        df_to_submission_csv( submission, csv_filename )
        stats_history += model_stats
        print(model_stats)

        atexit.unregister(onexit)

    print("")
    print("--- Stats History")
    print(stats_history)
    print("--- Finished")

    return stats_history

if __name__ == '__main__':
    image_data_generator_cnn_search(debug=False)