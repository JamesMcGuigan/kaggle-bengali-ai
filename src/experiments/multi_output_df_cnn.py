import os

import numpy as np
import pandas as pd

from src.dataset.DatasetDF import DatasetDF
from src.models.MultiOutputCNN import MultiOutputCNN
from src.settings import settings
from src.util.argparse import argparse_from_dicts
from src.util.csv import df_to_submission_csv
from src.util.hparam import model_compile_fit



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

