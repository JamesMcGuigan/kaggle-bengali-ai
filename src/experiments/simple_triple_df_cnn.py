# This is a first pass, simplest thing that could possibly work attempt
# We train three separate MINST style CNNs for each label, then combine the results
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3  # Disable Tensortflow Logging

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset.DatasetDF import DatasetDF
from src.models.SingleOutputCNN import SingleOutputCNN
from src.util.argparse import argparse_from_dicts
from src.util.csv import df_to_submission_csv
from src.util.hparam import model_compile_fit


# https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
config  = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)



def simple_triple_df_cnn(dirs, train_hparams, model_hparams):
    model_hparams_key = "-".join( f"{key}={value}" for key,value in model_hparams.items() ).replace(' ','')
    print("train_hparams", train_hparams)
    print("model_hparams", model_hparams)


    csv_data    = pd.read_csv(f"{dirs['data']}/train.csv")
    models      = {}
    model_files = {}
    model_stats = {}
    output_fields = [ "consonant_diacritic", "grapheme_root", "vowel_diacritic" ]
    for output_field in output_fields:
        model_stats[output_field] = []
        model_files[output_field] = f"{dirs['models']}/SingleOutputCNN-{model_hparams_key}-{output_field}.hdf5"

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

        for loop in range(1):
            for data_id in range(0,1):
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
                    log_dir    = f"{dirs['logs']}/simple_triple_df_cnn/{output_field}/",
                    verbose    = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch',
                    best_only  = True,
                )
                model_stats[output_field].append(stats)

        print("------------------------------")
        print(f"Completed | {output_field}")
        print(f"model_hparams: {model_hparams}")
        print(f"train_hparams: {train_hparams}")
        for stats in model_stats[output_field]: print(stats)
        print("------------------------------")


    ### Log Stats Results
    logfilename = f"{dirs['predictions']}/SingleOutputCNN-{model_hparams_key}-submission.log"
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
        f"{dirs['predictions']}/SingleOutputCNN-{model_hparams_key}-submission.csv"
    )



if __name__ == '__main__':
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        dirs = {
            "data":        "../input/bengaliai-cv19",
            "models":      "./",
            "predictions": "./",
            "logs":        "./logs",
        }
    else:
        dirs = {
            "data":        "./input/bengaliai-cv19",
            "models":      "./output/models/simple_triple_df_cnn",
            "predictions": "./output/predictions/simple_triple_df_cnn",
            "logs":        "./logs",
        }
    for dirname in dirs.values(): os.makedirs(dirname, exist_ok=True)

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
    hparams = {
        # "optimizer":     "Adagrad",
        # "scheduler":     "plateau2",
        # "learning_rate": 0.1,
        "optimizer":     "RMSprop",
        "scheduler":     "constant",
        "learning_rate": 0.001,
        "min_lr":        0.001,
        "split":         0.2,
        "batch_size":    128,
        "patience":      5,
        "fraction":      1.0,
        "loops":         1,
    }
    argparse_from_dicts([ dirs, hparams, model_hparams  ])

    simple_triple_df_cnn(dirs, hparams, model_hparams)