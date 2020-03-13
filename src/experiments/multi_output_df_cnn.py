import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.dataset.DatasetDF import DatasetDF
from src.models.MultiOutputCNN import MultiOutputCNN
# from src.util.hparam import model_compile_fit
from src.settings import settings
from src.util.argparse import argparse_from_dicts
from src.util.csv import df_to_submission_csv

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
                epochs=1,
                verbose=2,
                validation_data=(dataset.X["valid"], dataset.Y["valid"]),
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        verbose=True,
                        patience=hparams.get('patience'),
                        restore_best_weights=True
                    )
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
