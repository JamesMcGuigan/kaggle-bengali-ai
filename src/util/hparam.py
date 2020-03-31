import math
import os
import re
import time
from typing import AnyStr, Dict, Union

import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

from src.callbacks.KaggleTimeoutCallback import KaggleTimeoutCallback
from src.dataset.DatasetDF import DatasetDF
from src.settings import settings
from src.util.logs import model_stats_from_history
from src.vendor.CLR.clr_callback import CyclicLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging
# tf.keras.backend.set_floatx('float16')  # Potentially causes problems with Tensortflow


def hparam_key(hparams):
    return "-".join( f"{key}={value}" for key,value in sorted(hparams.items()) ).replace(' ','')


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


def losses(output_shape):
    if   isinstance(output_shape, list): losses = [ categorical_crossentropy      for n   in output_shape        ]
    elif isinstance(output_shape, dict): losses = { key: categorical_crossentropy for key in output_shape.keys() }
    else:                                losses = categorical_crossentropy
    return losses


def loss_weights(output_shape):
    # unique = dataset.apply(lambda col: col.nunique()); unique
    # grapheme_root           168   | sqrt = 12.9 / 54.9 = 0.24
    # vowel_diacritic          11   | sqrt =  3.3 / 54.9 = 0.06
    # consonant_diacritic       7   | sqrt =  2.6 / 54.9 = 0.05
    # grapheme               1295   | sqrt = 35.9 / 54.9 = 0.65
    if not isinstance(output_shape, dict): return None
    norm    = sum(map(math.sqrt, output_shape.values()))
    weights = {
        key: math.sqrt(value)/norm
        for key,value in output_shape.items()
    }
    return weights



def callbacks(hparams, dataset, model_file=None, log_dir=None, best_only=True, verbose=False ):
    schedule  = scheduler(hparams, dataset, verbose=verbose)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=verbose,
            patience=hparams.get('patience', 10),
            restore_best_weights=best_only
        ),
        schedule,
    ]
    if hparams.get("timeout"):
        callbacks += [
            KaggleTimeoutCallback( hparams.get("timeout"), verbose=False ),
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
    if log_dir and settings['verbose']['tensorboard'] and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        callbacks += [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            KerasCallback(log_dir, hparams)                                     # log train_hparams
        ]
    return callbacks



def model_compile(
        hparams:      Dict,
        model:        tf.keras.models.Model,
        output_shape: Union[None, int, Dict] = None,
    ):
    hparams   = { **settings['hparam_defaults'], **hparams }
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    loss      = losses(output_shape)
    weights   = loss_weights(output_shape) if hparams.get('loss_weights') else None

    model.compile(
        loss=loss,
        loss_weights=weights,
        optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.001)),
        metrics=['accuracy']
    )
    return model


def model_compile_fit(
        hparams:      Dict,
        model:        tf.keras.models.Model,
        dataset:      DatasetDF,
        epochs      = 999,
        output_shape: Union[None, int, Dict] = None,
        model_file:   AnyStr = None,
        log_dir:      AnyStr = None,
        best_only   = True,
        verbose     = settings['verbose']['fit'],
):
    timer_start = time.time()

    hparams = { **settings['hparam_defaults'], **hparams }
    model   = model_compile( hparams, model, output_shape )

    callback = callbacks(hparams, dataset, model_file, log_dir, best_only, verbose)
    history  = model.fit(
        dataset.X["train"], dataset.Y["train"],
        batch_size=hparams.get("batch_size", 128),
        epochs=epochs,
        verbose=verbose,
        validation_data=(dataset.X["valid"], dataset.Y["valid"]),
        callbacks=callback
    )
    timer_seconds = int(time.time() - timer_start)

    model_stats = model_stats_from_history(history, timer_seconds, best_only)
    return model_stats
