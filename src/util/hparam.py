import math
import re
import time
from typing import Dict, AnyStr

import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, \
    ModelCheckpoint

from src.dataset.DatasetDF import DatasetDF
from src.settings import settings
from vendor.CLR.clr_callback import CyclicLR


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


def model_compile_fit(
        hparams:    Dict,
        model:      tf.keras.models.Model,
        dataset:    DatasetDF,
        model_file: AnyStr = None,
        log_dir:    AnyStr = None,
        best_only   = True,
        verbose     = settings['verbose']['fit'],
):
    hparams   = { **settings['hparam_defaults'], **hparams }
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    schedule  = scheduler(hparams, dataset, verbose=verbose)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=verbose,
            patience=hparams.get('patience', hparams['patience']),
            restore_best_weights=best_only
        ),
        schedule,
        # ProgbarLogger(count_mode='samples', stateful_metrics=None)
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
    if log_dir:
        callbacks += [  
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            KerasCallback(log_dir, hparams)                                     # log train_hparams
        ]

    timer_start = time.time()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.0001)),
        metrics=['accuracy']
    )
    history = model.fit(
        dataset.X["train"], dataset.Y["train"],
        batch_size=hparams.get("batch_size", 128),
        epochs=999,
        verbose=verbose,
        validation_data=(dataset.X["valid"], dataset.Y["valid"]),
        callbacks=callbacks
    )
    timer_seconds = int(time.time() - timer_start)

    best_epoch            = history.history['val_loss'].index(min( history.history['val_loss'] )) if best_only else -1
    model_stats           = { key: value[best_epoch] for key, value in history.history.items() }
    model_stats['time']   = timer_seconds
    model_stats['epochs'] = len(history.history['loss'])

    return model_stats
