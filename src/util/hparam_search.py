import atexit
import itertools
import os
import shutil
from typing import Dict, List, AnyStr

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from src.dataset.DatasetDF import DatasetDF
from src.util import hparam


def hparam_options_length(hparam_options_value) -> int:
    try:
        if isinstance(hparam_options_value, (dict,list)): return len(hparam_options_value)
        else:                                             return len(hparam_options_value.values)
    except: return 1


def hparam_run_name(hparams: Dict, hparam_options: Dict) -> AnyStr:
    return "_".join([
        f"{key}={value}"
        for key, value in sorted(hparams.items())
        if key in hparam_options and hparam_options_length(hparam_options[key]) >= 2
    ])



def hparam_logdir(hparams: Dict, hparam_options: Dict, log_dir: AnyStr) -> AnyStr:
    key_name = "-".join([
        f"{key}"
        for key, value in sorted(hparams.items())
        if key in hparam_options
           and not str(key).startswith('~')  # exclude ~random
           and hparam_options_length(hparam_options[key]) >= 2
    ])
    run_name = hparam_run_name(hparams, hparam_options)
    dir_name = os.path.join(log_dir, key_name, run_name)
    return dir_name



# https://riptutorial.com/python/example/10160/all-combinations-of-dictionary-values
def hparam_combninations(hparam_options: Dict) -> List[Dict]:

    def get_hparam_options_values(key):
        if isinstance(hparam_options[key], dict):        return hparam_options[key].keys()
        if isinstance(hparam_options[key], list):        return hparam_options[key]
        if isinstance(hparam_options[key], hp.Discrete): return hparam_options[key].values
        else:                                            return [ hparam_options[key] ]     # Single value

    keys = hparam_options.keys()
    values = [ get_hparam_options_values(key) for key in keys ]

    hparams_list = [dict(zip(keys, combination)) for combination in itertools.product(*values)]  # generate combinations
    hparams_list = [dict(s) for s in set(frozenset(d.items()) for d in hparams_list)]            # unique

    # Merge dictionary options into hparams_list, after generating unique combinations
    lookup_keys = [ key for key in keys if isinstance(hparam_options[key], dict) ]
    for index, hparams in enumerate(hparams_list):
        for lookup_key in lookup_keys:
            if lookup_key in hparams:
                defaults = hparam_options[lookup_key][ hparams[lookup_key] ].copy()
                defaults.update(hparams_list[index])
                hparams_list[index] = defaults

    # random.shuffle(hparams_list)
    return hparams_list



def hparam_search(
        hparam_options: Dict,
        model:     tf.keras.Model,
        dataset:   'DatasetDF',
        log_root = None,
        verbose  = False,
        debug    = False
) -> List[Dict]:
    def onexit(log_dir):
        print('Ctrl-C KeyboardInterrupt')
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)  # remove logs for incomplete trainings
            print(f'rm -rf {log_dir}')

    model_config    = model.get_config()
    combninations   = hparam_combninations(hparam_options)
    logdir          = hparam_logdir(combninations[0], hparam_options, log_root)
    stats_history   = []

    # print(f"--- Model Config: ", model_config)
    print(f"--- Testing {len(combninations)} combinations in {logdir}")
    print(f"--- hparam_options: ", hparam_options)
    for index, hparams in enumerate(combninations):
        run_name = hparam_run_name(hparams, hparam_options)
        logdir   = hparam_logdir(hparams, hparam_options, log_root)

        print("")
        print(f"--- Starting trial {index+1}/{len(combninations)}: {logdir.split('/')[-2]} | {run_name}")
        print(hparams)
        if os.path.exists(logdir):
            print('Exists: skipping')
            continue
        if debug: continue

        atexit.register(onexit, logdir)

        # DOCS: https://www.tensorflow.org/guide/keras/save_and_serialize
        if model_config['name'] == 'sequential': model_clone = tf.keras.Sequential.from_config(model_config)
        else:                                    model_clone = tf.keras.Model.from_config(model_config)

        stats = hparam.model_compile_fit(hparams, model_clone, dataset, log_dir=logdir, verbose=verbose)
        stats_history += stats
        print(stats)

        atexit.unregister(onexit)

    print("")
    print("--- Stats History")
    print(stats_history)
    print("--- Finished")

    return stats_history