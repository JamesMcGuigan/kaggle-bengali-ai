from typing import Union, Dict

import simplejson


def model_stats_from_history(history, timer_seconds=0, best_only=False) -> Union[None, Dict]:
    if 'val_loss' in history.history:
        best_epoch            = history.history['val_loss'].index(min( history.history['val_loss'] )) if best_only else -1
        model_stats           = { key: value[best_epoch] for key, value in history.history.items() }
        model_stats['time']   = timer_seconds
        model_stats['epochs'] = len(history.history['loss'])
    else:
        model_stats = None
    return model_stats


def log_model_stats(model_stats, logfilename, model_hparams, train_hparams):
    with open(logfilename, 'w') as file:
        output = [
            "------------------------------",
            f"Completed",
            f"model_hparams: {model_hparams}",
            f"train_hparams: {train_hparams}",
        ]
        if isinstance(model_stats, dict):
            simplejson.dumps(
                { key: str(value) for key, value in model_stats.items() },
                sort_keys=False, indent=4*' '
            )
        elif isinstance(model_stats, list):
            output += [ "\n".join([ str(line) for line in model_stats ]) ]
        else:
            output += [ str(model_stats) ]

        output += [
            "------------------------------",
        ]
        output = "\n".join(output)
        print(      output )
        file.write( output )
        print("wrote:", logfilename)
