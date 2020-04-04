import datetime
import os
from typing import Dict, Union

import simplejson
import time

from src.settings import settings


def model_stats_from_history(history, timer_seconds=0, best_only=False) -> Union[None, Dict]:
    if 'val_loss' in history.history:
        best_epoch            = history.history['val_loss'].index(min( history.history['val_loss'] )) if best_only else -1
        model_stats           = { key: value[best_epoch] for key, value in history.history.items() }
        model_stats['time']   = timer_seconds
        model_stats['epochs'] = len(history.history['loss'])
    else:
        model_stats = None
    return model_stats


python_start = time.time()
def log_model_stats(model_stats, logfilename, nested_hparams: Dict[str, Dict]):
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    with open(logfilename, 'w') as file:
        output = [
            "------------------------------",
            f"Completed",
            "------------------------------",
        ]
        output += [ f"{name}: {hparams}" for name, hparams in sorted(nested_hparams.items()) ]
        output.append("------------------------------")
        output += [ f"settings[{key}]: {value}" for key, value in sorted(settings.items()) ]
        output.append("------------------------------")

        if isinstance(model_stats, dict):
            output.append(simplejson.dumps(
                { key: str(value) for key, value in model_stats.items() },
                sort_keys=False, indent=4*' '
            ))
        elif isinstance(model_stats, list):
            output += list(map(str, model_stats))
        else:
            output.append( str(model_stats) )

        output.append("------------------------------")
        output += [
            f"------------------------------",
            f"script started: { datetime.datetime.fromtimestamp( python_start ).strftime('%Y-%m-%d %H:%M:%S')}",
            f"script ended:   { datetime.datetime.fromtimestamp( time.time()  ).strftime('%Y-%m-%d %H:%M:%S')}",
            f"script runtime: { str(datetime.timedelta(seconds=int(model_stats['time']))) }",
            f"------------------------------",
        ]
        output = "\n".join(output)
        print(      output )
        file.write( output )
        print("wrote:", logfilename)
