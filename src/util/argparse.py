import argparse
import copy
from typing import Dict, List



def argparse_from_dicts(configs: List[Dict], inplace=False) -> List[Dict]:
    parser = argparse.ArgumentParser()
    for config in list(configs):
        for key, value in config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f'{key} (default: %(default)s)')
            else:
                parser.add_argument(f'--{key}', type=type(value),    default=value, help=f'{key} (default: %(default)s)')


    args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle

    outputs = configs if inplace else copy.deepcopy(configs)
    for index, output in enumerate(outputs):
        for key, value in outputs[index].items():
            outputs[index][key] = getattr(args, key)

    return outputs


def argparse_from_dict(config: Dict, inplace=False):
    return argparse_from_dicts([config], inplace)[0]
