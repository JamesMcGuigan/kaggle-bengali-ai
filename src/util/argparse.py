import argparse
from typing import List, Dict


def argparse_from_dicts(configs: List[Dict]):
    parser = argparse.ArgumentParser()
    for config in list(configs):
        for key, value in config.items():
            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'{key} (default: %(default)s)')

    args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle
    for config in list(configs):
        for key, value in config.items():
            config[key] = getattr(args, key)
