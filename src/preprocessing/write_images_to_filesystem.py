#!/usr/bin/env python
import copy
import os
import time

import glob2
import matplotlib
import matplotlib.image
import pandas as pd
from pyarrow.parquet import ParquetFile

from src.dataset.DatasetDF import DatasetDF
from src.settings import settings
from src.util.argparse import argparse_from_dicts



# Entries into the Bengali AI Competition often suffer from out of memory errors when reading from a dataframe
# Quick and dirty solution is to write data as images to a directory and use ImageDataGenerator.flow_from_directory()
def write_images_to_filesystem( data_dir, feature_dir, ext='png', only=None, verbose=False, force=False, transform_args={} ):
    transform_defaults = { 'resize': 2, 'denoise': True, 'center': True, 'invert': True, 'normalize': False }
    transform_args     = { **transform_defaults, **transform_args }

    time_start = time.time()
    filename_groups = {
        "test":  sorted(glob2.glob(f"{settings['dir']['data']}/test_image_data_*.parquet")),
        "train": sorted(glob2.glob(f"{settings['dir']['data']}/train_image_data_*.parquet")),
    }
    if only:
        for test_train_valid, parquet_filenames in list(filename_groups.items()):
            if only not in test_train_valid:
                del filename_groups[test_train_valid]

    image_count = 0
    for test_train_valid, parquet_filenames in filename_groups.items():
        image_dir = f'{feature_dir}/{test_train_valid}'
        os.makedirs(image_dir, exist_ok=True)

        # Skip image creation if all images have already been extracted
        if not force:
            expected_images = sum([ ParquetFile(file).metadata.num_rows for file in parquet_filenames ])
            existing_images = len(glob2.glob(f'{image_dir}/*.{ext}'))
            if existing_images == expected_images: continue

        for parquet_filename in parquet_filenames:
            if verbose >= 2:
                print(f'write_images_to_filesystem({only or ""}) - reading:  ', parquet_filename)

            dataframe  = pd.read_parquet(parquet_filename)
            image_ids  = dataframe['image_id'].tolist()
            image_data = DatasetDF.transform_X(dataframe, **transform_args )

            for index, image_id in enumerate(image_ids):
                image_filename = f'{image_dir}/{image_id}.{ext}'
                if not force and os.path.exists(image_filename):
                    print(f'write_images_to_filesystem({only or ""}) - skipping: ', image_filename)
                    continue

                matplotlib.image.imsave(image_filename, image_data[index].squeeze(), cmap='gray')
                image_count += 1
                if verbose:
                    print(f'write_images_to_filesystem({only or ""}) - wrote:    ', image_filename)

    if verbose >= 1:
        print( f'write_images_to_filesystem({only or ""}) - wrote: {image_count} files in: {round(time.time() - time_start,2)}s')



if __name__ == '__main__':
    args = {
        'data_dir':    settings['dir']['data'],
        'feature_dir': settings['dir']['features'],
        'ext':         'png',
        'verbose':      2,
        'force':        False,
    }
    transform_args = {
        'resize':    2,
        'denoise':   1,
        'center':    1,
        'invert':    1,
        'normalize': 0
    }
    argparse_from_dicts([args, transform_args], inplace=True)

    test_args = copy.deepcopy(args)
    test_args['force'] = True

    write_images_to_filesystem(only='test',  transform_args=transform_args, **test_args )
    write_images_to_filesystem(only='train', transform_args=transform_args, **args )
