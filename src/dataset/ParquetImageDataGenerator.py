# Notebook: https://www.kaggle.com/jamesmcguigan/reading-parquet-files-ram-cpu-optimization/
# Notebook: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing
import gc
import math
from collections import Callable

import glob2
import pandas as pd
from frozendict import frozendict
from keras_preprocessing.image import ImageDataGenerator
from pyarrow.parquet import ParquetFile


class ParquetImageDataGenerator(ImageDataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def flow_from_parquet(
            self,
            glob_path:       str,
            transform_X:     Callable,
            transform_Y:     Callable,
            transform_X_args = frozendict(),
            transform_Y_args = frozendict(),
            batch_size       = 32,
            reads_per_file   = 2,
            resamples        = 1,
            shuffle          = False,
            infinite         = True,
            test             = False,
    ):
        """
            Source: ./venv/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py
            # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        """
        if test:
            shuffle  = False
            infinite = False

        for (X,Y) in self.cache_XY_generator(
                glob_path=glob_path,
                transform_X=transform_X,
                transform_X_args=transform_X_args,
                transform_Y=transform_Y,
                transform_Y_args=transform_Y_args,
                reads_per_file=reads_per_file,
                resamples=resamples,
                shuffle=shuffle,
                infinite=infinite,
        ):
            cache_size  = X.shape[0]
            batch_count = math.ceil( cache_size / batch_size )
            for n_batch in range(batch_count):
                X_batch = X[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                if isinstance(Y, dict):
                    Y_batch = {
                        key: Y[key][ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                        for key in Y.keys()
                    }
                else:
                    Y_batch = Y[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                yield ( X_batch, Y_batch )


    @classmethod
    def cache_XY_generator(
            cls,
            glob_path:        str,
            transform_X:      Callable,
            transform_X_args: {},
            transform_Y:      Callable,
            transform_Y_args: {},
            reads_per_file  = 3,
            resamples       = 1,
            shuffle         = False,
            infinite        = False,
    ):
        for cache in cls.cache_generator(
                glob_path=glob_path,
                reads_per_file=reads_per_file,
                resamples=resamples,
                shuffle=shuffle,
                infinite=infinite,
        ):
            X = transform_X(cache, **transform_X_args)
            Y = transform_Y(cache, **transform_Y_args)
            yield (X, Y)


    @classmethod
    def cache_generator(
            cls,
            glob_path,
            reads_per_file = 3,
            resamples      = 1,
            shuffle        = False,
            infinite       = False,
    ):
        filenames = sorted(glob2.glob(glob_path))
        if len(filenames) == 0: raise Exception(f"{cls.__name__}.batch_generator() - invalid glob_path: {glob_path}")

        gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector
        while True:
            for filename in filenames:
                num_rows    = ParquetFile(filename).metadata.num_rows
                cache_size  = math.ceil( num_rows / reads_per_file )
                for n_read in range(reads_per_file):
                    gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector
                    cache = (
                        pd.read_parquet(filename)
                            # .set_index('image_id', drop=True)  # WARN: Don't do this, it breaks other things
                            .iloc[ cache_size * n_read : cache_size * (n_read+1) ]
                            .copy()
                    )
                    for resample in range(resamples):
                        if shuffle:
                            cache = cache.sample(frac=1)
                        yield cache
            if not infinite: break