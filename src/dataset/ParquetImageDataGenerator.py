# Source: https://www.kaggle.com/jamesmcguigan/reading-parquet-files-ram-cpu-optimization/
import gc
import math
from collections import Callable

import glob2
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from pyarrow.parquet import ParquetFile


class ParquetImageDataGenerator(ImageDataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def flow_from_parquet(self,
                          glob_path: str,
                          transform_X: Callable,
                          transform_Y: Callable,
                          batch_size     = 32,
                          reads_per_file = 2,
                          resamples      = 1,
                          shuffle        = False,
                          seed           = 0,
                          infinite       = True,
                          test           = False,
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

        for batch in self.batch_generator(glob_path,
                                          batch_size=batch_size,
                                          reads_per_file=reads_per_file,
                                          resamples=resamples,
                                          shuffle=shuffle,
                                          seed=seed,
                                          infinite=infinite,
        ):
            X = transform_X(batch)
            Y = transform_Y(batch)
            if test: yield X
            else:    yield (X,Y)


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing
    @classmethod
    def batch_generator(cls,
                        glob_path,
                        batch_size=128,
                        reads_per_file=2,
                        resamples=1,
                        shuffle=False,
                        seed=0,
                        infinite=False,
    ):
        filenames = sorted(glob2.glob(glob_path))
        if len(filenames) == 0: raise Exception(f"{cls.__name__}.batch_generator() - invalid glob_path: {glob_path}")

        gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector
        while True:
            for filename in filenames:
                num_rows    = ParquetFile(filename).metadata.num_rows
                cache_size  = math.ceil( num_rows / batch_size / reads_per_file ) * batch_size
                batch_count = math.ceil( cache_size / batch_size )
                for n_read in range(reads_per_file):
                    cache = pd.read_parquet(filename).iloc[ cache_size * n_read : cache_size * (n_read+1) ].copy()
                    gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector

                    for resample in range(resamples):
                        if shuffle:
                            cache = cache.sample(frac=1, random_state=seed)
                        for n_batch in range(batch_count):
                            yield cache[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()

            if not infinite: break