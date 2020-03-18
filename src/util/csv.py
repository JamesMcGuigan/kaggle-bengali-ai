import os

import gc
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.dataset.DatasetDF import DatasetDF
from src.dataset.ParquetImageDataGenerator import ParquetImageDataGenerator
from src.dataset.Transforms import Transforms
from src.settings import settings



### BUGFIX: Repeatedly calling model.predict(...) results in memory leak - https://github.com/keras-team/keras/issues/13118
def submission_df(model, output_shape):
    gc.collect()

    submission = pd.DataFrame(columns=output_shape.keys())
    # large datasets on submit, so loop
    for data_id in range(0,4):
        test_dataset      = DatasetDF(test_train='test', data_id=data_id, transform_X_args = { "normalize": True } )
        test_dataset_rows = test_dataset.X['train'].shape[0]
        batch_size        = 64
        for index in range(0, test_dataset_rows, batch_size):
            try:
                X_batch     = test_dataset.X['train'][index : index+batch_size]
                predictions = model.predict_on_batch(X_batch)
                # noinspection PyTypeChecker
                submission = submission.append(
                    pd.DataFrame({
                        key: np.argmax( predictions[index], axis=-1 )
                        for index, key in enumerate(output_shape.keys())
                        }, index=test_dataset.ID['train'])
                    )
            except Exception as exception:
                print('submission_df_generator()', exception)

        return submission

###
### Use submission_df() it seems to have more success on Kaggle
###
def submission_df_generator(model, output_shape):
    gc.collect()

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive':
        globpath = f"{settings['dir']['data']}/train_image_data_*.parquet"
    else:
        globpath = f"{settings['dir']['data']}/test_image_data_*.parquet"

    # large datasets on submit, so loop via generator to avoid Out-Of-Memory errors
    submission  = pd.DataFrame(columns=output_shape.keys())
    cache_index = 0
    for cache in ParquetImageDataGenerator.cache_generator(
            globpath,
            reads_per_file = 3,
            resamples      = 1,
            shuffle        = False,
            infinite       = False,
            ):
        try:
            cache_index      += 1
            batch_size        = 64
            test_dataset_rows = cache.shape[0]
            print(f'submission_df_generator() - submission: ', cache_index, submission.shape)
            if test_dataset_rows == 0: continue
            for index in range(0, test_dataset_rows, batch_size):
                try:
                    batch = cache[index : index+batch_size]
                    if batch.shape[0] == 0: continue
                    X           = Transforms.transform_X(batch, normalize=True)
                    predictions = model.predict_on_batch(X)
                    submission  = submission.append(
                        pd.DataFrame({
                            key: np.argmax( predictions[index], axis=-1 )
                            for index, key in enumerate(output_shape.keys())
                            }, index=batch['image_id'])
                        )
                except Exception as exception:
                    print('submission_df_generator() - batch', exception)
        except Exception as exception:
            print('submission_df_generator() - cache', exception)

    return submission



# def df_to_submission(df: DataFrame) -> DataFrame:
#     print('df_to_submission() - input', df.shape)
#     output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
#     submission = DataFrame(columns=['row_id', 'target'])
#     for index, row in df.iterrows():
#         for output_field in output_fields:
#             try:
#                 index = f"Test_{index}" if not str(index).startswith('T') else index
#                 submission = submission.append({
#                     'row_id': f"{index}_{output_field}",
#                     'target': df[output_field].loc[index],
#                     }, ignore_index=True)
#             except Exception as exception:
#                 print('df_to_submission()', exception)
#     print('df_to_submission() - output', submission.shape)
#     return submission


def df_to_submission(df: DataFrame) -> DataFrame:
    print('df_to_submission_columns() - input', df.shape)
    output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
    submissions = {}
    for output_field in output_fields:
        if 'image_id' in df.columns:
            submissions[output_field] = DataFrame({
                'row_id': df['image_id'] + '_' + output_field,
                'target': df[output_field],
            })
        else:
            submissions[output_field] = DataFrame({
                'row_id': df.index + '_' + output_field,
                'target': df[output_field],
            })
    submission = pd.concat(submissions.values())
    submission = submission.sort_values(by='row_id')
    print('df_to_submission_columns() - output', submission.shape)
    return submission


def df_to_submission_csv(df: DataFrame, filename: str):
    submission = df_to_submission(df)

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        submission.to_csv('submission.csv', index=False)
        print("wrote:", 'submission.csv', submission.shape)
    else:
        submission.to_csv(filename, index=False)
        print("wrote:", filename, submission.shape)
