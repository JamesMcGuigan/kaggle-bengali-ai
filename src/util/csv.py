import gc
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

### Predict Output Submssion
from src.dataset.DatasetDF import DatasetDF


### BUGFIX: Repeatedly calling model.predict(...) results in memory leak - https://github.com/keras-team/keras/issues/13118
def submission_df(model, output_shape):
    gc.collect()

    submission = pd.DataFrame(columns=output_shape.keys())
    # large datasets on submit, so loop
    for data_id in range(0,4):
        test_dataset      = DatasetDF(test_train='test', data_id=data_id, transform_X_args = { "normalize": True } )
        test_dataset_rows = test_dataset.X['train'].shape[0]
        batch_size        = 32
        for index in range(0, test_dataset_rows, 32):
            X_batch     = test_dataset.X['train'][index : index+batch_size]
            predictions = model.predict_on_batch(X_batch)
            # noinspection PyTypeChecker
            submission = submission.append(
                pd.DataFrame({
                    key: np.argmax( predictions[index], axis=-1 )
                    for index, key in enumerate(output_shape.keys())
                }, index=test_dataset.ID['train'])
            )
    return submission

###
### Use submission_df() it seems to have more success on Kaggle
###
# def submission_df_generator(model, output_shape):
#     gc.collect(); sleep(5)
#
#     # large datasets on submit, so loop via generator to avoid Out-Of-Memory errors
#     submission = pd.DataFrame(columns=output_shape.keys())
#     for batch in ParquetImageDataGenerator.batch_generator(
#         f"{settings['dir']['data']}/test_image_data_*.parquet",
#         reads_per_file = 3,
#         resamples      = 1,
#         shuffle        = False,
#         infinite       = False,
#     ):
#         X = Transforms.transform_X(batch, normalize=True)
#         predictions = model.predict_on_batch(X)
#         submission = submission.append(
#             pd.DataFrame({
#                 key: np.argmax( predictions[index], axis=-1 )
#                 for index, key in enumerate(output_shape.keys())
#             }, index=batch['image_id'])
#         )
#     return submission


def df_to_submission(df: DataFrame) -> DataFrame:
    output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
    submission = DataFrame(columns=['row_id', 'target'])
    for index, row in df.iterrows():
        for output_field in output_fields:
            index = f"Test_{index}" if not str(index).startswith('T') else index
            submission = submission.append({
                'row_id': f"{index}_{output_field}",
                'target': df[output_field].loc[index],
            }, ignore_index=True)
    return submission


def df_to_submission_csv(df: DataFrame, filename: str):
    submission = df_to_submission(df)
    submission.to_csv(filename, index=False)
    print("wrote:", filename, submission.shape)

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        submission.to_csv('submission.csv', index=False)
        print("wrote:", 'submission.csv', submission.shape)