import gc
import os
from time import sleep

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.dataset.DatasetDF import DatasetDF


### Predict Output Submssion
def submission_df(model, output_shape):
    gc.collect(); sleep(5)

    submission = pd.DataFrame(columns=output_shape.keys())
    for data_id in range(0,4):  # large datasets on submit, so loop
        test_dataset = DatasetDF(
            test_train='test',
            data_id=data_id,
            transform_X_args={ "normalize": True }
        )
        predictions  = model.predict(test_dataset.X['train'])
        # noinspection PyTypeChecker
        submission = submission.append(
            pd.DataFrame({
                key: np.argmax( predictions[index], axis=-1 )
                for index, key in enumerate(output_shape.keys())
            }, index=test_dataset.ID['train'])
        )
    return submission


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