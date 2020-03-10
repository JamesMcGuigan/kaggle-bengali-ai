from typing import Union

import pandas as pd
from pandas.io.formats.style import Styler


def combination_matrix(dataset: pd.DataFrame, x: str, y: str, z: str,
                       format=None, unique=True) -> Union[pd.DataFrame, Styler]:
    """
    Returns a combination matrix, showing all valid combinations between three DataFrame columns.
    Sort of like a heatmap, but returning lists of (optionally) unique values

    :param dataset: The dataframe to create a combination_matrx from
    :param x: column name to use for the X axis
    :param y: column name to use for the Y axis
    :param z: column name to use for the Z axis (values that appear in the cells)
    :param format: '', ', '-', ', '\n'    = format value lists as "".join() string
                    str, bool, int, float = cast value lists
    :param unique:  whether to return only unique values or not - eg: combination_matrix(unique=False).applymap(sum)
    :return: returns nothing
    """
    unique_y = sorted(dataset[y].unique())
    combinations = pd.DataFrame({
        n: dataset.where(lambda df: df[y] == n)
            .groupby(x)[z]
            .pipe(lambda df: df.unique() if unique else df )
            .apply(list)
            .apply(sorted)
        for n in unique_y
    }).T

    if isinstance(format, str):
        combinations = combinations.applymap(
            lambda cell: f"{format}".join([str(value) for value in list(cell) ])
            if isinstance(cell, list) else cell
        )
    if format == str:   combinations = combinations.applymap(lambda cell: str(cell)      if isinstance(cell, list) and len(cell) > 0 else ''     )
    if format == bool:  combinations = combinations.applymap(lambda cell: True           if isinstance(cell, list) and len(cell) > 0 else False  )
    if format == int:   combinations = combinations.applymap(lambda cell: int(cell[0])   if isinstance(cell, list) and len(cell)     else ''     )
    if format == float: combinations = combinations.applymap(lambda cell: float(cell[0]) if isinstance(cell, list) and len(cell)     else ''     )

    combinations.index.rename(y, inplace=True)
    combinations.fillna('', inplace=True)
    if format == '\n':
        return combinations.style.set_properties(**{'white-space': 'pre-wrap'})  # needed for display
    else:
        return combinations  # Allows for subsequent .applymap()
