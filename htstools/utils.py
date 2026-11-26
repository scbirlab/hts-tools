"""Miscellaneous utilities for hts-tools."""

from typing import Iterable, Mapping, Optional, Union
import sys

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, Series

def row_col_to_well(
    row: ArrayLike, 
    col: ArrayLike, 
    pad: bool = True
) -> Series:
    
    """Concatenate row label and column label columns to a well label column.

    Optionally left zero-pads the column label.

    Parameters
    ----------
    row : pandas.Series, numpy.ndarray, or list
        Row labels.
    col : pandas.Series, numpy.ndarray, or list
        Column labels.
    pad : bool, optional
        Whether to left zero-pad the column labels, 
        i.e. A, 1 -> A01. Default: True.

    Returns
    -------
    pandas.Series
        Well labels.

    Examples
    --------
    >>> row_col_to_well(row=['A', 'B', 'C'], col=[1, 6, 12])
    0    A01
    1    B06
    2    C12
    dtype: object
    >>> row_col_to_well(row=['A', 'B', 'C'], col=[1, 6, 12], pad=False)
    0     A1
    1     B6
    2    C12
    dtype: object

    """

    if pad:
        col = Series(col).astype(str).str.zfill(2)
    else:
        col = Series(col).astype(str)
    return Series(row).astype(str).str.cat(col)


def check_constant_within_groups(
    data: DataFrame,
    group: Optional[Union[str, Iterable[str]]],
    col: str,
    description: str = 'column'
) -> None:
    """Raise a ValueError if `col` takes more than one value within any group.

    Used to stop statistics being silently pooled across, for example, 
    different wavelengths. Does nothing if `col` is not in the data. If `group`
    is None the whole table is one group. NaN is treated as a value.

    """
    if col not in data.columns:
        return None
    if group is None:
        n_values = data[col].nunique(dropna=False)
        bad = {'(all data)': n_values} if n_values > 1 else {}
    else:
        group = [group] if isinstance(group, str) else list(group)
        counts = data.groupby(group, dropna=False)[col].nunique(dropna=False)
        bad = counts[counts > 1].to_dict()
    if len(bad) > 0:
        raise ValueError(
            f"The {description} '{col}' takes more than one value within "
            f"{len(bad)} group(s) of {group}, e.g. {list(bad)[:3]}. Statistics would be "
            f"pooled across them. Add '{col}' to the grouping columns."
        )
    return None
