"""Miscellaneous utilities for hts-tools."""

from typing import Mapping, Union
import sys

import numpy as np
from numpy.typing import ArrayLike
from pandas import Series

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