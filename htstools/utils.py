"""Miscellaneous utilities for hts-tools."""

from typing import Union
from collections.abc import Mapping
import sys

import numpy as np
import pandas as pd

cbpal = ('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB', "#000000")

def _print_err(*args, **kwargs) -> None:

    return print(*args, **kwargs, file=sys.stderr)


def pprint_dict(x: Mapping, 
                message: str) -> None:
    
    key_val_str = (f'{key}: {val:.2f}' if isinstance(val, float) else f'{key}: {val}'
                   for key, val in x.items())

    _print_err(f'{message}:\n\t' + '\n\t'.join(key_val_str))
    
    return None


def _pandasify(x: Union[list, np.ndarray]) -> pd.Series:

    return pd.Series(x)


def row_col_to_well(row: Union[pd.Series, list, np.ndarray], 
                    col: Union[pd.Series, list, np.ndarray], 
                    pad: bool = True) -> pd.Series:
    
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
        
        col = _pandasify(col).astype(str).str.zfill(2)

    return _pandasify(row).astype(str).str.cat(_pandasify(col).astype(str))