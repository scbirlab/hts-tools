"""Miscellaneous utilities for hts-tools."""

import warnings
from collections.abc import Iterable

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
    >>> row_col_to_well(row=['A', 'B', 'C'], col=[1, 6, 12]).tolist()
    ['A01', 'B06', 'C12']
    >>> row_col_to_well(row=['A', 'B', 'C'], col=[1, 6, 12], pad=False).tolist()
    ['A1', 'B6', 'C12']

    """

    if pad:
        col = Series(col).astype(str).str.zfill(2)
    else:
        col = Series(col).astype(str)
    return Series(row).astype(str).str.cat(col)


def check_constant_within_groups(
    data: DataFrame,
    group: str | Iterable[str] | None,
    col: str,
    description: str = 'column',
    allow: bool = False,
) -> None:
    """Raise a ValueError if `col` takes more than one value within any group.

    If `allow` is True, a UserWarning is emitted instead of raising.

    Used to stop statistics being silently pooled across, for example, 
    different wavelengths. Does nothing if `col` is not in the data. If `group`
    is None the whole table is one group. NaN is treated as a value.

    """
    if col not in data.columns:
        return None
    if group is None:
        n_values = data[col].nunique(dropna=False)
        bad: dict = {'(all data)': n_values} if n_values > 1 else {}
    else:
        group = as_list(group)
        counts = data.groupby(group, dropna=False)[col].nunique(dropna=False)
        bad = dict(counts[counts > 1].to_dict())
    if len(bad) > 0:
        message = (
            f"The {description} '{col}' takes more than one value within "
            f"{len(bad)} group(s) of {group}, e.g. {list(bad)[:3]}. Statistics would be "
            f"pooled across them. Add '{col}' to the grouping columns."
        )
        if not allow:
            raise ValueError(message)
        warnings.warn(
            message + " Continuing because mixed wavelengths were explicitly allowed; "
            "this is only meaningful for unitless, separately normalized values.",
            UserWarning,
            stacklevel=3,
        )
    return None


def as_list(x: str | Iterable[str]) -> list[str]:
    """Wrap a single column name in a list, or convert an iterable of names to a list."""
    return [x] if isinstance(x, str) else list(x)


def group_annotation(
    data: DataFrame,
    group: Iterable[str],
    col: str
) -> DataFrame:
    """One row per group, with the group columns and an annotation column `col`.

    If `col` takes several values within a group (only possible when mixed
    values were explicitly allowed), they are joined with "|" so that every 
    group still has exactly one row. If `col` is itself a grouping column, 
    only the group columns are returned.

    """
    group = list(group)
    if col in group:
        return data[group].drop_duplicates()
    return (
        data
        .groupby(group, dropna=False)[col]
        .agg(lambda x: '|'.join(sorted(map(str, x.dropna().unique()))))
        .reset_index()
    )
