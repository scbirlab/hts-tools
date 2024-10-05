"""Utilities for joining and pivoting tables."""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union
from functools import reduce

from carabiner import print_err
import numpy as np
import pandas as pd
from pandas import DataFrame

from .utils import row_col_to_well

_DUMMY_GROUP = "__group__"

def _join(
    left: DataFrame, 
    right: DataFrame,
    how: str = 'inner',
    sheet_name: Optional[str] = None
) -> DataFrame:
    
    left_cols, right_cols = left.columns.tolist(), right.columns.tolist()
    shared_cols = tuple(set(left_cols).intersection(right_cols))
    
    if len(shared_cols) == 0:
        raise AttributeError(
            'No shared columns for join.'
            f'\n\tLeft: {", ".join(left_cols)}'
            f'\n\tRight: {", ".join(right_cols)}'
            + (f'\n\tRight sheet name: {sheet_name}' 
            if sheet_name is not None else '')
        )

    try:
        data = pd.merge(
            left, 
            right, 
            how=how,
            on=shared_cols,
        )
    # probably joining on a type mismatch, usually int <-> str
    except ValueError as e:  
        col_types = {c: (left[c].dtype, right[c].dtype) 
                     for c in shared_cols}
        mismatches = [key for key, val in col_types.items() if val[0] != val[1]]
        print_err(f'{col_types=}')
        print_err(f'{mismatches=}')
        raise e
    else:    
        return shared_cols, data
    

def _join_reduce(
    how: str = 'inner'
) -> Tuple[Tuple[str], DataFrame]:
    
    def join_reduce(left, right):
        prev_shared_cols, left = left
        shared_cols, data = _join(left, right, how=how)
        return prev_shared_cols + (shared_cols, ), data
    
    return join_reduce


def join(
    left: DataFrame, 
    right: Union[DataFrame, Dict[str, DataFrame]],
    how: str = 'inner') -> DataFrame:
    
    """Perform a database-stype join (merge) between two dataframes.

    This is simply a wrapper around `pandas.merge()` to catch errors and 
    return the shared columns for joining.

    Parameters
    ----------
    left : pandas.DataFrame
        Left dataframe.
    right : pandas.DataFrame or dict
        Right dataframe. If a dict, this should map a str to a 
        pandas.DataFrame, as returned by pandas.read_excel() when
        reading multipel sheets. In this case, the sheets will be 
        joined in order.
    how : str, optional
        Style of join: "inner", "outer", "left". "right". Default: "inner".

    Returns
    -------
    Tuple[str, pandas.DataFrame]
        Shared column headers and joined dataframe.

    Raises
    ------
    AttributeError
        When there are no shared columns.
    ValueError
        When attempting to join on columns of different types. This often
        happens when integers are stored as int in one dataframe and
        str in the other.
    NotImplementedError
        If anything other than a pd.DataFrame or a dictionary mapping to
        pd.DataFrame is provided to the `right` parameter.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(dict(column=['A', 'B', 'A', 'B'], abs=[.1, .2, .23, .11]))
    >>> a  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        column   abs
    0      A  0.10
    1      B  0.20
    2      A  0.23
    3      B  0.11
    >>> b = pd.DataFrame(dict(column=['B', 'A'], drug=['TMP', 'RIF']))
    >>> b  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        column drug
    0      B  TMP
    1      A  RIF
    >>> shared_cols, data = join(a, b)
    >>> shared_cols
    ('column',)
    >>> data  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    column   abs drug
    0      A  0.10  RIF
    1      A  0.23  RIF
    2      B  0.20  TMP
    3      B  0.11  TMP

    """

    if isinstance(right, DataFrame):
        return _join(left, right, how)
    elif isinstance(right, dict):
        dict_values = list(right.values())
        if isinstance(dict_values[0], DataFrame):
            return reduce(
                _join_reduce(how=how), 
                dict_values,
                (tuple(), left),
            )
    else:
        raise NotImplementedError(f"Right table of type {type(right)} not supported.")


def _pivot_plate_df(
    df: DataFrame,
    value_name: str
) -> DataFrame:
    
    df = (
        df
        .reset_index(names='row_id')
        .melt(
            id_vars='row_id', 
            var_name='column_id', 
            value_name=value_name
        )
        .assign(
            well_id=lambda x: row_col_to_well(x['row_id'], x['column_id']),
            plate_id='',
        )
    )
    return df


def _pivot_plate_excel(
    df: Dict[str, DataFrame],
    value_name: str
) -> DataFrame:
    
    dfs = ((_pivot_plate_df(sheet_data, value_name)
                  .assign(plate_id=sheet_name)) 
            for sheet_name, sheet_data in df.items())

    return pd.concat(dfs, axis=0)


def pivot_plate(
    df: Union[DataFrame, Mapping[str, DataFrame]],
    value_name: str = 'value'
) -> DataFrame:
    
    """Pivot from a row x column plate format to a columnar format.

    Handy to convert a visual plate layout to a columnar format for
    data analysis.

    Parameters
    ----------
    df : pandas.DataFrame, Dict[str, pandas.DataFrame]
        Either a dataframe containing rows labels as index and 
        column labels as headings, or a dictionary of names mapping
        to such dataframes (as returned by `pandas.read_excel()`).
    value_name : str, optional
        The column heading to give the values within the plate. Default: "value".

    Returns
    -------
    pandas.DataFrame
        Columnar dataframe containign data from df.

    Raises
    ------
    ValueError
        If df is not a dataframe or a dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> a = pd.DataFrame(index=list("ABCDEFGH"), 
    ...                  columns=range(1, 13), 
    ...                  data=np.arange(1, 97).reshape(8, 12))
    >>> a  # doctest: +NORMALIZE_WHITESPACE
        1   2   3   4   5   6   7   8   9   10  11  12
    A   1   2   3   4   5   6   7   8   9  10  11  12
    B  13  14  15  16  17  18  19  20  21  22  23  24
    C  25  26  27  28  29  30  31  32  33  34  35  36
    D  37  38  39  40  41  42  43  44  45  46  47  48
    E  49  50  51  52  53  54  55  56  57  58  59  60
    F  61  62  63  64  65  66  67  68  69  70  71  72
    G  73  74  75  76  77  78  79  80  81  82  83  84
    H  85  86  87  88  89  90  91  92  93  94  95  96
    >>> pivot_plate(a, value_name="well_number")    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    row_id column_id  well_number well_id plate_id
    0       A         1            1     A01         
    1       B         1           13     B01         
    2       C         1           25     C01         
    3       D         1           37     D01         
    4       E         1           49     E01         
    ..    ...       ...          ...     ...      ...
    91      D        12           48     D12         
    92      E        12           60     E12         
    93      F        12           72     F12         
    94      G        12           84     G12         
    95      H        12           96     H12         
    <BLANKLINE>
    [96 rows x 5 columns]
    >>> pivot_plate({'sheet_1': a}, value_name="well_number")    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    row_id column_id  well_number well_id plate_id
    0       A         1            1     A01  sheet_1
    1       B         1           13     B01  sheet_1
    2       C         1           25     C01  sheet_1
    3       D         1           37     D01  sheet_1
    4       E         1           49     E01  sheet_1
    ..    ...       ...          ...     ...      ...
    91      D        12           48     D12  sheet_1
    92      E        12           60     E12  sheet_1
    93      F        12           72     F12  sheet_1
    94      G        12           84     G12  sheet_1
    95      H        12           96     H12  sheet_1
    <BLANKLINE>
    [96 rows x 5 columns]

    """

    if isinstance(df, Mapping):
        return _pivot_plate_excel(df, value_name)
    elif isinstance(df, DataFrame):
        return _pivot_plate_df(df, value_name)
    else:
        raise ValueError(f"df is a {type(df)}, which is not supported")


def _replicator(x: DataFrame) -> DataFrame:
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x.assign(replicate=idx + 1)


def replicate_table(
    data: DataFrame,
    group: Optional[Union[str, Iterable[str]]] = None,
    wide: Optional[str] = None
) -> DataFrame:
    
    """Annotate a dataframe with replicates within a group.

    Adds a column called "replicate" which contains integer labels randomly 
    assigned within groups indicating repeated measurements of the same experiemntal condition.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe. 
    group : str or list
        Columns which indicate the grouping within which statistics should be calculated. These
        groups indicate repeated measurements of the same experiemntal condition.
    wide : str, optional
        If provided, returns a "wide" dataframe with replciate labels as column headings and the column
        name porived as values for the table.

    Returns 
    -------
    pd.DataFrame
        Dataframe with a new column "replicate" with labels randomly assigned within the group.
        If a column name is provided to wide, then the table as the replicate labels as columns
        and the values from that column as values.

    Raises
    ------
    KeyError
        If wide is provided and not a column in the data.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(dict(group=['g1', 'g1', 'g2', 'g2'], 
    ...                  control=['n', 'n', 'p', 'p'], 
    ...                  m_abs_ch1=[.1, .2, .9, .8], 
    ...                  abs_ch1_wavelength=['600nm'] * 4))
    >>> a  # doctest: +NORMALIZE_WHITESPACE
        group control  m_abs_ch1 abs_ch1_wavelength
    0    g1       n        0.1              600nm
    1    g1       n        0.2              600nm
    2    g2       p        0.9              600nm
    3    g2       p        0.8              600nm
    >>> replicate_table(a, group='group')  # doctest:+NORMALIZE_WHITESPACE, +SKIP
        group control  m_abs_ch1 abs_ch1_wavelength  replicate
    0    g1       n        0.1              600nm          1
    1    g1       n        0.2              600nm          2
    2    g2       p        0.9              600nm          2
    3    g2       p        0.8              600nm          1
    >>> replicate_table(a, group='group', wide='m_abs_ch1')   # doctest: +NORMALIZE_WHITESPACE, +SKIP
    replicate  rep_1  rep_2
    group                  
    g1           0.2    0.1
    g2           0.8    0.9

    """

    if group is None:
        group = _DUMMY_GROUP
        data[group] = group

    data = (
        data
        .groupby(group, group_keys=False)
        .apply(_replicator)
    )
    
    if wide is not None:
        if wide not in data:
            raise KeyError(f"Wide column '{wide}' not in data.")
                       
        data = pd.pivot_table(
            data.assign(
                replicate=lambda x: 'rep_' + x['replicate'].astype(str)),
                index=group,
                columns='replicate', 
                values=wide,
            )
    
    if group == _DUMMY_GROUP:
        data = data.drop(columns=[group])

    return data