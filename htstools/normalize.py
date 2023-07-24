"""Functions for normalizing data."""

from typing import List, Union

import pandas as pd

def _unique_entries_str(x: pd.DataFrame) -> str:

    return ','.join(sorted(map(str, x.unique())))


def normalize(data: pd.DataFrame, 
              measurement_col: str,
              control_col: str,
              pos: str,
              neg: str,
              group: Union[str, List[str], None] = None,
              flip: bool = False) -> pd.DataFrame:
    
    """Normalize a column based on positive and negative controls, optionally within groups.

    Positive controls should represent the 0% signal, and negative controls
    should represent the 100% signal. If you set `flip = True`, then this is 
    reversed. 

    Calculations are performed within groups, such as batches or plates, 
    indicated by the `group` column.
    
    This function takes the group-wise mean of positive and negative controls 
    ($\mu_p$ and $\mu_n$), and then within each group calculates the normalized 
    signal, $s$, of each measured datapoint, $m$:

    $$s = \\frac{m - \mu_p}{\mu_n - \mu_p}$$

    If you set `flip = True`, then this is used instead:

    $$s = \\frac{m - \mu_n}{\mu_p - \mu_n}$$

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    measurement_col : str
        Name of column containing raw data.
    control_col : str
        Name of column containing control indicators.
    pos : str
        Name of positive controls.
    neg : str
        Name of negative controls.
    group : str or list, optional
        Name of column containing the grouping variable, such 
        as plates or batches. If not set, then entire the data is 
        taken as one big group.
    flip : bool, optional
        Set positive controls as 100% signal, and negative 
        controls as 0% signal.

    Returns
    -------
    pandas.DataFrame
        Input data with additional columns, containing mean positive and 
        negative control values (headers ending with "_neg_mean" and 
        "_pos_mean") and normalized data values (header ending with 
        "_norm").

    Raises
    ------
    KeyError
        If measurement_col is not in data.
    ValueError
        If neg or pos is not in data.
    TypeError
        If control_col is not a str column.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(dict(control=['n', 'n', '', '', 'p', 'p'], 
    ...                  m_abs_ch1=[.1, .2, .5, .4, .9, .8], 
    ...                  abs_ch1_wavelength=['600nm'] * 6))
    >>> a  # doctest: +NORMALIZE_WHITESPACE
        control  m_abs_ch1 abs_ch1_wavelength
    0       n        0.1              600nm
    1       n        0.2              600nm
    2                0.5              600nm
    3                0.4              600nm
    4       p        0.9              600nm
    5       p        0.8              600nm
    >>> normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1')  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
        control  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm
    0       n        0.1              600nm                0.15                0.85        1.071429
    1       n        0.2              600nm                0.15                0.85        0.928571
    2                0.5              600nm                0.15                0.85        0.500000
    3                0.4              600nm                0.15                0.85        0.642857
    4       p        0.9              600nm                0.15                0.85       -0.071429
    5       p        0.8              600nm                0.15                0.85        0.071429
    >>> normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1', flip=True)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
        control  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm
    0       n        0.1              600nm                0.15                0.85       -0.071429
    1       n        0.2              600nm                0.15                0.85        0.071429
    2                0.5              600nm                0.15                0.85        0.500000
    3                0.4              600nm                0.15                0.85        0.357143
    4       p        0.9              600nm                0.15                0.85        1.071429
    5       p        0.8              600nm                0.15                0.85        0.928571
    
    """

    neg_colname, pos_colname = (measurement_col + s + '_mean' 
                                for s in ('_neg', '_pos'))
    norm_colname = measurement_col + '_norm'

    if group is None:

        group = '__unigroup__'
        data[group] = group

    if measurement_col not in data:
        raise KeyError(f"{measurement_col=} is "
                       f"not in data:\n\t{','.join(data.columns.tolist())}")

    if pos not in data[control_col].values:
        try:
            raise ValueError(f"{pos=} is not in data:\n\t" +
                             _unique_entries_str(data[control_col]))
        except TypeError:
            raise TypeError(f"{control_col=} is not a str column.")

    if neg not in data[control_col].values:
        raise ValueError(f"{neg=} is not in data:\n\t" +
                         _unique_entries_str(data[control_col]))

    for q, name in zip((neg, pos), 
                       (neg_colname, pos_colname)):

        control = (data.query(f'{control_col} == "{q}"')
                       .groupby(group)[[measurement_col]]
                       .mean()
                       .reset_index(names=group)
                       .rename(columns={measurement_col: name}))

        data = pd.merge(data, control, how='outer')

    if group == '__unigroup__':

        data = data.drop(columns=group)
    
    if flip:
        data[norm_colname] = ((data[measurement_col] - data[neg_colname]) / 
                              (data[pos_colname] - data[neg_colname]))
    else:
        data[norm_colname] = ((data[measurement_col] - data[pos_colname]) / 
                              (data[neg_colname] - data[pos_colname]))

    return data