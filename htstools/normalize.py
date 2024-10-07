"""Functions for normalizing data."""

from typing import Iterable, List, Optional, Union
from functools import partial

from carabiner import print_err
import pandas as pd

def _unique_entries_str(x: pd.DataFrame) -> str:
    return ','.join(sorted(map(str, x.unique())))


def _check_controls(
    data: pd.DataFrame, 
    measurement_col: str,
    control_col: str,
    value: str
) -> None:
        
    if value not in data[control_col].values:
        try:
            raise ValueError(f"{value=} is not in data:\n\t" +
                            _unique_entries_str(data[control_col]))
        except TypeError:
            raise TypeError(f"{control_col=} is not a str column.")

    return None


def _get_grouped_control_means(
    data: pd.DataFrame, 
    measurement_col: str,
    control_col: str,
    neg: str,
    pos: Optional[str] = None,
    group: Optional[Union[str, Iterable[str]]] = None
) -> pd.DataFrame:

    if group is None:
        group = '__group__'
        data = data.assign(**{group: group})
    if measurement_col not in data:
        raise KeyError(
            f"{measurement_col=} is not in data:\n\t{','.join(data.columns.tolist())}"
        )
    
    control_values = {
        control: value for control, value in zip(('neg_mean', 'pos_mean'), (neg, pos))
        if value is not None
    }
    mean_control_column_names = {
        control: measurement_col + s + '_mean' for control, s in zip(control_values, ('_neg', '_pos'))
    }

    control_checker = partial(
        _check_controls,
        data=data,
        measurement_col=measurement_col,
        control_col=control_col,
    )
    control_checker(value=neg)
    
    if pos is not None:
        control_checker(value=pos)

    for control, value in control_values.items():
        name = mean_control_column_names[control]
        control = (
            data
            .query(f'{control_col} == "{value}"')
            .groupby(group)[[measurement_col]]
            .mean()
            .reset_index(names=group)
            .rename(columns={measurement_col: name})
        )
        data = pd.merge(data, control, how='outer')

    if group == '__group__':
        data = data.drop(columns=group)
    return mean_control_column_names, data


def _normalize_pon(data: pd.DataFrame, 
                   measurement_col: str,
                   neg_mean: str,
                   pos_mean: Union[str, None] = None,
                   flip: bool = False) -> pd.DataFrame:

    norm_colname = measurement_col + '_norm.pon'
    
    if flip:
        data[norm_colname] = 1. - (data[measurement_col] / data[neg_mean])
    else:
        data[norm_colname] = data[measurement_col] / data[neg_mean]

    return data


def _normalize_npg(data: pd.DataFrame, 
                   measurement_col: str,
                   neg_mean: str,
                   pos_mean: str,
                   flip: bool = False) -> pd.DataFrame:

    norm_colname = measurement_col + '_norm.npg'
    
    if flip:
        data[norm_colname] = ((data[measurement_col] - data[neg_mean]) / 
                              (data[pos_mean] - data[neg_mean]))
    else:
        data[norm_colname] = ((data[measurement_col] - data[pos_mean]) / 
                              (data[neg_mean] - data[pos_mean]))

    return data


_NORMALIZATION_METHODS = {'npg': _normalize_npg, 
                          'pon': _normalize_pon}

def normalize(data: pd.DataFrame, 
              measurement_col: str,
              control_col: str,
              neg: str,
              method: Union[str, None] = None,
              pos: Union[str, None] = None,
              group: Union[str, List[str], None] = None,
              flip: bool = False) -> pd.DataFrame:
    
    r"""Normalize a column based on controls, optionally within groups.

    Positive controls should represent the 0% signal, and negative controls
    should represent the 100% signal. If you set `flip = True`, then this is 
    reversed. 

    Calculations are performed within groups, such as batches or plates, 
    indicated by the `group` column. This function takes the group-wise mean 
    negative controls $\mu_n$ and, optionally, positive controls
    $\mu_p$. Then within each group calculates the normalized 
    signal.

    Two methods are offered:

    - Normalized proportion of growth (NPG)
    
    Within each group calculates the normalized signal, $s$, of each 
    measured datapoint, $m$:

    $$s = \\frac{m - \mu_p}{\mu_n - \mu_p}$$

    If you set `flip = True`, then this equation is used instead:

    $$s = \\frac{m - \mu_n}{\mu_p - \mu_n}$$

    Requires both positive and negative controls.

    - Proportion of negative (PON)

    Within each group calculates the normalized signal, $s$, of each 
    measured datapoint, $m$:

    $$s = \\frac{m}{\mu_n}$$

    If you set `flip = True`, then this equation is used instead:

    $$s = 1 - \\frac{m}{\mu_n}$$

    Requires only negative controls.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    measurement_col : str
        Name of column containing raw data.
    control_col : str
        Name of column containing control indicators.
    neg : str
        Name of negative controls.
    method : str
        One of PON or NPG. Default PON.
    pos : str, optional
        Name of positive controls.
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
    >>> a = pd.DataFrame(dict(compound=['p', 'p', 'c1', 'c2', 'n', 'n'], 
    ...                       m_abs_ch1=[.1, .2, .5, .4, .9, .8], 
    ...                       abs_ch1_wavelength=['600nm'] * 6))
    >>> a  # doctest: +NORMALIZE_WHITESPACE
        compound  m_abs_ch1 abs_ch1_wavelength
    0        p        0.1              600nm
    1        p        0.2              600nm
    2       c1        0.5              600nm
    3       c2        0.4              600nm
    4        n        0.9              600nm
    5        n        0.8              600nm
    >>> normalize(a, control_col='compound', pos='p', neg='n', measurement_col='m_abs_ch1')  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
        compound  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm.pon
    0        p        0.1              600nm                0.85                0.15            0.117647
    1        p        0.2              600nm                0.85                0.15            0.235294
    2       c1        0.5              600nm                0.85                0.15            0.588235
    3       c2        0.4              600nm                0.85                0.15            0.470588
    4        n        0.9              600nm                0.85                0.15            1.058824
    5        n        0.8              600nm                0.85                0.15            0.941176
    >>> normalize(a, control_col='compound', pos='p', neg='n', measurement_col='m_abs_ch1', flip=True)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
        compound  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm.pon
    0        p        0.1              600nm                0.85                0.15            0.882353
    1        p        0.2              600nm                0.85                0.15            0.764706
    2       c1        0.5              600nm                0.85                0.15            0.411765
    3       c2        0.4              600nm                0.85                0.15            0.529412
    4        n        0.9              600nm                0.85                0.15           -0.058824
    5        n        0.8              600nm                0.85                0.15            0.058824
    >>> normalize(a, control_col='compound', pos='p', neg='n', measurement_col='m_abs_ch1', method='npg')  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE, +SKIP
        compound  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm.npg
    0        p        0.1              600nm                0.85                0.15           -0.071429
    1        p        0.2              600nm                0.85                0.15            0.071429
    2       c1        0.5              600nm                0.85                0.15            0.500000
    3       c2        0.4              600nm                0.85                0.15            0.357143
    4        n        0.9              600nm                0.85                0.15            1.071429
    5        n        0.8              600nm                0.85                0.15            0.928571

    """

    method = (method or 'pon').casefold()
    
    try:
        normalization_function = _NORMALIZATION_METHODS[method]
    except KeyError as e:
        raise AttributeError(f"Normalization method {method} is not supported.")
    
    if method == 'npg' and pos is None:
        raise AttributeError(f"Normalization method {method} requires positive controls.")
    
    control_mean_names, data = _get_grouped_control_means(
        data=data, 
        measurement_col=measurement_col,
        control_col=control_col,
        neg=neg,
        pos=pos,
        group=group,
    )

    return normalization_function(
        data=data, 
        measurement_col=measurement_col,
        flip=flip, 
        **control_mean_names,
    )
