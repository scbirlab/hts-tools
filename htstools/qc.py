"""Functions for performing quality control checks on data."""

from collections.abc import Callable, Iterable
from functools import partial, reduce
from itertools import product

import numpy as np
from pandas import DataFrame, merge
from pandas.api.typing import DataFrameGroupBy
from scipy import stats

from .utils import as_list, check_constant_within_groups, group_annotation

_SummaryFunction = Callable[[DataFrameGroupBy], DataFrame]

_SUMMARY_FUNCTIONS: dict[str, dict[bool, _SummaryFunction]] = {
    'sd': {
        False: lambda x: x.std(),
        True: lambda x: x.agg(stats.median_abs_deviation),
    },
    'var': {
        False: lambda x: x.var(),
        True: lambda x: x.var(),
    },
    'mean': {
        False: lambda x: x.mean(),
        True: lambda x: x.median(),
    } 
}

def _var(
    df: DataFrameGroupBy, 
    vartype: str = 'sd',
    robust: bool = False
) -> DataFrame:
    return _SUMMARY_FUNCTIONS[vartype][robust](df)

def _mean(
    df: DataFrameGroupBy, 
    robust: bool = False
) -> DataFrame:
    return _SUMMARY_FUNCTIONS['mean'][robust](df)

def _make_grouped_mean_variance(
    df: DataFrame, 
    measurement_col: str,
    control_col: str,
    pos: str,
    neg: str,
    group: str | Iterable[str] | None = None,
    vartype: str = 'sd',
    robust: bool = False,
    allow_mixed_wavelengths: bool = False
) -> tuple[tuple[str, str, str, str], DataFrame]:
    if vartype not in ('sd', 'var'):
        raise ValueError(f"{vartype=} not supported.")
    
    neg_colname, pos_colname = (
        measurement_col + s for s in ('_neg', '_pos')
    )
    
    read_type = '_'.join(measurement_col.split('_')[1:3]) + '_wavelength'
    if read_type not in df.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")

    # Work on a slim copy so the caller's dataframe is never modified.
    group_is_dummy = group is None
    group_cols = ['__group__'] if group is None else as_list(group)
    check_constant_within_groups(df, group, read_type, 'wavelength column', allow=allow_mixed_wavelengths)
    needed_cols = [control_col, measurement_col, read_type] + ([] if group is None else group_cols)
    df = df[list(dict.fromkeys(needed_cols))]
    if group is None:
        df = df.assign(**{group_cols[0]: group_cols[0]})
    group = group_cols

    z_data = group_annotation(df, group, read_type)

    for q, name in zip((neg, pos), 
                       (neg_colname, pos_colname)):
        (mean_name, var_name) = (
            name + s for s in ('_mean', '_' + vartype)
        )

        grouped = (
            df[df[control_col] == q]
            .groupby(group, dropna=False)[[measurement_col]]
        )

        mean_ = (
            _mean(grouped, robust)
            .reset_index()
            .rename(columns={measurement_col: mean_name})
        )
        var_ = (
            _var(grouped, vartype, robust)
            .reset_index()
            .rename(columns={measurement_col: var_name})
        )
        z_data = reduce(
            partial(merge, how='outer', on=group), 
            (z_data, mean_, var_),
        )
        
    (neg_mean, 
     neg_var, 
     pos_mean, 
     pos_var) = (
        a + b for a, b in product(
            (neg_colname, pos_colname), 
            ('_mean', '_' + vartype),
        )
    )
    
    if group_is_dummy:
        z_data = z_data.drop(columns=group)
    return (neg_mean, neg_var, pos_mean, pos_var), z_data


def ssmd(
    data: DataFrame, 
    measurement_col: str,
    control_col: str,
    pos: str,
    neg: str,
    group: str | Iterable[str] | None = None,
    robust: bool = False,
    allow_mixed_wavelengths: bool = False
) -> DataFrame:
    
    r"""Calculate SSMD based on positive and negative controls, 
    optionally within groups.

    Calculations are performed within groups, such as batches or plates, 
    indicated by the `group` column.
    
    This function takes the group-wise mean and variance of positive and negative 
    controls ($\mu_p$, $\mu_n$, $\sigma_p^2$, $\sigma_n^2$), and then within each group 
    calculates the SSMD, $s$:

    $$s = \frac{\mu_p - \mu_n}{\sqrt{\sigma_n^2 + \sigma_p^2}}$$

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
    robust : bool, optional
        Use median instead of mean (still uses variance). Default: False.
    allow_mixed_wavelengths : bool, optional
        Allow the wavelength to vary within a group (raises ValueError by default).
        Wavelengths are then joined with "|" in the output. Default: False.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe with columns for mean, variance, and SSMD.
    
    """

    ((neg_mean, 
      neg_var, 
      pos_mean, 
      pos_var), 
      z_data) = _make_grouped_mean_variance(
        df=data, 
        measurement_col=measurement_col,
        control_col=control_col,
        pos=pos,
        neg=neg,
        group=group,
        vartype='var',
        robust=robust,
        allow_mixed_wavelengths=allow_mixed_wavelengths,
    )

    ssmd_col = measurement_col + '_ssmd'
    z_data[ssmd_col] = (
        (z_data[pos_mean] - z_data[neg_mean]) / np.sqrt(z_data[pos_var] + z_data[neg_var])
    )

    return z_data


def z_prime_factor(data: DataFrame, 
                   measurement_col: str,
                   control_col: str,
                   pos: str,
                   neg: str,
                   group: str | Iterable[str] | None = None,
                   robust: bool = False,
                   allow_mixed_wavelengths: bool = False) -> DataFrame:
    
    r"""Calculate Z'-factor based on positive and negative controls, 
    optionally within groups.

    Calculations are performed within groups, such as batches or plates, 
    indicated by the `group` column.
    
    This function takes the group-wise mean and standard deviation of positive and 
    negative controls ($\mu_p$, $\mu_n$, $\sigma_p$, $\sigma_n$), and then within 
    each group calculates the Z'-factor, $s$:

    $$s = 1 - 3 \frac{\sigma_n + \sigma_p}{\lvert \mu_n - \mu_p \rvert}$$

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
    robust : bool, optional
        Use median and MAD instead of mean and standard deviation. 
        Default: False.
    allow_mixed_wavelengths : bool, optional
        Allow the wavelength to vary within a group (raises ValueError by default).
        Wavelengths are then joined with "|" in the output. Default: False.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe with columns for mean, variance, and Z'-factor.
    
    """

    ((neg_mean, neg_sd, 
      pos_mean, pos_sd), 
      z_data) = _make_grouped_mean_variance(
        df=data, 
        measurement_col=measurement_col,
        control_col=control_col,
        pos=pos,
        neg=neg,
        group=group,
        vartype='sd',
        robust=robust,
        allow_mixed_wavelengths=allow_mixed_wavelengths,
    )

    zprime_col = measurement_col + '_zprime'
    z_data[zprime_col] = 1. - (
        3. * (z_data[pos_sd] + z_data[neg_sd]) / (z_data[pos_mean] - z_data[neg_mean]).abs()
    )
    return z_data
