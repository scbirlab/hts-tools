"""Functions for statistical testing and hit-calling."""

from typing import Callable, List, Union
from collections.abc import Iterable
from functools import reduce

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind


def _group_getter(name: str, 
                  group: list, 
                  control: list) -> str:

    neg_name = tuple(n for n, g in zip(name, group) if g not in control)
    neg_name = neg_name if len(neg_name) > 1 else neg_name[0]

    return neg_name


def _mw_u(negatives: Iterable, 
          column: str,
          group: list,
          control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[column]

    def mw_u(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        mwu = mannwhitneyu(data[column],
                           negs.get_group(neg_name),
                           method='exact',
                           alternative='two-sided')
        
        df = pd.DataFrame({column + '_mwu': [mwu.statistic],
                           column + '_mwu_p': [mwu.pvalue]})

        return df
    
    return mw_u


def _ttest(negatives: Iterable, 
           column: str,
           group: list,
           control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[column]

    def ttest(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        tt = ttest_ind(data[column],
                        negs.get_group(neg_name),
                        equal_var=False,
                        alternative='two-sided')
        
        df = pd.DataFrame({column + '_t': [tt.statistic],
                           column + '_t_p': [tt.pvalue]})

        return df
    
    return ttest


def _ssmd(negatives: Iterable, 
          column: str,
          group: list,
          control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[column]

    def ssmd(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        this_neg = negs.get_group(neg_name)
        this_neg_mean, this_neg_var = this_neg.mean(), this_neg.var()
        data_mean, data_var = data[column].mean(), data[column].var()

        this_ssmd = (data_mean - this_neg_mean) / ((data_var + this_neg_var) ** .5)
        
        df = pd.DataFrame({column + '_ssmd': [this_ssmd]})

        return df
    
    return ssmd


def _lfc(negatives: Iterable, 
         column: str,
         group: list,
         control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[column]

    def lfc(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        this_neg = negs.get_group(neg_name)
        this_neg_mean = this_neg.mean()
        data_mean = data[column].mean()

        this_lfc = np.nan_to_num(np.log10(data_mean),
                                 nan=-15.) - np.log10(this_neg_mean)
        
        df = pd.DataFrame({column + '_log10fc': [this_lfc]})

        return df
    
    return lfc


def summarize(data: pd.DataFrame, 
              measurement_col: str,
              neg: Union[List[str], str],
              control_col: Union[List[str], str],
              group: Union[List[str], str]) -> pd.DataFrame:

    """Add summary statstics to dataframe.

    Calculates log fold-change (LFC), strictly standardized mean difference (SSMD), T-test, and
    Mann-Whitney U for the measurement_col column of data. There must also be a column heading
    starting with measurement_col and ending in "_wavelength".

    Statstics are within a group indicating repeated measurements of the same condition and, 
    where appropriate, calculated relative to the negative control label neg from the column control_column.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe. Must contain a column measurement_col along with a column heading
        starting with measurement_col and ending in "_wavelength". 
    measurement_col : str
        The column for which statistics should be calculated.
    group : str or list
        Columns which indicate the grouping within which statistics should be calculated. These
        groups indicate repeated measurements of the same experiemntal condition.
    neg : str or list
        Negative control label(s). If more than one, should be in the same order as 
        control_col.
    control_col : str
        Column(s) from which to take the negative control label(s).

    Returns
    -------
    pandas.DataFrame
        Dataframe with summary statistics.

    Raises
    ------
    ValueError
        If length of neg and control_col are not identical.
    KeyError
        If a column heading starting with measurement_col and ending in 
        "_wavelength" is not present in data.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(dict(group=['g1', 'g1', 'g2', 'g2'], 
    ...                       control=['n', 'n', 'p', 'p'], 
    ...                       m_abs_ch1=[.1, .2, .9, .8], 
    ...                       abs_ch1_wavelength=['600nm'] * 4))
    >>> a  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        group control  m_abs_ch1 abs_ch1_wavelength
    0    g1       n        0.1              600nm
    1    g1       n        0.2              600nm
    2    g2       p        0.9              600nm
    3    g2       p        0.8              600nm
    >>> summarize(a, measurement_col='m_abs_ch1', control_col='control', neg='n', group='group')  # doctest: +SKIP
     
    """
    
    if not isinstance(control_col, list):
        control_col = [control_col]

    if not isinstance(neg, list):
        neg = [neg]

    if not len(neg) == len(control_col):
        raise ValueError("Neg and control_col are different lengths:"
                         f"\n\t({len(neg)})\t{', '.join(neg)}"
                         f"\n\t({len(control_col)})\t{', '.join(control_col)}")
    
    if not isinstance(group, list):
        group = [group]
    
    query = ' and '.join([f'{c} == "{n}"' 
                         for c, n in zip(control_col, neg)])
    
    neg_group = [g for g in group if g not in control_col]
    neg_controls = (data.query(query)
                        .groupby(neg_group))

    read_type = '_'.join(measurement_col.split('_')[1:3]) + '_wavelength'

    if not read_type in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    read_type_ann = data[group + [read_type]].drop_duplicates()
    
    grouped = data.groupby(group)

    summaries_to_calculate = ('mean', 'std', 'var', 'count')
    summaries_str = (key if isinstance(key, str) else key.__name__ 
                     for key in summaries_to_calculate)
    renamer = {key:  measurement_col + '_' + key for key in summaries_str}
    summary = (grouped[measurement_col].agg(summaries_to_calculate)
                      .reset_index()
                      .rename(columns=renamer))
    
    funcs_to_apply = [f(neg_controls, measurement_col, group, control_col) 
                      for f in (_mw_u, _ttest, _ssmd, _lfc)]
    applied = [(grouped[[measurement_col]].apply(f)
                       .reset_index()) 
                for f in funcs_to_apply]

    df = reduce(pd.merge, 
                (read_type_ann, summary) + tuple(applied))

    return df