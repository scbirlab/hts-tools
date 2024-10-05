"""Statistical testing and hit-calling."""

from typing import Callable, Iterable, List, Union
from functools import reduce

from carabiner import cast
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

def _group_getter(
    names: Union[str, Iterable[str]], 
    groups: Iterable[str], 
    control: Iterable[str]
) -> str:

    names = tuple(cast(names, to=list))
    neg_name = tuple(
        name for name, group in zip(names, groups) 
        if group not in control
    )
    # neg_name = neg_name if len(neg_name) > 1 else neg_name[0]
    return neg_name


def _mw_u(negatives: pd.DataFrame, 
          measurement_col: str,
          group: list,
          control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[measurement_col]

    def mw_u(data: pd.DataFrame) -> pd.DataFrame:
        neg_name = _group_getter(data.name, group, control)
        mwu = mannwhitneyu(
            data[measurement_col],
            negs.get_group(neg_name),
            method='exact',
            alternative='two-sided',
        )
        df = pd.DataFrame({
            measurement_col + '_mwu.stat': [mwu.statistic],
            measurement_col + '_mwu.p': [mwu.pvalue]
        })
        return df
    
    return mw_u


def _ttest(negatives: Iterable, 
           measurement_col: str,
           group: list,
           control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[measurement_col]

    def ttest(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        tt = ttest_ind(data[measurement_col],
                       negs.get_group(neg_name),
                       equal_var=False,
                       alternative='two-sided')
        
        df = pd.DataFrame({measurement_col + '_t.stat': [tt.statistic],
                           measurement_col + '_t.p': [tt.pvalue]})

        return df
    
    return ttest


def _ssmd(negatives: Iterable, 
          measurement_col: str,
          group: list,
          control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[measurement_col]

    def ssmd(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        this_neg = negs.get_group(neg_name)
        this_neg_mean, this_neg_var = this_neg.mean(), this_neg.var()
        data_mean, data_var = data[measurement_col].mean(), data[measurement_col].var()

        this_ssmd = (data_mean - this_neg_mean) / np.sqrt(data_var + this_neg_var)
        
        df = pd.DataFrame({measurement_col + '_ssmd': [this_ssmd]})

        return df
    
    return ssmd


def _lfc(negatives: Iterable, 
         measurement_col: str,
         group: list,
         control: list) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    negs = negatives[measurement_col]

    def lfc(data: pd.DataFrame) -> pd.DataFrame:

        neg_name = _group_getter(data.name, group, control)

        this_neg = negs.get_group(neg_name)
        this_neg_mean = this_neg.mean()
        data_mean = data[measurement_col].mean()

        this_lfc = np.nan_to_num(np.log10(data_mean),
                                 nan=-15.) - np.log10(this_neg_mean)
        
        df = pd.DataFrame({measurement_col + '_log10fc': [this_lfc]})

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
        groups indicate repeated measurements of the same experimental condition.
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
    >>> a = pd.DataFrame(dict(gene=['g1', 'g1', 'g2', 'g2', 'g1', 'g1', 'g2', 'g2'], 
    ...                       compound=['n', 'n', 'n', 'n', 'cmpd1', 'cmpd1', 'cmpd2', 'cmpd2'], 
    ...                       m_abs_ch1=[.1, .2, .9, .8, .1, .3, .5, .45], 
    ...                       abs_ch1_wavelength=['600nm'] * 8))
    >>> a  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        gene compound  m_abs_ch1 abs_ch1_wavelength
    0    g1        n       0.10              600nm
    1    g1        n       0.20              600nm
    2    g2        n       0.90              600nm
    3    g2        n       0.80              600nm
    4    g1    cmpd1       0.10              600nm
    5    g1    cmpd1       0.30              600nm
    6    g2    cmpd2       0.50              600nm
    7    g2    cmpd2       0.45              600nm
    >>> summarize(a, measurement_col='m_abs_ch1', control_col='compound', neg='n', group='gene')  # doctest: +SKIP
      gene abs_ch1_wavelength  m_abs_ch1_mean  m_abs_ch1_std  ...  m_abs_ch1_t.stat  m_abs_ch1_t.p  m_abs_ch1_ssmd  m_abs_ch1_log10fc
    0   g1              600nm          0.1750       0.095743  ...          0.361158       0.742922        0.210042           0.066947
    1   g2              600nm          0.6625       0.221265  ...         -1.544396       0.199787       -0.807183          -0.108233

    [2 rows x 12 columns]
    >>> summarize(a, measurement_col='m_abs_ch1', control_col='compound', neg='n', group=['gene', 'compound'])  # doctest: +SKIP
    gene compound abs_ch1_wavelength  m_abs_ch1_mean  ...  m_abs_ch1_t.stat  m_abs_ch1_t.p  m_abs_ch1_ssmd  m_abs_ch1_log10fc
    0   g1        n              600nm           0.150  ...          0.000000       1.000000        0.000000           0.000000
    1   g2        n              600nm           0.850  ...          0.000000       1.000000        0.000000           0.000000
    2   g1    cmpd1              600nm           0.200  ...          0.447214       0.711723        0.316228           0.124939
    3   g2    cmpd2              600nm           0.475  ...         -6.708204       0.044534       -4.743416          -0.252725

    [4 rows x 13 columns]

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
    
    neg_group = [g for g in group 
                 if g not in control_col]
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
    renamer = {key:  measurement_col + '_' + key 
               for key in summaries_str}
    summary = (grouped[measurement_col]
                      .agg(summaries_to_calculate)
                      .reset_index()
                      .rename(columns=renamer))
    
    funcs_to_apply = [summary_function(neg_controls, 
                                       measurement_col, 
                                       group, 
                                       control_col) 
                      for summary_function in (_mw_u, _ttest, _ssmd, _lfc)]
    applied = [(grouped[[measurement_col]]
                       .apply(func)
                       .reset_index()) 
                for func in funcs_to_apply]
    applied = [a.drop(columns=[c for c in a if c.startswith('level_') and c[-1].isdigit()])
               for a in applied]

    df = reduce(pd.merge, 
                (read_type_ann, summary) + tuple(applied))

    return df