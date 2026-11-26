"""Statistical testing and hit-calling."""

from collections.abc import Iterable
from functools import reduce

import numpy as np
import pandas as pd
from carabiner import print_err
from scipy.stats import mannwhitneyu, ttest_ind

from .utils import as_list, check_constant_within_groups, group_annotation

_STAT_SUFFIXES = (
    '_mwu.stat', '_mwu.p', 
    '_t.stat', '_t.p', 
    '_ssmd', 
    '_log10fc',
)


def _as_tuple(key) -> tuple:
    return key if isinstance(key, tuple) else (key, )


def _hashable_key(key) -> tuple:
    """Group key usable in a dict: NaN != NaN, so missing values are replaced by None."""
    return tuple(None if pd.isna(k) else k for k in _as_tuple(key))


def _group_getter(
    names: str | Iterable[str], 
    groups: Iterable[str], 
    control: Iterable[str]
) -> tuple:
    """Convert the key of a data group into the key of its matching negative control group."""
    return _hashable_key(tuple(
        name for name, group in zip(_as_tuple(names), groups) 
        if group not in control
    ))


def _compare_to_negative(
    this: pd.Series, 
    neg: pd.Series
) -> dict[str, float]:
    """Calculate MWU, Welch's T-test, SSMD, and LFC of `this` relative to `neg`.

    All statistics are NaN if either sample is empty (after dropping NaNs).
    The log10 fold-change is NaN if either mean is not positive, because it 
    is undefined there; no sentinel value is substituted.

    """
    results = dict.fromkeys(_STAT_SUFFIXES, np.nan)
    this, neg = this.dropna(), neg.dropna()
    if this.size == 0 or neg.size == 0:
        return results

    mwu = mannwhitneyu(
        this,
        neg,
        method='auto',
        alternative='two-sided',
    )
    tt = ttest_ind(
        this,
        neg,
        equal_var=False,
        alternative='two-sided',
    )
    this_mean, neg_mean = this.mean(), neg.mean()
    results['_mwu.stat'], results['_mwu.p'] = mwu.statistic, mwu.pvalue
    results['_t.stat'], results['_t.p'] = tt.statistic, tt.pvalue
    results['_ssmd'] = (this_mean - neg_mean) / np.sqrt(this.var() + neg.var())
    if this_mean > 0. and neg_mean > 0.:
        results['_log10fc'] = np.log10(this_mean) - np.log10(neg_mean)
    return results


def _grouped_comparison(
    negatives: pd.DataFrame, 
    measurement_col: str,
    group: list,
    control: list
):
    """Make a function to apply to each data group, comparing it to its negative controls.
    
    Returns the function and a set which is populated with the keys of groups
    for which there were no matching negative controls.

    """
    neg_group = [g for g in group if g not in control]
    if len(neg_group) > 0:
        negs = {
            _hashable_key(key): values[measurement_col] 
            for key, values in negatives.groupby(neg_group, dropna=False)
        }
    else:
        negs = {tuple(): negatives[measurement_col]}
    missing = set()
    no_negs = pd.Series(dtype=float)

    def compare(data: pd.DataFrame) -> pd.Series:
        neg_name = _group_getter(data.name, group, control)
        this_neg = negs.get(neg_name, no_negs)
        if this_neg.notna().sum() == 0:  # no rows, or only NaNs
            missing.add(neg_name)
        stats = _compare_to_negative(data[measurement_col], this_neg)
        return pd.Series({
            measurement_col + suffix: stats[suffix] for suffix in _STAT_SUFFIXES
        })
    
    return compare, missing


def summarize(data: pd.DataFrame, 
              measurement_col: str,
              neg: list[str] | str,
              control_col: list[str] | str,
              group: list[str] | str,
              allow_mixed_wavelengths: bool = False) -> pd.DataFrame:

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
    allow_mixed_wavelengths : bool, optional
        By default a ValueError is raised if the wavelength varies within a 
        group, or within the groups that negative controls are matched on, 
        because statistics would then pool or compare different wavelengths. 
        Set to True to allow this (wavelengths are joined with "|" in the 
        output). Default: False.

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
    
    control_col = as_list(control_col)
    neg = as_list(neg)

    if not len(neg) == len(control_col):
        raise ValueError("Neg and control_col are different lengths:"
                         f"\n\t({len(neg)})\t{', '.join(map(str, neg))}"
                         f"\n\t({len(control_col)})\t{', '.join(control_col)}")
    
    group = as_list(group)
    
    read_type = '_'.join(measurement_col.split('_')[1:3]) + '_wavelength'

    if read_type not in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    # negatives are matched on the grouping columns other than the control column(s), 
    # so the wavelength must be constant within those groups for comparisons to be valid
    neg_group = [g for g in group if g not in control_col]
    # Treat infinite measurements as missing once, up front, so that descriptive 
    # and comparative statistics (and missing-control warnings) use the same observations.
    data = data.assign(**{
        measurement_col: data[measurement_col].replace([np.inf, -np.inf], np.nan)
    })
    check_constant_within_groups(
        data, 
        neg_group if len(neg_group) > 0 else None, 
        read_type, 
        'wavelength column',
        allow=allow_mixed_wavelengths,
    )

    is_neg = reduce(
        lambda a, b: a & b, 
        (data[c] == n for c, n in zip(control_col, neg)),
    )
    if not is_neg.any():
        raise ValueError(
            f"No negative controls found where {control_col} == {neg}."
        )
    neg_controls = data[is_neg]

    read_type_ann = group_annotation(data, group, read_type)
    
    grouped = data.groupby(group, dropna=False)

    summaries_to_calculate = ('mean', 'std', 'var', 'count')
    renamer = {
        key: measurement_col + '_' + key for key in summaries_to_calculate
    }
    summary = (
        grouped[measurement_col]
        .agg(list(summaries_to_calculate))
        .reset_index()
        .rename(columns=renamer)
    )

    compare, missing = _grouped_comparison(
        neg_controls, 
        measurement_col, 
        group, 
        control_col,
    )
    # one pass over the groups for all four statistics
    stats = grouped[[measurement_col]].apply(compare).reset_index()

    if len(missing) > 0:
        print_err(
            f"WARNING: {len(missing)} group(s) of {measurement_col} have no matching "
            "negative controls, so their comparison statistics are NaN. "
            "First few: " + '; '.join(map(str, sorted(missing, key=str)[:5]))
        )

    df = reduce(
        lambda left, right: pd.merge(left, right, on=group), 
        (read_type_ann, summary, stats),
    )

    return df
