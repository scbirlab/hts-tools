"""Regression tests for bugs fixed in hts-tools."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import htstools as hts
from htstools.plot import _grouping_labels


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for plate in (1., 2., np.nan):  # float plate IDs with NaN
        for supp in ('none', 'TMP10'):
            for rep in (1, 2):  # numeric grouping column
                for cmpd, base, n in (('DMSO', 1., 6), ('TMP', .1, 6), ('A', .6, 2)):
                    if (plate, supp, cmpd) == (2., 'none', 'DMSO'):
                        continue  # no negative control for this group
                    for _ in range(n):
                        rows.append(dict(
                            plate=plate, supplement=supp, replicate=rep, compound=cmpd,
                            measured_abs_ch1=rng.normal(base, .05),
                            abs_ch1_wavelength='600nm',
                        ))
    return pd.DataFrame(rows)


GROUP = ['plate', 'supplement', 'replicate']


def test_grouping_labels_handle_nan_and_numbers():
    labels = _grouping_labels(_data(), GROUP + ['abs_ch1_wavelength'], 'abs_ch1')
    assert labels.str.startswith('1:none:1:').any()
    assert labels.str.startswith('NA:').any()


def test_qc_plots_with_numeric_and_nan_groups():
    d = hts.normalize(_data(), 'measured_abs_ch1', 'compound', neg='DMSO', pos='TMP', group=GROUP)
    d = d.rename(columns={'measured_abs_ch1_norm.pon': 'calc_abs_ch1_norm.pon'})
    z = hts.z_prime_factor(d, 'calc_abs_ch1_norm.pon', 'compound', 'TMP', 'DMSO', group=GROUP)
    s = hts.ssmd(d, 'calc_abs_ch1_norm.pon', 'compound', 'TMP', 'DMSO', group=GROUP)
    z = z.merge(
        s.drop(columns=[c for c in s if c in z and c not in GROUP + ['abs_ch1_wavelength']]), 
        on=GROUP + ['abs_ch1_wavelength'],
    )
    x = GROUP + ['abs_ch1_wavelength']
    for f in (hts.plot_mean_sd, hts.plot_zprime):
        fig, _ = f(z, x=x, y='abs_ch1_norm.pon')
        plt.close(fig)


def test_qc_without_group_does_not_modify_input():
    d = _data()
    before = d.columns.tolist()
    z = hts.z_prime_factor(d, 'measured_abs_ch1', 'compound', 'TMP', 'DMSO')
    assert d.columns.tolist() == before
    assert z.shape[0] == 1


def test_summarize_missing_negative_group_gives_nan():
    d = _data()
    res = hts.summarize(
        d, 'measured_abs_ch1', neg='none', control_col='supplement', 
        group=['plate', 'compound', 'supplement'],
    )
    missing = res.query('plate == 2 and compound == "DMSO"')
    assert missing.shape[0] == 1  # still reported, with NaN statistics
    assert np.isnan(missing.iloc[0]['measured_abs_ch1_t.p'])
    ok = res.query('plate == 1 and compound == "TMP" and supplement == "TMP10"').iloc[0]
    assert np.isfinite(ok['measured_abs_ch1_t.p'])


def test_summarize_lfc_is_nan_not_sentinel_for_nonpositive_mean():
    d = pd.DataFrame(dict(
        g=['a'] * 4, c=['n', 'n', 'x', 'x'], m_abs_ch1=[1., 2., 0., 0.], abs_ch1_wavelength='600nm',
    ))
    res = hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['g', 'c'])
    lfc = res.set_index('c')['m_abs_ch1_log10fc']
    assert np.isnan(lfc['x'])  # mean of 0: undefined, not -15
    assert lfc['n'] == 0.


def test_normalize_keeps_row_order_and_needs_no_positive():
    d = _data()
    n = hts.normalize(d, 'measured_abs_ch1', 'compound', neg='DMSO', group=GROUP)
    assert (n['measured_abs_ch1'].values == d['measured_abs_ch1'].values).all()


def test_replicate_table_seeded_and_complete():
    d = _data()
    group = ['plate', 'supplement', 'compound']
    a = hts.replicate_table(d, group=group, seed=1)
    b = hts.replicate_table(d, group=group, seed=1)
    assert a.shape[0] == d.shape[0]  # NaN group keys are not dropped
    assert (a['replicate'] == b['replicate']).all()
    assert a.groupby(group, dropna=False)['replicate'].max().eq(
        a.groupby(group, dropna=False).size()
    ).all()


def test_plot_replicates_control_not_in_grouping_raises_clear_error():
    d = _data().drop(columns='replicate')
    d = hts.replicate_table(d, group=['plate', 'supplement', 'compound'], seed=0)
    d = d.assign(replicate=d['replicate'].clip(upper=2))
    with pytest.raises(ValueError, match="must be one of the grouping columns"):
        hts.plot_replicates(
            d, 'measured_abs_ch1', grouping=['plate', 'supplement'],
            control_col='compound', negative='DMSO', positive='TMP',
        )


# ---- hand-calculated values ------------------------------------------------

def _tiny() -> pd.DataFrame:
    return pd.DataFrame(dict(
        c=['p', 'p', 'n', 'n', 'x', 'x', 'x'],
        m_abs_ch1=[1., 3., 5., 7., 1., 2., 3.],
        abs_ch1_wavelength='600nm',
    ))


def test_normalize_pon_and_npg_by_hand():
    d = _tiny()  # neg mean = 6, pos mean = 2
    pon = hts.normalize(d, 'm_abs_ch1', 'c', neg='n', group=None)
    assert np.allclose(pon['m_abs_ch1_norm.pon'], d['m_abs_ch1'] / 6.)
    npg = hts.normalize(d, 'm_abs_ch1', 'c', neg='n', pos='p', method='npg')
    assert np.allclose(npg['m_abs_ch1_norm.npg'], (d['m_abs_ch1'] - 2.) / 4.)


def test_ssmd_and_zprime_by_hand():
    d = _tiny()  # pos: mean 2, var 2, sd sqrt(2); neg: mean 6, var 2, sd sqrt(2)
    s = hts.ssmd(d, 'm_abs_ch1', 'c', pos='p', neg='n')
    assert np.isclose(s['m_abs_ch1_ssmd'].iloc[0], (2. - 6.) / np.sqrt(4.))  # -2
    z = hts.z_prime_factor(d, 'm_abs_ch1', 'c', pos='p', neg='n')
    assert np.isclose(z['m_abs_ch1_zprime'].iloc[0], 1. - 3. * 2. * np.sqrt(2.) / 4.)


def test_summarize_statistics_by_hand():
    d = pd.DataFrame(dict(
        c=['n'] * 3 + ['x'] * 3, m_abs_ch1=[4., 5., 6., 1., 2., 3.], abs_ch1_wavelength='600nm',
    ))
    r = hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['c']).set_index('c').loc['x']
    assert np.isclose(r['m_abs_ch1_t.stat'], -3. / np.sqrt(2. / 3.))  # Welch, both var = 1
    assert np.isclose(r['m_abs_ch1_mwu.stat'], 0.)  # complete separation
    assert np.isclose(r['m_abs_ch1_mwu.p'], 0.1)  # exact two-sided p for n = m = 3
    assert np.isclose(r['m_abs_ch1_ssmd'], -3. / np.sqrt(2.))
    assert np.isclose(r['m_abs_ch1_log10fc'], np.log10(2. / 5.))


# ---- invariants -----------------------------------------------------------

def test_multiple_wavelengths_in_group_raise():
    d = _tiny()
    d.loc[0, 'abs_ch1_wavelength'] = '700nm'
    with pytest.raises(ValueError, match="wavelength"):
        hts.normalize(d, 'm_abs_ch1', 'c', neg='n', group=None)
    with pytest.raises(ValueError, match="wavelength"):
        hts.ssmd(d, 'm_abs_ch1', 'c', pos='p', neg='n')
    with pytest.raises(ValueError, match="wavelength"):
        hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['c'])
    # fine when the grouping separates them
    d['w'] = d['abs_ch1_wavelength']
    hts.normalize(d, 'm_abs_ch1', 'c', neg='n', group=['w'])


def test_merge_on_validates_cardinality_and_conflicts():
    from htstools.cli import _merge_on
    a = pd.DataFrame(dict(g=[1, 2], w=['600nm', None], x=[1., 2.]))
    b = pd.DataFrame(dict(g=[2, 3], w=['600nm', '600nm'], y=[3., 4.]))
    m = _merge_on([a, b], ['g'])
    assert m.set_index('g')['w'].tolist() == ['600nm', '600nm', '600nm']  # coalesced
    with pytest.raises(ValueError, match="conflicting"):
        _merge_on([a.assign(w='600nm'), b.assign(w='700nm')], ['g'])
    with pytest.raises(ValueError, match="uniquely identify"):
        _merge_on([a, pd.concat([b, b])], ['g'])


def test_plot_replicates_rejects_ambiguous_grouping():
    d = _data().drop(columns='replicate')
    d = hts.replicate_table(d, group=['plate', 'supplement', 'compound'], seed=0)
    d = d.assign(replicate=d['replicate'].clip(upper=2))  # duplicate (group, replicate) pairs
    with pytest.raises(ValueError, match="uniquely identify"):
        hts.plot_replicates(
            d, 'measured_abs_ch1', grouping=['plate', 'supplement', 'compound'],
            control_col='compound', negative='DMSO', positive='TMP',
        )


def test_summarize_warns_when_negatives_all_nan(capsys):
    d = pd.DataFrame(dict(
        c=['n', 'n', 'x', 'x'], m_abs_ch1=[np.nan, np.nan, 1., 2.], abs_ch1_wavelength='600nm',
    ))
    r = hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['c'])
    assert np.isnan(r.set_index('c').loc['x', 'm_abs_ch1_t.p'])
    assert 'no matching negative controls' in capsys.readouterr().err


def test_normalize_twice_replaces_control_means():
    d = _tiny()
    once = hts.normalize(d, 'm_abs_ch1', 'c', neg='n', pos='p')
    twice = hts.normalize(once, 'm_abs_ch1', 'c', neg='n', pos='p')
    assert twice.columns.is_unique
    assert twice.shape[0] == d.shape[0]


# ---- io -------------------------------------------------------------------

def test_xlsx_zero_cells_are_preserved(tmp_path):
    from openpyxl import Workbook, load_workbook

    from htstools.io import _get_sheet_values
    wb = Workbook()
    wb.active.append(['A', 0, 0.0, None, 1.5])
    path = tmp_path / 'z.xlsx'
    wb.save(path)
    sheet = next(iter(load_workbook(path, data_only=True, read_only=True)))
    assert _get_sheet_values(sheet) == [['A', 0, 0, '', 1.5]]


# ---- CLI ------------------------------------------------------------------

def test_cli_normalize_without_positive(tmp_path):
    import subprocess
    import sys
    src = tmp_path / 'in.tsv'
    _data().to_csv(src, sep='\t', index=False)
    out = tmp_path / 'out.tsv'
    cmd = [sys.executable, '-m', 'htstools.cli', 'normalize', str(src), '-c', 'compound', 
           '-n', 'DMSO', '-g', 'plate', 'supplement', '-o', str(out)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert 'calc_abs_ch1_norm.pon' in pd.read_csv(out, sep='\t').columns


# ---- second-round review --------------------------------------------------

def test_summarize_rejects_cross_wavelength_negative_comparison():
    d = pd.DataFrame(dict(
        plate=[1] * 4, compound=['DMSO', 'DMSO', 'A', 'A'], m_abs_ch1=[1., 1.1, .5, .6],
        abs_ch1_wavelength=['600nm', '600nm', '700nm', '700nm'],
    ))
    # group is consistent in wavelength, but A would be compared with DMSO read at another wavelength
    with pytest.raises(ValueError, match="wavelength"):
        hts.summarize(d, 'm_abs_ch1', neg='DMSO', control_col='compound', group=['plate', 'compound'])
    # grouping by wavelength is the fix
    r = hts.summarize(
        d, 'm_abs_ch1', neg='DMSO', control_col='compound', 
        group=['plate', 'compound', 'abs_ch1_wavelength'],
    )
    assert r.columns.is_unique


def test_wavelength_in_grouping_gives_no_duplicate_columns():
    d = _tiny()
    z = hts.z_prime_factor(d, 'm_abs_ch1', 'c', pos='p', neg='n', group=['abs_ch1_wavelength'])
    assert z.columns.is_unique and z.shape[0] == 1


def test_normalization_undefined_values_are_nan_not_inf(capsys):
    d = pd.DataFrame(dict(
        c=['p', 'p', 'n', 'n', 'x'], m_abs_ch1=[1., 1., 0., 0., 3.], abs_ch1_wavelength='600nm',
    ))
    pon = hts.normalize(d, 'm_abs_ch1', 'c', neg='n')
    assert pon['m_abs_ch1_norm.pon'].isna().all() and not np.isinf(pon['m_abs_ch1_norm.pon']).any()
    assert 'PON values there are NaN' in capsys.readouterr().err
    # NPG is well defined when the negative mean is zero but the positive mean is not
    npg = hts.normalize(d, 'm_abs_ch1', 'c', neg='n', pos='p', method='npg')
    assert np.allclose(npg['m_abs_ch1_norm.npg'], (d['m_abs_ch1'] - 1.) / -1.)
    assert capsys.readouterr().err == ''
    # identical control means: NPG undefined
    same = d.assign(m_abs_ch1=1.)
    npg = hts.normalize(same, 'm_abs_ch1', 'c', neg='n', pos='p', method='npg')
    assert npg['m_abs_ch1_norm.npg'].isna().all()


def test_summarize_ignores_infinite_values():
    d = pd.DataFrame(dict(
        c=['n', 'n', 'n', 'x', 'x', 'x'], m_abs_ch1=[4., 5., 6., 1., 2., np.inf], abs_ch1_wavelength='600nm',
    ))
    r = hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['c']).set_index('c').loc['x']
    assert np.isfinite(r['m_abs_ch1_t.stat'])
    # descriptive statistics use the same observations as the comparisons
    assert r['m_abs_ch1_count'] == 2 and np.isclose(r['m_abs_ch1_mean'], 1.5)
    assert np.isclose(r['m_abs_ch1_log10fc'], np.log10(1.5 / 5.))


def test_summarize_all_infinite_negatives_warn(capsys):
    d = pd.DataFrame(dict(
        c=['n', 'n', 'x', 'x'], m_abs_ch1=[np.inf, np.inf, 1., 2.], abs_ch1_wavelength='600nm',
    ))
    hts.summarize(d, 'm_abs_ch1', neg='n', control_col='c', group=['c'])
    assert 'no matching negative controls' in capsys.readouterr().err


def test_wide_replicates_rejects_placeholder_collision():
    from htstools.tables import wide_replicates
    d = pd.DataFrame(dict(g=['__NA__', 'a'], replicate=[1, 1], v=[1., 2.]))
    with pytest.raises(ValueError, match="reserved"):
        wide_replicates(d, 'g', 'v')


def test_histogram_bins_cover_nonpositive_values():
    from htstools.plot import _log_like_bins
    v = pd.Series([-2., -.5, 0., .1, 3.])
    bins = _log_like_bins(v, 5, all_positive=False, linthresh=.1)
    assert bins[0] <= v.min() and bins[-1] >= v.max() and np.all(np.diff(bins) > 0)
    v = pd.Series([.1, 1., 10.])
    assert np.allclose(_log_like_bins(v, 3, True, 1.), [.1, 1., 10.])
    assert _log_like_bins(pd.Series([1.]), 5, True, 1.) == 5


def test_nan_grouping_keys_are_kept_everywhere():
    d = _data()
    n_nan_groups = d[d['plate'].isna()].groupby(['supplement', 'compound']).ngroups
    norm = hts.normalize(d, 'measured_abs_ch1', 'compound', neg='DMSO', pos='TMP', group=['plate', 'supplement'])
    nan_rows = norm['plate'].isna()
    assert nan_rows.any() and norm.loc[nan_rows, 'measured_abs_ch1_norm.pon'].notna().all()
    s = hts.summarize(d, 'measured_abs_ch1', neg='none', control_col='supplement', group=['plate', 'compound', 'supplement'])
    assert s['plate'].isna().sum() == n_nan_groups
    assert s.loc[s['plate'].isna() & (s['supplement'] == 'TMP10'), 'measured_abs_ch1_t.p'].notna().all()
    q = hts.z_prime_factor(d, 'measured_abs_ch1', 'compound', 'TMP', 'DMSO', group=['plate', 'supplement'])
    assert q.loc[q['plate'].isna(), 'measured_abs_ch1_zprime'].notna().all()
    w = hts.replicate_table(d.drop(columns='replicate'), group=['plate', 'supplement', 'compound'], wide='measured_abs_ch1', seed=0)
    assert w.index.get_level_values('plate').isna().any()


def test_plot_replicates_runs_with_nan_keys():
    d = _data().drop(columns='replicate')
    d = hts.replicate_table(d, group=['plate', 'supplement', 'compound'], seed=0)
    fig, _ = hts.plot_replicates(
        d, 'measured_abs_ch1', grouping=['plate', 'supplement', 'compound'],
        control_col='compound', negative='DMSO', positive='TMP',
    )
    plt.close(fig)


def test_biotek_row_overflow_becomes_nan_and_zero_is_kept():
    from htstools.io import BiotekData, _biotek_row
    data = BiotekData(
        Meta={'Plate Number': [['1']]}, Procedure_Details={}, Layout={},
        Results={'main': [
            ['Well', 'Well ID', 'Read 1:600'],
            ['A1', 'x', '0.5'], ['A2', 'y', 'OVRFLW'], ['A3', 'z', '0'],
        ]},
    )
    df, _ = _biotek_row(data, filename='delim:f.csv:s')
    assert df['measured_abs_ch1'].isna().tolist() == [False, True, False]
    assert df['measured_abs_ch1'].iloc[2] == 0.


def test_allow_mixed_wavelengths_gives_one_row_per_group():
    d = pd.DataFrame(dict(
        s=['WT', 'WT', 'mut', 'mut'], m_fluor_ch1=[1., 2., 3., 4.],
        fluor_ch1_wavelength=['ex1;em1', 'ex1;em1', 'ex2;em2', 'ex2;em2'],
    ))
    with pytest.raises(ValueError, match="wavelength"):
        hts.summarize(d, 'm_fluor_ch1', neg='WT', control_col='s', group=['s'])
    with pytest.warns(UserWarning):
        r = hts.summarize(d, 'm_fluor_ch1', neg='WT', control_col='s', group=['s'], allow_mixed_wavelengths=True)
    assert r.shape[0] == 2  # not duplicated by the wavelength annotation
    with pytest.warns(UserWarning, match="mixed wavelengths were explicitly allowed"):
        hts.summarize(d, 'm_fluor_ch1', neg='WT', control_col='s', group=['s'], allow_mixed_wavelengths=True)
    with pytest.raises(TypeError):
        hts.normalize(d, 'm_fluor_ch1', 's', neg='WT', method='pon', allow_mixed_wavelengths=True)
    with pytest.warns(UserWarning):
        z = hts.z_prime_factor(
            d.assign(c=['p', 'n', 'p', 'n']), 'm_fluor_ch1', 'c', pos='p', neg='n', allow_mixed_wavelengths=True,
        )
    assert z.shape[0] == 1 and z['fluor_ch1_wavelength'].iloc[0] == 'ex1;em1|ex2;em2'
