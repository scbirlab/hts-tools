"""Untilities for generating plots."""

from collections.abc import Iterable
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from carabiner.mpl import add_legend, grid
from matplotlib import axes, figure
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer
from pandas import DataFrame
from scipy import stats

from .tables import wide_replicates
from .utils import as_list

PlotTuple = tuple[figure.Figure, axes.Axes]
"""A figure and its single set of axes, as returned by most plotting functions."""
PlotGridTuple = tuple[figure.Figure, np.ndarray]
"""A figure and an array of its axes, as returned by multi-panel plotting functions."""

_DARK_GREY = "dimgrey"
_LIGHT_GREY = "lightgrey"
_MARKER_SIZE = 3.
_PANEL_SIZE = 2.5

def _format_label_part(values: pd.Series) -> pd.Series:
    """Convert a column to str for use in labels, tolerating NaN and floats-that-are-ints."""
    if pd.api.types.is_float_dtype(values):
        return values.map(
            lambda v: 'NA' if pd.isna(v) 
            else str(int(v)) if float(v).is_integer() 
            else str(v)
        )
    return values.astype(str)


def _grouping_labels(
    data: DataFrame, 
    x: Iterable[str],
    y: str | None = None
) -> pd.Series:
    """Join grouping columns into one str label per row.
    
    Columns can be of any type (including numeric columns with NaN). 
    "_wavelength" columns are only used if they belong to the 
    measurement `y`.

    """
    x = as_list(x)
    if y is not None:
        channel = '_'.join(y.split('_')[:2]) + '_wavelength'
        x = [x_ for x_ in x if not x_.endswith('wavelength') or x_ == channel]
    labels = [_format_label_part(data[x_]) for x_ in x]
    if len(labels) == 0:
        return pd.Series('', index=data.index)
    return reduce(lambda a, b: a + ':' + b, labels)


def _plot_errbars(
    ax: axes.Axes,
    data: DataFrame,
    x: str, 
    y: str,
    ci: float = .95,
    **kwargs
) -> tuple[axes.Axes, ErrorbarContainer]:
    
    grouped = data.groupby([x])[[y]]
    
    this_mean = grouped.mean().reset_index()
    this_sd = grouped.sem().reset_index()
    ci = stats.norm.interval(ci)[-1]

    y_is_all_nan = np.all(np.isnan(this_sd[y]))
    errbar = ax.errorbar(
        this_mean[x], 
        this_mean[y], 
        yerr=(this_sd[y] * ci) if not y_is_all_nan else 0., 
        fmt='o-', 
        **kwargs
    )
    return ax, errbar


def _plot_scatter_open_circles(
    ax: axes.Axes,
    data: DataFrame,
    x: str, 
    y: str,
    color: str,
    **kwargs
) -> tuple[axes.Axes, PathCollection]:
    scatter = ax.scatter(
        data[x], 
        data[y], 
        edgecolors=color, 
        facecolors='none',
        **kwargs
    )
    return ax, scatter


def _plot_mean_and_scatter(
    ax: axes.Axes,
    data: DataFrame,
    x: str, 
    y: str,
    color: str,
    ci: float = .95,
    s: float = _MARKER_SIZE,
    zorder: int = 0,
    **kwargs
) -> tuple[axes.Axes, PathCollection, ErrorbarContainer]:
    ax, errbars = _plot_errbars(
        ax, 
        data, 
        x, 
        y, 
        ci=ci,
        color=color,
        markerfacecolor=color,
        markersize=s,
        zorder=zorder,
        **kwargs
    )
    ax, scatter = _plot_scatter_open_circles(
        ax, 
        data, 
        x, 
        y, 
        color=color, 
        s=s * 3., 
        zorder=zorder,
        label='_default'
    )
    return ax, scatter, errbars
    

def plot_dose_response(
    data: DataFrame,
    x: str, 
    y: str, 
    file_prefix: str,
    color: str | None = None, 
    color_control: str | None = None, 
    facet: str | None = None,
    files: str | None = None,
    hlines: Iterable[float] | None = None,
    panel_size: float = _PANEL_SIZE,
    format: str = 'pdf',
    sharey: bool = False,
    sharex: bool = False,
    x_log: bool = False,
    y_log: bool = False
) -> list[str]:

    """Plot dose response curves, optionally splitting data across files, facets and colors.
    
    This is a flexible function for data exploration and presentation. Uses a color-blind friendly palette.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data in columnar format.
    x : str
        Column to use as x-axis.
    y: str
        Column to use as y-axis.
    file_prefix : str
        Prefix to use in output filenames.
    color : str, optional
        If provided, use this column to split data into separate colored lines.
    color_control : str, optional
        If provided, plot this value from the color column as a dark grey.
    facet : str, optional
        If provided, split plots into separate facets (panels) based on this column.
    files : str, optional
        If provided, split plots into separate files based on this column.
    hlines : list of float, optional
        Plot horizontal guidelines at these y-intercepts. Default: [0.].
    panel_size : float, optional
        Size of a single panel (facet) in inches. Default: 3.0.
    format : str, optional
        File format to save plots. Default: "pdf".
    sharex : bool, optional
        Whether to have shared x-axis ranges. Default: False.
    sharey : bool, optional
        Whether to have shared y-axis ranges. Default: False.
    x_log : bool, optional
        Whether to make x-axis log scale. Default: False.
    y_log : bool, optional
        Whether to make y-axis log scale. Default: False.

    Returns
    -------
    list
        Filenames in which plots were saved.

    """
    
    if format.lower() not in ('pdf', 'png'):
        raise NotImplementedError(f"Saving as {format=} not implemented")
    
    DEFAULT = '__default__'
    filenames = []
    capsize, markersize = 2., _MARKER_SIZE
    data = data.assign(**{DEFAULT: DEFAULT})

    color = color or DEFAULT
    facet = facet or DEFAULT
    files = files or DEFAULT
    hlines = hlines or []

    for dim in dict.fromkeys((color, facet, files)):
        data = data[data[dim] != "blank"]

    n_facets = data[facet].unique().size
    n_facet_rows = int(np.ceil(np.sqrt(n_facets)))
    n_facet_cols = int(np.ceil(n_facets / n_facet_rows))

    assert (n_facet_cols * n_facet_rows) >= n_facets, f"{n_facets} != {n_facet_rows} x {n_facet_cols}"

    for file_name, file_data in (
        data.sort_values(by=x).groupby(files, dropna=False)
    ):
        if files == DEFAULT:
            filename = f'{file_prefix}.{format}'
        else:
            filename = f'{file_prefix}_{files}={file_name}.{format}'

        fig, _ = grid(
            nrow=n_facet_rows, 
            ncol=n_facet_cols, 
            aspect_ratio=1.1,
            sharey=sharey,
            sharex=sharex,
        )

        n_panels_used = 0
        for ax, (facet_name, facet_data) in zip(fig.axes, file_data.groupby(facet, dropna=False)):
            n_panels_used += 1
            ax.set(
                title=facet_name if facet_name != DEFAULT else '',
                xlabel=x, 
                ylabel=y,
                xscale='log' if x_log else "linear",
                yscale='log' if y_log else "linear",
            )

            for i in hlines:
                ax.axhline(i, c=_LIGHT_GREY, zorder=-5)

            if (
                color_control is not None and 
                color_control in facet_data[color].values
            ):
                control_data = facet_data[facet_data[color] == color_control]
                _plot_mean_and_scatter(
                    ax, 
                    control_data, 
                    x, 
                    y, 
                    color=_DARK_GREY,
                    s=markersize, 
                    capsize=capsize, 
                    zorder=5,
                )

            for i, (color_name, color_data) in enumerate(facet_data.groupby(color, dropna=False)):
                if color_name != color_control:
                    _plot_mean_and_scatter(
                        ax, 
                        color_data, 
                        x, 
                        y, 
                        color=f"C{i}" if color_name != DEFAULT else "C1",
                        s=markersize, 
                        capsize=capsize,
                        label=color_name if color_name != DEFAULT else '',
                    )

        for unused_ax in fig.axes[n_panels_used:]:
            unused_ax.set_visible(False)
        add_legend(ax)
        fig.savefig(
            filename, 
            dpi=300 if format.lower() == 'png' else 'figure',
            bbox_inches='tight',
        )
        plt.close(fig)
        filenames.append(filename)
    return filenames


def plot_mean_sd(
    data: DataFrame,
    x: str | Iterable[str], 
    y: str, 
    panel_size: float = _PANEL_SIZE
) -> PlotTuple:

    x = as_list(x)
    y_pos_mean, y_pos_sd = (
        'calc_' + y + suffix for suffix in ('_pos_mean', '_pos_sd')
    )
    y_neg_mean, y_neg_sd = (
        'calc_' + y + suffix for suffix in ('_neg_mean', '_neg_sd')
    )

    data_to_plot = data[x + [y_pos_mean, y_pos_sd, y_neg_mean, y_neg_sd]].copy()
    data_to_plot['grouping'] = _grouping_labels(data_to_plot, x, y)

    data_to_plot = data_to_plot.sort_values('grouping')

    fig, ax = grid(
        panel_size=panel_size / .7, 
        aspect_ratio=.7,
    )
    # Numeric positions with explicit tick labels behave the same across matplotlib versions
    positions = np.arange(data_to_plot.shape[0])
    for m, s, label in zip((y_pos_mean, y_neg_mean), (y_pos_sd, y_neg_sd), ("Positive", "Negative")):
        mean = data_to_plot[m].to_numpy(dtype=float)
        sd = data_to_plot[s].to_numpy(dtype=float)
        line, = ax.plot(positions, mean, label=label)
        ax.fill_between(
            positions, 
            mean - sd, 
            mean + sd,
            color=line.get_color(),
            alpha=.7,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(data_to_plot['grouping'].astype(str).tolist())
    ax.tick_params(
        axis='x', 
        labelrotation=90, 
        labelsize='small',
    )
    ax.set(
        xlabel='Batch', 
        ylabel=y,
    )
    add_legend(ax)
    return fig, ax


def plot_zprime(
    data: DataFrame,
    x: str | Iterable[str], 
    y: str, 
    panel_size: float = _PANEL_SIZE * 1.8
) -> PlotTuple:
    x = as_list(x)
    fig, ax = grid(
        panel_size=panel_size, 
        aspect_ratio=.8,
    )
    markersize = _MARKER_SIZE * 2.
    y_ = ('calc_' + y + '_zprime')

    data_to_plot = data[x + [y_]].copy()
    data_to_plot['grouping'] = _grouping_labels(data_to_plot, x, y)
    data_to_plot = data_to_plot.sort_values('grouping')

    ax.plot(
        'grouping', y_, 
        data=data_to_plot,
    )
    ax.scatter(
        'grouping', y_, 
        data=data_to_plot,
        s=markersize,
    )

    for n in np.linspace(0., 1., num=3):
        ax.axhline(n, color='lightgray')

    ax.tick_params(
        axis='x', 
        labelrotation=90, 
        labelsize='small',
    )
    ax.set(
        xlabel='Batch', 
        ylabel=y + '_zprime',
    )
    return fig, ax


def plot_heatmap(
    data: DataFrame,
    x: str | Iterable[str],
    y: str,
    panel_size: float = _PANEL_SIZE
) -> PlotGridTuple:
    
    x = as_list(x)
    plates = data[x].drop_duplicates()
    n_plates = plates.shape[0]

    n_cols = int(np.ceil(np.sqrt(n_plates)))
    n_rows = int(np.ceil(n_plates / n_cols))

    n_panels = n_cols * n_rows

    if not n_panels >= n_plates:
        ## This should never happen!
        raise ValueError(
            f"ERROR: Number of heatmap panels ({n_panels}) is less than the number of plates ({n_plates})!")

    fig, axes = grid(
        nrow=n_rows, 
        ncol=n_cols,
    )
    for ax in fig.axes[n_plates:]:
        ax.set_visible(False)

    for ax, (plate_name, plate_data) in zip(fig.axes, data.groupby(x, dropna=False)):
        this_plate_data = pd.pivot_table(
            plate_data, 
            index='row_id', 
            columns='column_id',
            values=y,
        )
        axes_image = ax.imshow(
            this_plate_data,
            cmap='cividis',
        )
        cbar = fig.colorbar(
            axes_image, 
            ax=ax, 
            shrink=.5,
            orientation='vertical',
        )
        cbar.ax.tick_params(labelsize='xx-small')

        ax.set_yticks(
            np.arange(this_plate_data.index.values.size), 
            labels=this_plate_data.index.values,
            size='x-small',
        )
        x_locs = [
            x for x in range(this_plate_data.columns.values.size) if x % 2 == 0
        ]
        ax.set_xticks(
            x_locs, 
            labels=[str(this_plate_data.columns.values[loc]) for loc in x_locs],
            size='x-small',
        )
        ax.set_title(
            ':'.join(map(str, plate_name if isinstance(plate_name, tuple) else (plate_name, ))),
            fontdict={'fontsize': 8.},
        )
        
    return fig, axes


def _log_like_bins(
    values: pd.Series, 
    n_bins: int, 
    all_positive: bool, 
    linthresh: float
):
    """Bin edges evenly spaced on a log axis or, if some values are <= 0, evenly spaced 
    in the inverse hyperbolic sine of the data (approximating a symmetric-log axis).

    The bins always span the full range of `values`. Returns `n_bins` if no
    range exists to make bins from.

    """
    if values.nunique() < 2:
        return n_bins
    low, high = values.min(), values.max()
    if all_positive:
        return np.geomspace(low, high, num=n_bins)
    bins = linthresh * np.sinh(
        np.linspace(np.arcsinh(low / linthresh), np.arcsinh(high / linthresh), num=n_bins)
    )
    bins[0], bins[-1] = low, high  # undo floating-point drift so no data fall outside the bins
    return bins


def plot_histogram(
    data: DataFrame,
    x: str,
    control_col: str,
    negative: str,
    positive: str,
    panel_size: float = _PANEL_SIZE
) -> PlotTuple:
    
    read_type = '_'.join(x.split('_')[1:3]) + '_wavelength'

    if read_type not in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    data = data[data[read_type] != '']
    data = data[[control_col, read_type, x]].dropna().copy()
    data = data[~np.isinf(data[x])].copy() 
    n_read_types = data[read_type].unique().size
    
    n_cols = 4
    fig, axes = grid(
        nrow=n_read_types, 
        ncol=n_cols, 
        squeeze=False,
    )

    for row, (wv, wv_df) in zip(axes, data.groupby(read_type)):
        # zero or negative values need a symmetric-log axis, with its own bins
        all_positive = bool((wv_df[x] > 0.).all())
        nonzero = wv_df[x][wv_df[x] != 0.].abs()
        linthresh = float(nonzero.quantile(.05)) if nonzero.size > 0 else 1.
        for i, control in enumerate((positive, negative, None)):
            if control is not None:
                this_data = wv_df[wv_df[control_col] == control]
                axes = row[0], row[2]
                title = 'Controls'
            else:
                this_data = wv_df[~wv_df[control_col].isin([negative, positive])]
                axes = row[1], row[3]
                title = 'Experiment'
            
            n_bins = 5 + this_data.shape[0] // 20
            bins_log = _log_like_bins(this_data[x], n_bins, all_positive, linthresh)

            for ax, b in zip(axes, (n_bins, bins_log)):
                ax.hist(
                    x, 
                    data=this_data, 
                    density=False,
                    histtype='stepfilled',
                    bins=b,
                    zorder=3 - i,
                )
            
                if '_norm.' in x:
                    for m in (0., 1.):
                        ax.axvline(
                            m, 
                            color='lightgray', 
                            zorder=0,
                        )

                ax.set(
                    xlabel='_'.join(x.split('_')[1:]) + f': {wv}',
                    ylabel='Frequency',
                    title=title,
                )
            # symlog handles zero or negative values, which a log axis cannot show
            if all_positive:
                ax.set_xscale('log')
            else:
                ax.set_xscale('symlog', linthresh=linthresh)
            if title == "Controls":
                add_legend(ax)
    return fig, ax


def plot_replicates(
    data: DataFrame,
    x: str,
    grouping: str | Iterable[str],
    control_col: str,
    negative: str,
    positive: str,
    panel_size: float = _PANEL_SIZE
) -> PlotGridTuple:

    grouping = as_list(grouping)
    if control_col not in grouping:
        raise ValueError(
            f"The control column '{control_col}' must be one of the grouping columns: {grouping}"
        )
    read_type = '_'.join(x.split('_')[1:3]) + '_wavelength'
    if read_type not in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    n_read_types = data[read_type].unique().size
   
    do_log = True #n_gt_zero > .5
    n_cols = 2 if do_log else 1
    n_rows = n_read_types
    
    fig, axes = grid(
        nrow=n_rows, 
        ncol=n_cols, 
        squeeze=False,
    )
    for row, (wv, wv_df) in zip(axes, data.groupby(read_type)):
        this_title = '_'.join(x.split('_')[1:]) + f': {wv}'
        n_gt_zero = np.mean(wv_df[x] > 0)
        sub_df = (
            wv_df
            .query('replicate < 3')
        )
        if sub_df.duplicated(grouping + ['replicate']).any():
            raise ValueError(
                f"Grouping columns {grouping} do not uniquely identify a measurement within "
                "each replicate: pooling them would average different measurements. "
                "Add more grouping columns (e.g. compound or well)."
            )
        df_wide = wide_replicates(sub_df, grouping, x)
        if not {'rep_1', 'rep_2'} <= set(df_wide.columns):
            raise ValueError(
                f"Need at least 2 replicates per group to plot replicate correlations, "
                f"found columns {df_wide.columns.tolist()}."
            )
        pearson_corr = float(df_wide[['rep_1', 'rep_2']].corr().to_numpy()[0, 1])  # ignores NaNs pairwise

        control_values = pd.Series(
            df_wide.index.get_level_values(control_col), 
            index=df_wide.index,
        )

        for ax in row:
            for i, control in enumerate((positive, negative, None)):
                if control is not None:
                    this_data = df_wide[control_values == control]
                else:
                    this_data = df_wide[~control_values.isin([negative, positive]).to_numpy()]
                ax.scatter(
                    'rep_1', 'rep_2', 
                    data=this_data, 
                    s=_MARKER_SIZE + (6. if control is not None else 0.),
                    zorder=3 - i,
                )
    
            ax.plot(
                ax.get_xlim(), ax.get_xlim(), 
                color='lightgray',
                zorder=0,
            )
            ax.set(
                # aspect='equal', 
                xlabel='Replicate 1',
                ylabel='Replicate 2',
            )
            ax.set_title(
                this_title + f'\nr = {pearson_corr:.2f}',
                fontdict={'fontsize': 8.},
            )

        if do_log:
            df_gt_zero = df_wide.query('rep_1 > 0 and rep_2 > 0')
            pearson_corr = float(df_gt_zero[['rep_1', 'rep_2']].apply(np.log).corr().to_numpy()[0, 1])
            ax.set(
                xscale='log', 
                yscale='log',
            )
            ax.set_title(
                this_title + f'\nr = {pearson_corr:.2f} | hidden points: {100. * (1. - n_gt_zero):.1f} %', 
                fontdict={'fontsize': 8.},
            )

    return fig, axes


def plot_scatter(
    data: DataFrame,
    measurement_col: str,
    x: str, 
    y: str, 
    color: str | None = None,
    log_color: bool = False,
    hlines: Iterable[float] | None = None,
    vlines: Iterable[float] | None = None,
    x_log: bool = False,
    y_log: bool = False,
    panel_size: float = _PANEL_SIZE,
    **kwargs
) -> PlotTuple:
    
    hlines = hlines or []
    vlines = vlines or []
    data = data.copy()
    fig, ax = grid(
        panel_size=panel_size,
        **kwargs
    )
    x_col, y_col, c_col = (
        ('calc_' + measurement_col + dimension) 
        if dimension is not None else dimension 
        for dimension in (x, y, color)
    )

    sc = ax.scatter(
        x_col, 
        y_col, 
        c=data[c_col] if c_col is not None else c_col,
        cmap="cividis" if c_col is not None else None,
        norm="log" if log_color else None,
        data=data,
        s=_MARKER_SIZE,
        zorder=3,
    )
    
    if c_col is not None:
        cbar = fig.colorbar(sc)
        cbar.set_label(c_col.split("_")[-1])

    for y_ in hlines:
        ax.axhline(
            y_, 
            color=_LIGHT_GREY, 
            zorder=0,
        )
        
    for x_ in vlines:
        ax.axvline(
            x_, 
            color=_LIGHT_GREY, 
            zorder=0,
        )
    
    ax.set(
        xscale="log" if x_log else "linear", 
        yscale="log" if y_log else "linear",
        xlabel=x.split("_")[-1], 
        ylabel=y.split("_")[-1], 
        title=measurement_col,
    )
    return fig, ax

