"""Functions for generating plots."""

from typing import List, Tuple, Union
from collections.abc import Iterable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axes, figure
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import PathCollection
import numpy as np
from scipy import stats

from .utils import cbpal

plt.rcParams["font.family"] = 'DejaVu Sans'

PlotTuple = Tuple[axes.Axes, figure.Figure]

def _plot_errbars(ax: axes.Axes,
                  data: pd.DataFrame,
                  x: str, 
                  y: str,
                  ci: float = .95,
                  **kwargs) -> ErrorbarContainer:
    
    grouped = data.groupby([x])[[y]]
    
    this_mean = grouped.mean().reset_index()
    this_sd = grouped.sem().reset_index()

    ci = stats.norm.interval(ci)[-1]

    y_is_all_nan = np.all(np.isnan(this_sd[y]))

    errbar = ax.errorbar(this_mean[x], this_mean[y], 
                         yerr=(this_sd[y] * ci) if not y_is_all_nan else 0., 
                         fmt='o-', 
                         **kwargs)

    return ax, errbar


def _plot_scatter_open_circles(ax: axes.Axes,
                               data: pd.DataFrame,
                               x: str, 
                               y: str,
                               color: str,
                               **kwargs) -> PathCollection:
    
    scatter = ax.scatter(x, y, 
                      data=data,
                      edgecolors=color, 
                      facecolors='none',
                      **kwargs)

    return ax, scatter


def _plot_mean_and_scatter(ax: axes.Axes,
                           data: pd.DataFrame,
                           x: str, 
                           y: str,
                           color: str,
                           ci: float = .95,
                           s: float = 2.,
                           zorder: int = 0,
                           **kwargs) -> Tuple[PathCollection, ErrorbarContainer]:

    ax, errbars = _plot_errbars(ax, data, 
                                x, y, 
                                ci=ci,
                                color=color,
                                markerfacecolor=color,
                                markersize=s,
                                zorder=zorder,
                                **kwargs)
    ax, scatter = _plot_scatter_open_circles(ax, data, x, y, 
                                             color=color, 
                                             s=s * 1.5, 
                                             zorder=zorder,
                                             label='_default')

    return scatter, errbars
    

def plot_dose_response(data: pd.DataFrame,
                       x: str, 
                       y: str, 
                       file_prefix: str,
                       color: Union[str, None] = None, 
                       color_control: Union[str, None] = None, 
                       facet: Union[str, None] = None,
                       files: Union[str, None] = None,
                       hlines: List[float] = [0.],
                       panel_size: float = 3.,
                       format: str = 'pdf') -> List[str]:

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

    Returns
    -------
    list
        Filenames in which plots were saved.

    """
    
    if format.lower() not in ('pdf', 'png'):

        raise NotImplementedError(f"Saving as {format=} not implemented")
    
    DEFAULT = '__default__'

    filenames = []

    capsize, markersize = 2., 3.
    
    data[DEFAULT] = ''

    color = color or DEFAULT
    facet = facet or DEFAULT
    files = files or DEFAULT

    for dim in (color, facet, files):

        data = data.query(f'{dim} != "blank"')

    n_facets = data[facet].unique().size
    n_facet_rows = int(np.ceil(np.sqrt(n_facets)))
    n_facet_cols = int(np.ceil(n_facets / n_facet_rows))

    assert (n_facet_cols * n_facet_rows) >= n_facets, \
        f"{n_facets} != {n_facet_rows} x {n_facet_cols}"

    for file_name, file_data in (data.sort_values(by=x)
                                     .groupby(files)):

        if files == DEFAULT:
            filename = f'{file_prefix}.{format}'
        else:
            filename = f'{file_prefix}_{files}={file_name}.{format}'

        fig, _ = plt.subplots(n_facet_rows, n_facet_cols, 
                              figsize=(n_facet_cols * panel_size + 2., 
                                       panel_size * n_facet_rows),
                              sharey=True)

        for ax, (facet_name, facet_data) in zip(fig.axes, file_data.groupby(facet)):

            ax.set(title=facet_name if facet_name != DEFAULT else '',
                   xlabel=x, 
                   ylabel=y,
                   xscale='log')

            ax.tick_params(axis='y', 
                           labelleft=True)

            for i in hlines:
                ax.axhline(i, c='gainsboro', zorder=-5)

            if (color_control is not None and 
                color_control in facet_data[color].values):
                
                control_data = facet_data.query(f'{color} == "{color_control}"')
                
                _plot_mean_and_scatter(ax, control_data, 
                                       x, y, 
                                       color='dimgray',
                                       s=markersize, 
                                       capsize=capsize, 
                                       zorder=5)

            for (color_name, color_data), c in zip(facet_data.groupby(color), cbpal):

                if color_name != color_control:

                    _plot_mean_and_scatter(ax, color_data, 
                                        x, y, 
                                        color=c if color_name != DEFAULT else 'dimgray',
                                        s=markersize, 
                                        capsize=capsize,
                                        label=color_name if color_name != DEFAULT else '')

        ax.legend(bbox_to_anchor=(1.1, 1.05))

        fig.tight_layout()
        fig.savefig(filename, 
                    dpi=300 if format.lower() == 'png' else 'figure',
                    bbox_inches='tight')

        filenames.append(filename)

    return filenames


def plot_mean_sd(data: pd.DataFrame,
                 x: Union[list, str], 
                 y: str, 
                 panel_size: float = 3.) -> PlotTuple:

    if isinstance(x, str):
        x = [x]

    y_pos_mean, y_pos_sd = ('calc_' + y + suffix 
                            for suffix in ('_pos_mean', '_pos_sd'))
    y_neg_mean, y_neg_sd = ('calc_' + y + suffix 
                            for suffix in ('_neg_mean', '_neg_sd'))

    data_to_plot = data[x + [y_pos_mean, y_pos_sd, y_neg_mean, y_neg_sd]].copy()
    data_to_plot['grouping'] = data_to_plot[x[0]].str.cat(data_to_plot[x[1:]], sep=':')

    data_to_plot = data_to_plot.sort_values('grouping')

    fig, ax = plt.subplots(figsize=(panel_size, panel_size))

    for m, s in zip((y_pos_mean, y_neg_mean), (y_pos_sd, y_neg_sd)):

        y1 = data_to_plot[m] - data_to_plot[s]
        y2 = data_to_plot[m] + data_to_plot[s]

        ax.plot('grouping', m, 
                data=data_to_plot,
                color='dimgray')
        ax.fill_between('grouping', y1, y2,
                        data=data_to_plot,
                        color='lightgray')
        
    ax.tick_params(axis='x', 
                   labelrotation=90, 
                   labelsize='small')
    ax.set(xlabel='Batch', 
           ylabel=y)

    fig.tight_layout()

    return fig, ax


def plot_zprime(data: pd.DataFrame,
                x: list, 
                y: list, 
                panel_size: float = 3.) -> PlotTuple:

    fig, ax = plt.subplots(figsize=(panel_size, panel_size))

    markersize = 3.

    y_ = ('calc_' + y + '_zprime')

    data_to_plot = data[x + [y_]].copy()
    x = [x_ for x_ in x 
         if not x_.endswith('wavelength') 
         or y in x_]
    data_to_plot['grouping'] = data_to_plot[x[0]].str.cat(data_to_plot[x[1:]], sep=':')

    data_to_plot = data_to_plot.sort_values('grouping')

    ax.plot('grouping', y_, 
            data=data_to_plot,
            color='dimgray')
    ax.scatter('grouping', y_, 
            data=data_to_plot,
            color='dimgray',
            s=markersize)

    for n in (0., .5,  1.):
        ax.axhline(n, color='lightgray')

    ax.tick_params(axis='x', 
                   labelrotation=90, 
                   labelsize='small')
    ax.set(xlabel='Batch', 
           ylabel=y + '_zprime')

    fig.tight_layout()

    return fig, ax


def plot_heatmap(data: pd.DataFrame,
                 x: list,
                 y: str,
                 panel_size: float = 2.5) -> PlotTuple:
    
    plates = data[x].drop_duplicates()
    n_plates = plates.shape[0]

    n_cols = int(np.ceil(n_plates ** .5))
    n_rows = int(n_plates // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(n_cols * panel_size, 
                                      n_rows * panel_size))

    for ax, (plate_name, plate_data) in zip(fig.axes, data.groupby(x)):

        this_plate_data = pd.pivot_table(plate_data, 
                                         index='row_id', 
                                         columns='column_id',
                                         values=y)

        ax.imshow(this_plate_data)
        ax.set_title(':'.join(plate_name), 
                     fontdict={'fontsize': 8.})

    fig.tight_layout()

    return fig, axes


def plot_histogram(data: pd.DataFrame,
                   x: str,
                   control_col: str,
                   negative: str,
                   positive: str,
                   panel_size: float = 2.5) -> PlotTuple:
    
    read_type = '_'.join(x.split('_')[1:3]) + '_wavelength'
    data = data.query(f"{read_type} != ''")

    if not read_type in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    n_read_types = data[read_type].unique().size

    data = data[[control_col, read_type, x]].dropna().copy()
    data = data[~np.isinf(data[x])].copy() 
    
    n_cols = 4
    fig, axes = plt.subplots(n_read_types, n_cols, 
                            figsize=(n_cols * panel_size, 
                                     n_read_types * panel_size),
                            # sharex=True,
                            squeeze=False)

    for row, (wv, wv_df) in zip(axes, data.groupby(read_type)):

        for i, (control, color) in enumerate(zip((positive, negative, None), 
                                                cbpal)):

            if control is not None:
                q = f'{control_col} == "{control}"'
                axes = row[0], row[2]
                title = 'Controls'
            else:
                q = f'{control_col} not in ["{negative}", "{positive}"]'
                axes = row[1], row[3]
                title = 'Experiment'
            
            this_data = wv_df.query(q)
            n_bins = 5 + this_data.shape[0] // 20
            bins_log = np.exp(np.linspace(np.log(this_data[x].min()),
                                          np.log(this_data[x].max()),
                                          num=n_bins))

            for ax, b in zip(axes, (n_bins, bins_log)):
                
                ax.hist(x, 
                        data=this_data, 
                        density=False,
                        histtype='stepfilled',
                        color=color,
                        bins=b,
                        zorder=3 - i)
            
                if x.endswith('_norm'):

                    for m in (0., 1.):

                        ax.axvline(m, 
                                color='lightgray', 
                                zorder=0)

                ax.set(xlabel='_'.join(x.split('_')[1:]) + f': {wv}',
                       ylabel='Frequency',
                       title=title)
                
            ax.set_xscale('log')

    fig.tight_layout()

    return fig, ax


def plot_replicates(data: pd.DataFrame,
                    x: str,
                    grouping: list,
                    control_col: str,
                    negative: str,
                    positive: str,
                    panel_size: float = 2.5) -> PlotTuple:

    
    read_type = '_'.join(x.split('_')[1:3]) + '_wavelength'

    if not read_type in data.columns:
        raise KeyError(f"The column {read_type} is missing from the input data.")
    
    n_read_types = data[read_type].unique().size
   
    do_log = True #n_gt_zero > .5
    n_cols = 2 if do_log else 1
    n_rows = n_read_types
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(n_cols * panel_size, 
                                      n_rows * panel_size),
                             squeeze=False)

    for row, (wv, wv_df) in zip(axes, data.groupby(read_type)):

        this_title = '_'.join(x.split('_')[1:]) + f': {wv}'

        n_gt_zero = np.mean(wv_df[x] > 0)

        sub_df = (wv_df.query('replicate < 3')
                       .assign(replicate=lambda x: 'rep_' + x['replicate'].astype(str)))

        df_wide = pd.pivot_table(sub_df,
                                 index=grouping,
                                 columns='replicate', 
                                 values=x)
    
        pearson_corr = np.corrcoef(df_wide.values,
                                   rowvar=False)[0, -1]

        for ax in row:
            
            for i, (control, color) in enumerate(zip((positive, negative, None), 
                                                    cbpal)):

                if control is not None:
                    q = f'{control_col} == "{control}"'
                else:
                    q = f'{control_col} not in ["{negative}", "{positive}"]'
                
                this_data = df_wide.query(q)

                ax.scatter('rep_1', 'rep_2', 
                            data=this_data, 
                            color=color,
                            s=.2 + (3. if control is not None else 0.),
                            zorder=3 - i)
            
            ax.plot(ax.get_xlim(), ax.get_xlim(), 
                    color='lightgray',
                    zorder=0)
            ax.set(aspect='equal', 
                   xlabel='Replicate 1',
                   ylabel='Replicate 2')
            ax.set_title(this_title + f'\nr = {pearson_corr:.2f}',
                        fontdict={'fontsize': 8.})

        if do_log:

            df_gt_zero = df_wide.query('rep_1 > 0 and rep_2 > 0')
            pearson_corr = np.corrcoef(np.log(df_gt_zero),
                                       rowvar=False)[0, -1]
            ax.set(xscale='log', yscale='log')
            ax.set_title(this_title + f'\nr = {pearson_corr:.2f} | hidden points: {100. * (1. - n_gt_zero):.1f} %', 
                          fontdict={'fontsize': 8.})

    fig.tight_layout()

    return fig, axes


def plot_scatter(data: pd.DataFrame,
                 measurement_col: str,
                 x: str, 
                 y: str, 
                 color: str = None,
                 log_color: bool = False,
                 hlines: Iterable = [],
                 vlines: Iterable = [],
                 xscale: str = 'linear',
                 yscale: str = 'linear',
                 panel_size: float = 3.) -> PlotTuple:
    
    data = data.copy()
    
    fig, ax = plt.subplots(figsize=(panel_size, 
                                    panel_size))

    x_col, y_col, c_col = (('calc_' + measurement_col + dimension) 
                           if dimension is not None else dimension
                           for dimension in (x, y, color))
    
    if log_color and c_col is not None:
        data[c_col] = np.log10(data[c_col])

    sc = ax.scatter(x_col, y_col, 
                    c=data[c_col] if c_col is not None else c_col,
                    data=data,
                    s=.5,
                    zorder=3)
    
    if c_col is not None:
        cbar = fig.colorbar(sc)
        cbar.set_label(('log10 ' if log_color else '') + 
                       c_col.split("_")[-1])

    for y_ in hlines:
        ax.axhline(y_, 
                   color='lightgrey', 
                   zorder=0)
        
    for x_ in vlines:
        ax.axvline(x_, 
                   color='lightgrey', 
                   zorder=0)
    
    ax.set(xscale=xscale, 
           yscale=yscale,
           xlabel=x.split("_")[-1], 
           ylabel=y.split("_")[-1], 
           title=measurement_col)

    return fig, ax

