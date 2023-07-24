"""Command-line interface for hts-tools."""

from typing import Dict, IO, List, TextIO, Tuple, Union
from collections.abc import Callable
import argparse
from functools import partial, reduce
import os
import sys

## too slow to load just for type hints
# from matplotlib.figure import Figure
# from pandas import DataFrame

from .io import from_platereader, _sniff
from .tables import join, pivot_plate, replicate_table
from .normalize import normalize
from .plot import (plot_dose_response,
                   plot_mean_sd, plot_zprime,
                   plot_replicates, plot_heatmap, 
                   plot_histogram, plot_scatter, PlotTuple)
from .qc import z_prime_factor, ssmd
from .summarize import summarize
from .utils import _print_err, pprint_dict

_FORMAT2DELIM = {'tsv': '\t', 'csv': ',', 'xlsx': 'xlsx'}

def _format2delim(format: str = None, 
                  allow_excel: bool = True) -> Union[None, str]:

    try:
        format = _FORMAT2DELIM[format.casefold()]
    except AttributeError:
        return None
    else:
        if allow_excel or format != 'xlsx':
            return format
        else:
            return None
        

def _load_table(file: Union[str, IO],
                format: str = None,
                sheet_name: Union[str, int, None, list] = 0):# -> DataFrame:
    
    import pandas as pd
    
    delimiter = (_format2delim(format) or 
                 _sniff(file)) 

    if delimiter != 'xlsx':
        return pd.read_csv(file, sep=delimiter,
                        encoding='unicode_escape')
    else:
        
        return pd.read_excel(file.name, engine='openpyxl',
                             sheet_name=sheet_name)
    

def _save_plot(fig, #: Figure, ## too slow to load just for type hints
               prefix: str, 
               column: str,
               suffix: str,
               format: str = 'pdf') -> str:
    
    filename = f'{prefix}_{column}_{suffix}.{format}'
    fig.savefig(filename, 
                dpi=300 if format.lower() == 'png' else 'figure',
                bbox_inches='tight')
    
    return filename

    
def _subset_cols(data, #: DataFrame, ## too slow to load just for type hints
                 starts: str = '', 
                 ends: str = '',
                 logic: str = 'OR') -> List[str]:
    
    if logic.upper() == 'OR':
        return [c for c in data.columns 
                if c.endswith(ends) or c.startswith(starts)]
    elif logic.upper() == 'AND':
        return [c for c in data.columns 
                if c.endswith(ends) and c.startswith(starts)]
    else:
        raise NotImplementedError(f"{logic=} not implemented")

def _table_out(df, #: DataFrame, ## too slow to load just for type hints
               output: Union[TextIO, str],
               format: str = None) -> None:
    
    delimiter = _format2delim(format, 
                              allow_excel=False) or _sniff(output) 
    
    if delimiter == 'xlsx':

        delimiter = '\t'
    
    try:
        df.to_csv(output,
                  sep=delimiter,
                  index=False)
    except BrokenPipeError:
        pass

    return None
    

def _parse(args: argparse.Namespace) -> None:

    pprint_dict(vars(args), 
                'Parsing platereader file with the following parameters')

    df = from_platereader(args.input,
                          shape=args.data_shape,
                          vendor=args.vendor,
                          delimiter=args.format,
                          measurement_prefix=args.measurement_prefix)
    
    _table_out(df, args.output, args.format)

    return None


def _join(args: argparse.Namespace) -> None:

    pprint_dict(vars(args), 
                'Joining CSV or TSV files with the following parameters')
    
    left = _load_table(args.input, args.format)
    right = _load_table(args.right, args.format_right,
                        sheet_name=None)

    shared_cols, joined_data = join(left, right,
                                    how=args.join_type)
    
    if isinstance(shared_cols[0], tuple):

        shared_cols = ('-'.join(cols) for cols in shared_cols)

    if isinstance(right, dict):

        right_shape = '; '.join(str(r.shape[0]) for r in right.values())

    else:

        right_shape = right.shape[0]

    _print_err(f"""Processed {args.input.name} <-> {args.right.name}
                   Overlapping columns: {', '.join(shared_cols)}
                   Number of rows:
                        at start: {left.shape[0]} <-> {right_shape}
                        at end: {joined_data.shape[0]}""")
    
    _table_out(joined_data, args.output, args.format)

    return None


def _col_renamer(data, #: DataFrame,
                 mcol: str,
                 rep: Tuple[str, str]) -> Dict[str, str]:

    return {col: col.replace(*rep) 
            for col in data.columns 
            if col.startswith(mcol) and col != mcol}


def _normalize(args: argparse.Namespace) -> None:

    pprint_dict(vars(args), 
                'Normalizing data with the following parameters')
    
    data = _load_table(args.input, args.format)

    measurement_cols = _subset_cols(data, 
                                    starts=args.measurement_prefix,
                                    logic='AND')

    for measurement_col in measurement_cols:

        # print(measurement_col)

        data = normalize(data, 
                         measurement_col=measurement_col,
                         control_col=args.control,
                         pos=args.positive,
                         neg=args.negative, 
                         group=args.grouping)
        data = data.rename(columns=_col_renamer(data, 
                                                measurement_col, 
                                                (args.measurement_prefix, 'calc_')))
    
    _print_err(f"""Processed {args.input.name}
                        Measurement columns: {', '.join(measurement_cols)}
                        Number of data points: {data.shape[0]}""")
    
    _table_out(data, args.output, args.format)

    return None


def _pivot(args: argparse.Namespace) -> None:

    import pandas as pd

    pprint_dict(vars(args), 
                'Pivoting the files with the following parameters')

    df = map(partial(_load_table, 
                     format=args.format, 
                     sheet_name=None), 
             args.input)

    try:
        df = map(lambda x: {key: val.set_index('Unnamed: 0') 
                            for key, val in x.items()}, 
                df)
    except AttributeError:
        df = map(lambda x: x.set_index('Unnamed: 0'), df)
        
    df = map(partial(pivot_plate, value_name=args.name), 
             df)
    filenames = tuple(os.path.basename(f.name) for f in args.input)

    df = pd.concat((d.assign(filename=filename)
                    for d, filename in zip(df, filenames)), 
                   axis=0)
    plate_info = (df.groupby(['filename', 'plate_id'], 
                             as_index=False)[['well_id']]
                    .count()
                    .rename(columns={'well_id': 'format'}))

    _print_err(f"""Parsed plate from {', '.join(filenames)}
                   Source plates:""".lstrip())
    _table_out(plate_info, sys.stderr, 'tsv')

    renamer = {col: f'{args.prefix}_{col}' 
               for col in df if col in ('filename', 'plate_id')}

    _table_out(df.rename(columns=renamer), args.output, args.format)

    return None


def _qc(args: argparse.Namespace) -> None:

    import pandas as pd

    pprint_dict(vars(args), 
                'Performing QC with the following parameters')

    data = _load_table(args.input, args.format)
    measurement_cols = _subset_cols(data, 
                                    starts=args.measurement_prefix,
                                    ends='norm')

    z_data = []

    for measurement_col in measurement_cols:

        for f in (z_prime_factor, ssmd):

            z = f(data, 
                  measurement_col=measurement_col,
                  control_col=args.control,
                  pos=args.positive,
                  neg=args.negative, 
                  group=args.grouping)
            z = z.rename(columns=_col_renamer(z, measurement_col, 
                                              (args.measurement_prefix, 'calc_')))
            z_data.append(z)

    z_data = reduce(partial(pd.merge, how='outer'), 
                    z_data)
    
    _print_err(f"""Processed {args.input.name}
                    Measurement columns: {", ".join(measurement_cols)}
                    Number of data points: {z_data.shape[0]}""")
    
    _table_out(z_data, args.output, args.format)

    if args.plot is not None:

        all_cols_to_plot = _subset_cols(z_data, 
                                        starts='calc_',
                                        ends='zprime',
                                        logic='AND')
        col_groups = tuple(set(['_'.join(c.split('_')[:-1]) 
                                for c in all_cols_to_plot]))

        grouping = [c for c in z_data.columns 
                    if not c.startswith('calc_')]

        for c in col_groups:

            c_str = c.replace("calc_", "")

            fig, _ = plot_mean_sd(z_data,
                                  x=grouping,
                                  y=c_str, 
                                  panel_size=5.)
            _save_plot(fig, args.plot, c_str, 'mean_sd',
                       format=args.plot_format)
            
            fig, _ = plot_zprime(z_data,
                                 x=grouping,
                                 y=c_str, 
                                 panel_size=5.)
            _save_plot(fig, args.plot, c_str, 'zprime',
                       format=args.plot_format)

    return None


def _summarize(args: argparse.Namespace) -> None:

    import pandas as pd

    pprint_dict(vars(args), 
                'Summarizing data with the following parameters')
    
    data = _load_table(args.input, args.format)
    measurement_cols = _subset_cols(data, 
                                    starts=args.measurement_prefix,
                                    ends='norm')
    
    summaries = []

    for measurement_col in measurement_cols:

        z = summarize(data, 
                      measurement_col=measurement_col,
                      control_col=args.control,
                      neg=args.negative, 
                      group=args.grouping)
        z = z.rename(columns=_col_renamer(z, measurement_col, 
                                          (args.measurement_prefix, 'calc_')))
        summaries.append(z)

    summaries = reduce(partial(pd.merge, how='outer'), 
                       summaries)
    
    _print_err(f"""Processed {args.input.name}
                    Measurement columns: {", ".join(measurement_cols)}
                    Number of data points: {summaries.shape[0]}""")
    
    _table_out(summaries, args.output, args.format)

    if args.plot is not None:

        measurements = _subset_cols(summaries, 
                                    ends='_wavelength',
                                    logic='AND')
        col_groups = tuple(set(['_'.join(c.split('_')[:-1]) for c in measurements]) |
                       set(['_'.join(c.split('_')[:-1]) + '_norm' for c in measurements]))

        for c in col_groups:

            c_str = c.replace("calc_", "")

            fig, _ = plot_scatter(summaries,
                                  measurement_col=c_str,
                                  x='_log10fc',
                                  y='_ssmd',
                                  color='_t_p',
                                  log_color=True,
                                  vlines=[0.],
                                  hlines=[0.],
                                  panel_size=2.5)
            _save_plot(fig, args.plot, c_str, 'flashlight',
                       format=args.plot_format)
            
            fig, _ = plot_scatter(summaries,
                                  measurement_col=c_str,
                                  x='_log10fc',
                                  y='_t_p',
                                  vlines=[0.],
                                  yscale='log',
                                  panel_size=2.5)
            _save_plot(fig, args.plot, c_str, 'volcano',
                       format=args.plot_format)

    return None


def _plot_columns(args: argparse.Namespace,
                  name: str,
                  plotting_function: Callable,
                  data_function: Callable = lambda x: x) -> None:

    pprint_dict(vars(args), 
                f'Plotting {name} with the following parameters')

    data = data_function(_load_table(args.input, args.format))

    all_cols_to_plot = _subset_cols(data, 
                                    starts=args.measurement_prefix,
                                    ends='_norm')

    filenames = []

    for column in all_cols_to_plot:

        fig, _ = plotting_function(data, column)
        
        filename = _save_plot(fig, args.output, column, name,
                              format=args.plot_format)
        filenames.append(filename)

    _print_err(f"""Processed {args.input.name}
                    Plotted columns: {', '.join(all_cols_to_plot)}
                    Number of data points: {data.shape[0]}
                    Filenames:\n\t""" + 
                    '\n\t'.join(filenames))

    return None


def _plot_dose(args: argparse.Namespace) -> None:

    pprint_dict(vars(args), 
                'Plotting dose-response with the following parameters')

    data = _load_table(args.input, args.format)

    all_cols_to_plot = _subset_cols(data, 
                                    starts=args.measurement_prefix,
                                    ends='_norm')

    filenames = []

    for column in all_cols_to_plot:

        this_file_prefix = f'{args.output}_{column}_'

        these_filenames = plot_dose_response(data, 
                                             file_prefix=this_file_prefix,
                                             x=args.x, 
                                             y=column, 
                                             color=args.color, 
                                             color_control=args.control,
                                             facet=args.facet, 
                                             files=args.files,
                                             hlines=[0., 1.] if column.endswith('_norm') else [0.],
                                             format=args.plot_format)
        
        filenames += these_filenames

    _print_err(f"""Processed {args.input.name}
                    Plotted columns: {', '.join(all_cols_to_plot)}
                    Number of data points: {data.shape[0]}
                    Filenames:\n\t""" + 
                    '\n\t'.join(filenames))
    
    return None


def _plot_rep(args: argparse.Namespace) -> None:
    
    _plot_columns(args, 'replicates',
                  partial(plot_replicates,
                          grouping=args.grouping,
                          control_col=args.control,
                          negative=args.negative,
                          positive=args.positive,
                          panel_size=2.5),
                  partial(replicate_table, group=args.grouping))

    return None


def _plot_hm(args: argparse.Namespace) -> None:

    _plot_columns(args, 'heatmap',
                  lambda x, y: plot_heatmap(x, 
                                            x=args.grouping,
                                            y=y, 
                                            panel_size=2.5))

    return None


def _plot_hist(args: argparse.Namespace) -> None:

    _plot_columns(args, 'histogram',
                  partial(plot_histogram,
                          control_col=args.control,
                          negative=args.negative,
                          positive=args.positive,
                          panel_size=2.5))

    return None


def main() -> None:

    parser = argparse.ArgumentParser(description='''
    Tools for analysing medium- and high-throughout platereader data.
    ''')

    subcommands = parser.add_subparsers(title='Sub-commands', 
                                        dest='subcommand',
                                        help='Use these commands to specify the tool you want to use.')
    
    parse = subcommands.add_parser('parse', 
                                   help='Convert raw platereader files to columnar '
                                        'format suitable for analysis.')
    parse.set_defaults(func=_parse)
    join = subcommands.add_parser('join', 
                                   help='Join two CSV or TSV files on common columns. '
                                        'Also known as a merge.')
    join.set_defaults(func=_join)
    pivot = subcommands.add_parser('pivot', 
                                   help='Convert an Excel or TSV or CSV file containing '
                                        'visual row x column layout into a columnar format '
                                        'suitable for analysis.')
    pivot.set_defaults(func=_pivot)
    normalize = subcommands.add_parser('normalize', 
                                      help='Normalize data according to grouped '
                                           'positive and negative controls.')
    normalize.set_defaults(func=_normalize)
    qc = subcommands.add_parser('qc', 
                                    help='Run quality control checks based on '
                                         'positive and negative controls.')
    qc.set_defaults(func=_qc)

    plot_dose = subcommands.add_parser('plot-dose', 
                                     help='Generate dose-response plots with arbitrary '
                                          'x, y, color, facet, and file splits.')
    plot_dose.set_defaults(func=_plot_dose)

    plot_hm = subcommands.add_parser('plot-hm', 
                                     help='Generate plate heatmaps to visualize '
                                          'within-plate variability.')
    plot_hm.set_defaults(func=_plot_hm)

    plot_rep = subcommands.add_parser('plot-rep', 
                                     help='Generate replicate correlation plots.')
    plot_rep.set_defaults(func=_plot_rep)

    plot_hist = subcommands.add_parser('plot-hist', 
                                       help='Generate histograms of data.')
    plot_hist.set_defaults(func=_plot_hist)

    summarize = subcommands.add_parser('summarize', 
                                       help='Generate summary statistics.')
    summarize.set_defaults(func=_summarize)

    parse.add_argument('input', 
                       type=argparse.FileType('r'), 
                       nargs='*',
                       help='File that was exported from platereader. Required.')
    parse.add_argument('--data-shape', '-s', 
                       type=str,
                       choices=['plate', 'row'],
                       required=True,
                       help='Shape of the platereader export. Either '
                            '"row" for row-wise table, or "plate" for '
                            'plate shaped table. Required.')
    parse.add_argument('--vendor', '-r', 
                       type=str,
                       default='Biotek',
                       choices=['Biotek'],
                       help='Platereader vendor. Default: %(default)s')
    
    join.add_argument('--right', '-r', 
                      type=argparse.FileType('r'),
                      required=True,
                      help='Right CSV file to join on. Required.')
    join.add_argument('--join-type', '-j', 
                      type=str,
                      default='left',
                      help='What type of join to execute. Default: %(default)s')
    join.add_argument('--format-right', '-g', 
                      type=str,
                      default=None,
                      help='Override file extensions for right table input. '
                           'Default: infer from file extension')
    
    pivot.add_argument('input', 
                        default=sys.stdin,
                        type=argparse.FileType('r'), 
                        nargs='*',
                        help='Excel, CSV or TSV with compound layout. Required')
    pivot.add_argument('--name', '-n', 
                        type=str,
                        required=True,
                        help='Column title to use for the well values. Required.')
    pivot.add_argument('--prefix', '-x', 
                        type=str,
                        default='',
                        help='Prefix to add to sheet_name, filename, and '
                             'well_id column names. Default: no prefix.')
    
    plot_dose.add_argument('--x', '-x', 
                           type=str, 
                           required=True,
                           help='Column to use for plot x-axis. Required')
    plot_dose.add_argument('--color', '-c', 
                           type=str, 
                           default=None,
                           help='Column to use for plot color. Optional')
    plot_dose.add_argument('--control', '-l', 
                           type=str, 
                           default=None,
                           help='Column to use for plot color. Optional')
    plot_dose.add_argument('--facet', '-p', 
                           type=str, 
                           default=None, 
                           help='Column to use for plot facet panels. Optional')
    plot_dose.add_argument('--files', '-z', 
                           type=str, 
                           default=None,
                           help='Column to use to split plots into separate files. Optional')

    for p in (join, normalize, summarize, qc, 
              plot_dose, plot_rep, plot_hm, plot_hist):
   
        p.add_argument('input', 
                       type=argparse.FileType('r'), 
                       default=sys.stdin,
                       nargs='?',
                       help='Columnar input file in CSV or TSV format. Default STDIN')
        
    for p in (parse, join, pivot, normalize, summarize, qc):

        p.add_argument('--output', '-o', 
                       type=argparse.FileType('w'),
                       default=sys.stdout,
                       help='Output file. Default STDOUT')

    for p in (parse, join, pivot, normalize, summarize, qc, 
                plot_dose, plot_hist, plot_rep, plot_hm):                   
        p.add_argument('--format', '-f', 
                       type=str,
                       default=None,
                       choices=list(_FORMAT2DELIM) + list(map(str.upper, _FORMAT2DELIM)),
                       help='Override file extensions for input and output. '
                            'Default: infer from file extension.')
        
    for p in (plot_dose, plot_hist, plot_rep, plot_hm):

        p.add_argument('--output', '-o', 
                       type=str,
                       required=True,
                       help='Save plot as PDF with this prefix. Required.')
    
    for p in (parse, join, normalize, summarize, qc, 
              plot_dose, plot_rep, plot_hm, plot_hist):

        p.add_argument('--measurement-prefix', '-m', 
                       type=str,
                       default='measured_',
                       help='Prefix of column headings containing '
                            'raw measured data. Default: %(default)s')
        
    for p in (normalize, summarize, qc, plot_rep, plot_hist):

        p.add_argument('--control', '-c', 
                       type=str,
                       required=True,
                       help='Column containing values indicating controls. Required')
        p.add_argument('--positive', '-p', 
                       type=str,
                       required=True,
                       help='Positive control value. Required')
        p.add_argument('--negative', '-n', 
                       type=str,
                       required=True,
                       help='Negative control value. Required')
        
    for p in (normalize, summarize, qc, plot_rep, plot_hm):
        p.add_argument('--grouping', '-g', 
                       type=str,
                       nargs='*',
                       required=True,
                       help='Column heading(s) indicating the data groups '
                            'within which to process measurements. Required.')

    for p in (summarize, qc):
        p.add_argument('--plot', '-t', 
                       type=str,
                       default=None,
                       help='If used, plots are generated and saved as PDF with this '
                            'filename prefix.')
        
    for p in (summarize, qc, plot_dose, plot_rep, plot_hm, plot_hist):
        p.add_argument('--plot-format',
                       type=str,
                       choices=['pdf', 'png'],
                       default='png',
                       help='File format for plots. Default: %(default)s.')

    
    args = parser.parse_args()
    args.func(args)

    return None


if __name__ == "__main__":

    main()