"""Command-line interface for hts-tools."""

__version__ = "0.0.4.post1"

from typing import Callable, Dict, Iterable, IO, List, Mapping, TextIO, Tuple, Union
from argparse import FileType, Namespace
from functools import partial, reduce
from io import TextIOWrapper
import os
from re import compile, search
import sys

from carabiner import (
    cast,
    clicommand, 
    CLIApp, 
    CLICommand,
    CLIOption,
    print_err,
)
from carabiner.mpl import figsaver
from carabiner.pd import get_formats, read_table, write_stream
from matplotlib.figure import Figure
from pandas import concat, merge, DataFrame

from .io import from_platereader
from .tables import join, pivot_plate, replicate_table
from .normalize import normalize
from .plot import (
    plot_dose_response,
    plot_mean_sd, 
    plot_zprime,
    plot_replicates, 
    plot_heatmap, 
    plot_histogram, 
    plot_scatter, 
    PlotTuple,
)
from .qc import z_prime_factor, ssmd
from .summarize import summarize
    
def _save_plot(
    fig: Figure,
    prefix: str, 
    column: str,
    suffix: str,
    format: str = "pdf",
) -> str:
    format = format.casefold()
    filename = f"{prefix}_{column}_{suffix}.{format}"
    figsaver(
        prefix=f"{prefix}_{column}_", 
        format=format,
    )(fig, name=suffix)
    return filename


def _subset_cols(
    data: DataFrame, 
    starts: str = '', 
    ends: str = '',
    logic: str = 'OR'
) -> List[str]:
    starts, ends = compile('^' + starts), compile(ends + '$')
    if logic.upper() == 'OR':
        return [c for c in data.columns 
                if search(ends, c) or search(starts, c)]
    elif logic.upper() == 'AND':
        return [c for c in data.columns 
                if search(ends, c) and search(starts, c)]
    else:
        raise NotImplementedError(f"{logic=} not implemented")


@clicommand(message='Parsing platereader file with the following parameters')
def _parse(args: Namespace) -> None:

    df = from_platereader(
        args.input,
        shape=args.data_shape,
        vendor=args.vendor,
        delimiter=args.format,
        measurement_prefix=args.measurement_prefix,
    )
    write_stream(df, args.output, args.format)
    return None


@clicommand(message='Joining CSV or TSV files with the following parameters')
def _join(args: Namespace) -> None:
    
    left = read_table(args.input, args.format)
    right = read_table(args.right, args.format_right)
    shared_cols, joined_data = join(
        left, 
        right,
        how=args.join_type,
    )
    
    if isinstance(shared_cols[0], tuple):
        shared_cols = ('-'.join(cols) for cols in shared_cols)

    if isinstance(right, Mapping):
        right_shape = '; '.join(str(r.shape[0]) for r in right.values())
    else:
        right_shape = right.shape[0]

    print_err(f"""Processed {args.input.name} <-> {args.right.name}
                   Overlapping columns: {', '.join(shared_cols)}
                   Number of rows:
                        at start: {left.shape[0]} <-> {right_shape}
                        at end: {joined_data.shape[0]}""")
    
    write_stream(joined_data, args.output, args.format)

    return None


def _col_renamer(
    data: DataFrame,
    mcol: str,
    rep: Iterable[str],
) -> Dict[str, str]:
    return {
        col: col.replace(*rep) for col in data.columns 
        if col.startswith(mcol) and col != mcol
    }


@clicommand(message='Normalizing data with the following parameters')
def _normalize(args: Namespace) -> None:
    
    data = read_table(args.input, args.format)
    measurement_cols = _subset_cols(
        data, 
        starts=args.measurement_prefix,
        logic='AND',
    )

    for measurement_col in measurement_cols:

        data = normalize(data, 
                         measurement_col=measurement_col,
                         control_col=args.control,
                         method=args.method,
                         pos=args.positive,
                         neg=args.negative, 
                         group=args.grouping)
        data = data.rename(columns=_col_renamer(data, 
                                                measurement_col, 
                                                (args.measurement_prefix, 'calc_')))
    
    print_err(f"""Processed {args.input.name}
                        Measurement columns: {', '.join(measurement_cols)}
                        Number of data points: {data.shape[0]}""")
    
    write_stream(data, args.output, args.format)

    return None

@clicommand(message='Pivoting the files with the following parameters')
def _pivot(args: Namespace) -> None:
    
    loader = partial(
        read_table, 
        format=args.format, 
        sheet_name=None,
    )
    input_files = cast(args.input, to=list)
    df = tuple(loader(file) for file in input_files)
    filenames = tuple(os.path.basename(file.name) for file in input_files)

    try:  # assume XLSX with sheets
        df = tuple({
            sheet_name: sheet.set_index('Unnamed: 0') for sheet_name, sheet in data.items()
            } for data in df)
    except AttributeError:  # this happens when not XLSX - just a DataFrame per file
        df = tuple(sheet.set_index('Unnamed: 0') for sheet in df)

    pivoter = partial(
        pivot_plate, 
        value_name=args.name,
    )
    df = tuple(pivoter(data) for data in df)
    df = concat((
        data.assign(filename=filename) for data, filename in zip(df, filenames)
        ), axis=0)
    plate_info = (
        df
        .groupby(
            ['filename', 'plate_id'], 
            as_index=False
        )[['well_id']]
        .count()
        .rename(columns={'well_id': 'format'})
    )

    print_err(
        f"""Parsed plate from {', '.join(filenames)}
                Source plates:""".lstrip()
    )
    write_stream(plate_info, sys.stderr, 'tsv')
    renamer = {
        col: f'{args.prefix}_{col}' for col in df if col in ('filename', 'plate_id')
    }
    write_stream(df.rename(columns=renamer), args.output, args.format)
    return None


@clicommand(message='Performing QC with the following parameters')
def _qc(args: Namespace) -> None:

    import pandas as pd

    data = read_table(args.input, args.format)
    measurement_cols = _subset_cols(
        data, 
        starts=args.measurement_prefix,
        ends=r'norm\..{3}',
    )

    z_data = []
    for measurement_col in measurement_cols:
        for f in (z_prime_factor, ssmd):
            z = f(data, 
                  measurement_col=measurement_col,
                  control_col=args.control,
                  pos=args.positive,
                  neg=args.negative, 
                  group=args.grouping)
            z = z.rename(
                columns=_col_renamer(
                    z, 
                    measurement_col, 
                    (args.measurement_prefix, 'calc_'),
                )
            )
            z_data.append(z)

    z_data = reduce(
        partial(merge, how='outer'), 
        z_data,
    )
    
    print_err(
        f"""Processed {args.input.name}
                Measurement columns: {", ".join(measurement_cols)}
                Number of data points: {z_data.shape[0]}"""
    )
    
    write_stream(z_data, args.output, args.format)
    if args.plot is not None:
        all_cols_to_plot = _subset_cols(
            z_data, 
            starts='calc_',
            ends='zprime',
            logic='AND',
        )
        col_groups = sorted(set([
            '_'.join(c.split('_')[:-1]) for c in all_cols_to_plot
        ]))

        grouping = [
            c for c in z_data.columns if not c.startswith('calc_')
        ]

        for c in col_groups:
            c_str = c.replace("calc_", "")
            fig, _ = plot_mean_sd(
                z_data,
                x=grouping,
                y=c_str,
            )
            _save_plot(fig, args.plot, c_str, 'mean_sd',
                       format=args.plot_format)
            
            fig, _ = plot_zprime(
                z_data,
                x=grouping,
                y=c_str,
            )
            _save_plot(fig, args.plot, c_str, 'zprime',
                       format=args.plot_format)
    return None


@clicommand(message='Summarizing data with the following parameters')
def _summarize(args: Namespace) -> None:
    
    data = read_table(args.input, args.format)
    measurement_cols = _subset_cols(
        data, 
        starts=args.measurement_prefix,
        ends=r'norm\..{3}',
    )
    
    summaries = []

    for measurement_col in measurement_cols:

        z = summarize(
            data, 
            measurement_col=measurement_col,
            control_col=args.control,
            neg=args.negative, 
            group=args.grouping,
        )
        z = z.rename(
            columns=_col_renamer(
                z, 
                measurement_col, 
                (args.measurement_prefix, 'calc_'),
            )
        )
        summaries.append(z)

    summaries = reduce(
        partial(merge, how='outer'), 
        summaries,
    )
    
    print_err(
        f"""Processed {args.input.name}
                Measurement columns: {", ".join(measurement_cols)}
                Number of data points: {summaries.shape[0]}"""
    )
    
    write_stream(summaries, args.output, args.format)
    if args.plot is not None:
        measurements = _subset_cols(
            summaries, 
            ends='_wavelength',
            logic='AND',
        )
        channels = set([
            '_'.join(col.split('_')[:-1]) for col in measurements
        ])
        normalized = set(['_'.join(col.split('_')[:-1]) 
                          for col in summaries 
                          if col.__contains__('_norm.') and 
                          any(col.__contains__(channel) for channel in channels)])
        col_groups = tuple(channels | normalized)

        for c in col_groups:
            c_str = c.replace("calc_", "")
            fig, _ = plot_scatter(
                summaries,
                measurement_col=c_str,
                x='_log10fc',
                y='_ssmd',
                color='_t.p',
                log_color=True,
                vlines=[0.],
                hlines=[0.],
                aspect_ratio=1.15,
            )
            _save_plot(
                fig, 
                args.plot, 
                c_str, 
                'flashlight',
                format=args.plot_format,
            )
            fig, _ = plot_scatter(
                summaries,
                measurement_col=c_str,
                x='_log10fc',
                y='_t.p',
                vlines=[0.],
                y_log=True,
            )
            _save_plot(
                fig, 
                args.plot, 
                c_str, 
                'volcano',
                format=args.plot_format,
            )
    return None

def _plot_columns(
    args: Namespace,
    name: str,
    plotting_function: Callable,
    data_function: Callable = lambda x: x
) -> None:

    data = data_function(read_table(args.input, args.format))
    all_cols_to_plot = _subset_cols(
        data, 
        starts=args.measurement_prefix,
        ends=r'_norm\..{3}',
    )

    filenames = []
    for column in all_cols_to_plot:
        fig, _ = plotting_function(data, column)
        filename = _save_plot(fig, args.output, column, name,
                              format=args.plot_format)
        filenames.append(filename)

    print_err(
        f"""Processed {args.input.name}
                Plotted columns: {', '.join(all_cols_to_plot)}
                Number of data points: {data.shape[0]}
                Filenames:\n\t""" + 
                '\n\t'.join(map(str, filenames))
    )

    return None


@clicommand(message='Plotting dose-response with the following parameters')
def _plot_dose(args: Namespace) -> None:

    data = read_table(args.input, args.format)
    all_cols_to_plot = _subset_cols(
        data, 
        starts=args.measurement_prefix,
        ends=r'_norm\..{3}'
    )

    filenames = []
    for column in all_cols_to_plot:
        this_file_prefix = f'{args.output}_{column}_'
        these_filenames = plot_dose_response(
            data, 
            file_prefix=this_file_prefix,
            x=args.x, 
            y=column, 
            color=args.color, 
            color_control=args.control,
            facet=args.panel, 
            files=args.files,
            hlines=[0., 1.] if column.__contains__('_norm.') else [0.],
            format=args.plot_format,
            x_log=args.x_log,
            sharey=args.share_y,
        )
        filenames += these_filenames

    print_err(
        f"""Processed {args.input.name}
            Plotted columns: {', '.join(all_cols_to_plot)}
            Number of data points: {data.shape[0]}
            Filenames:\n\t""" + 
            '\n\t'.join(filenames)
    )
    return None


@clicommand(message=f'Plotting replicates with the following parameters')
def _plot_rep(args: Namespace) -> None:
    
    _plot_columns(
        args, 
        'replicates',
        partial(
        plot_replicates,
            grouping=args.grouping,
            control_col=args.control,
            negative=args.negative,
            positive=args.positive,
        ),
        partial(replicate_table, group=args.grouping)
    )
    return None


@clicommand(message=f'Plotting heatmaps with the following parameters')
def _plot_hm(args: Namespace) -> None:
    _plot_columns(
        args, 
        'heatmap',
        lambda x, y: plot_heatmap(
            x, 
            x=args.grouping,
            y=y, 
        )
    )
    return None


@clicommand(message=f'Plotting histograms with the following parameters')
def _plot_hist(args: Namespace) -> None:

    _plot_columns(
        args, 
        'histogram',
        partial(
            plot_histogram,
            control_col=args.control,
            negative=args.negative,
            positive=args.positive,
        )
    )
    return None


def main() -> None:

    multi_input = CLIOption(
        'input', 
        type=FileType('r'), 
        nargs='*',
        help='Input filenames.',
    )
    std_input = CLIOption(
        'input',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Columnar input file in CSV or TSV format. Default STDIN',
    )
    output = CLIOption(
        '--output', '-o', 
        type=FileType('w'),
        default=sys.stdout,
        help='Output file. Default: STDOUT',
    )
    formatting = CLIOption(
        '--format', '-f', 
        type=str,
        default=None,
        choices=get_formats(),
        help='Override file extensions for input and output. '
             'Default: infer from file extension.',
    )
    std_io = [std_input, output, formatting]

    data_shape = CLIOption(
        '--data-shape', '-s', 
        type=str,
        choices=['plate', 'row'],
        required=True,
        help='Shape of the platereader export. Either '
            '"row" for row-wise table, or "plate" for '
            'plate shaped table.',
    )
    vendor = CLIOption(
        '--vendor', '-r', 
        type=str,
        default='Biotek',
        choices=['Biotek'],
        help='Platereader vendor.',
    )
    parse_opts = [data_shape, vendor]

    right_file = CLIOption(
        '--right', '-r', 
        type=FileType('r'),
        required=True,
        help='Right CSV file to join on.'
    )
    join_type = CLIOption(
        '--join-type', '-j', 
        type=str,
        default='inner',
        choices=["inner", "outer", "left", "right"],
        help='Type of join to perform.',
    )
    format_right = CLIOption(
        '--format-right', '-g', 
        type=str,
        default=None,
        help='Override file extensions for right table input. '
            'Default: infer from file extension',
    )
    join_opts = [right_file, join_type, format_right]

    cmpd_names = CLIOption(
        '--name', '-n', 
        type=str,
        required=True,
        help='Column title to use for the well values.',
    )
    col_prefix = CLIOption(
        '--prefix', '-x', 
        type=str,
        default='',
        help='Prefix to add to sheet_name, filename, and '
             'well_id column names. Default: no prefix.'
    )
    pivot_opts = [cmpd_names, col_prefix]

    norm_method = CLIOption(
        '--method', '-t', 
        type=str,
        default='PON',
        choices=['PON', 'NPG', 'pon', 'npg'],
        help='Normalization method.',
    )

    measurement_prefix = CLIOption(
        '--measurement-prefix', '-m', 
        type=str,
        default='measured_',
        help='Prefix of column headings containing raw measured data.',
    )
    control = CLIOption(
        '--control', '-c', 
        type=str,
        required=True,
        help='Column containing values indicating controls.',
    )
    pos_val = CLIOption(
        '--positive', '-p', 
        type=str,
        required=True,
        help='Positive control value.',
    )
    neg_val = CLIOption(
        '--negative', '-n', 
        type=str,
        required=True,
        help='Negative control value.',
    )
    ctrl_opts = [control, pos_val, neg_val]

    grouping = CLIOption(
        '--grouping', '-g', 
        type=str,
        nargs='*',
        required=True,
        help='Column heading(s) indicating the data groups '
             'within which to process measurements.',
    )
    plot_x = CLIOption(
        '--x', '-x', 
        type=str, 
        required=True,
        help='Column to use for plot x-axis.',
    )
    plot_color = CLIOption(
        '--color', '-c', 
        type=str, 
        default=None,
        help='Column to use for plot color.',
    )
    plot_control = CLIOption(
        '--control', '-l', 
        type=str, 
        default=None,
        help='Column to use for plot color.',
    )
    plot_panel = CLIOption(
        '--panel', '-p', 
        type=str, 
        default=None, 
        help='Column to use for plot facet panels.',
    )
    plot_files = CLIOption(
        '--files', '-z', 
        type=str, 
        default=None,
        help='Column to use to split plots into separate files.'
    )
    x_log = CLIOption(
        '--x-log',
        action='store_true', 
        default=False,
        help='Make x-axis a log scale.'
    )
    share_y = CLIOption(
        '--share-y',
        action='store_true', 
        default=False,
        help='Make y-axis the same range on every panel.'
    )
    plot_dose_opts = [plot_x, plot_color, plot_control, plot_panel, plot_files, x_log, share_y]

    plot_prefix = CLIOption(
        '--plot', '-t', 
        type=str,
        default=None,
        help='If used, plots are generated and saved as PDF with this '
             'filename prefix.',
    )
    plot_format = CLIOption(
        '--plot-format',
        type=str,
        choices=['pdf', 'png'],
        default='png',
        help='File format for plots.',
    )
    plot_output_name = CLIOption(
        '--output', '-o', 
        type=str,
        required=True,
        help='Save plot with this prefix.',
    )
    plot_opts = [plot_format, plot_output_name]

    parse = CLICommand(
        "parse",
        description="Convert raw platereader files to columnar format suitable for analysis.",
        options=[multi_input, formatting, output] + parse_opts + [measurement_prefix],
        main=_parse,
    )
    pivot = CLICommand(
        "pivot",
        description="""
        Convert an Excel or TSV or CSV file containing 
        visual row x column layout into a columnar format
        suitable for analysis.
        """,
        options=std_io + pivot_opts,
        main=_pivot,
    )
    join = CLICommand(
        "join",
        description="""
        Join two CSV or TSV files on common columns. 
        Also known as a merge.
        """,
        options=std_io + join_opts + [measurement_prefix],
        main=_join,
    )
    normalize = CLICommand(
        "normalize",
        description="""
        Normalize data according to grouped 
        positive and negative controls.
        """,
        options=std_io + [norm_method, measurement_prefix, grouping] + ctrl_opts,
        main=_normalize,
    )
    summarize = CLICommand(
        "summarize",
        description="""Generate summary statistics.""",
        options=std_io + [measurement_prefix, grouping] + ctrl_opts + [plot_prefix, plot_format],
        main=_summarize,
    )
    qc = CLICommand(
        "qc",
        description="""
        Run quality control checks based on 
        positive and negative controls.
        """,
        options=std_io + [measurement_prefix, grouping] + ctrl_opts + [plot_prefix, plot_format],
        main=_qc,
    )
    plot_dose = CLICommand(
        "plot-dose",
        description="""
        Generate dose-response plots with arbitrary 
        x, y, color, panel, and file splits.
        """,
        options=[std_input] + plot_dose_opts + [measurement_prefix, formatting] + plot_opts,
        main=_plot_dose,
    )
    plot_hm = CLICommand(
        "plot-hm",
        description="""
        Generate plate heatmaps to visualize 
        within-plate variability.
        """,
        options=[std_input, measurement_prefix, grouping, formatting] + plot_opts,
        main=_plot_hm,
    )
    plot_rep = CLICommand(
        "plot-rep",
        description="""Generate replicate correlation plots.""",
        options=[std_input, measurement_prefix, grouping, formatting] + ctrl_opts + plot_opts,
        main=_plot_rep,
    )
    plot_hist = CLICommand(
        "plot-hist",
        description="""Generate histograms of data.""",
        options=[std_input, measurement_prefix, formatting] + ctrl_opts + plot_opts,
        main=_plot_hist,
    )
    
    app = CLIApp(
        "hts-tools", 
        version=__version__,
        description="Tools for analysing medium- and high-throughout platereader data.",
        commands=[
            parse, 
            join, 
            pivot, 
            normalize, 
            qc, 
            plot_dose, 
            plot_hm, 
            plot_rep, 
            plot_hist, 
            summarize
        ],
    )

    app.run()

    return None


if __name__ == "__main__":
    main()