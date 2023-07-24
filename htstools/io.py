"""Utilities for reading files from platereaders and writing columnar data."""

from typing import BinaryIO, Dict, IO, List, TextIO, Tuple, Union
from collections.abc import Iterable
from collections import defaultdict, namedtuple
import csv
from datetime import datetime
from functools import partial
from io import StringIO, TextIOBase
import os

from openpyxl import load_workbook
import pandas as pd

from .utils import row_col_to_well

_FORMAT: Dict[str, str] = {'.txt' : '\t', 
                           '.tsv' : '\t', 
                           '.csv' : ',', 
                           '.xlsx': 'xlsx'}

BiotekData = namedtuple('BiotekData',
                        ('Meta', 'Procedure_Details', 
                         'Layout', 'Results'),
                         defaults=[None for _ in range(4)])

def _sniff(file: Union[str, IO]) -> str:

    _, ext = os.path.splitext(file.name if isinstance(file, TextIOBase) 
                              else file)

    try:
        return _FORMAT[ext.casefold()]
    except KeyError:
        return '\t'


def _get_sheet_values(sheet) -> List[str]:

    s = []

    for row in sheet: 

        values = [r.value or '' for r in row]
        values = [r.isoformat() if isinstance(r, datetime) 
                  else r 
                  for r in values]

        s.append([r.value or '' for r in row])

    return s


def _read_xlsx(file: str) -> Dict[str, List[str]]:

    basename = os.path.basename(file)

    return {f'xlsx:{basename}:{sheet.title}': _get_sheet_values(sheet) 
            for sheet in 
            load_workbook(file, data_only=True, read_only=True)}


def _read_delim(file: Union[TextIO, str], 
                delimiter: str = None) -> dict:
    
    delimiter = delimiter or _sniff(file)

    if isinstance(file, str):

        file = open(file, 'r')

    basename = os.path.basename(file.name)

    return {f'delim:{basename}:': csv.reader(file, delimiter=delimiter)}


def _biotek_extract(x: Iterable) -> BiotekData:

    data = BiotekData(**{section: defaultdict(list) 
                         for section in BiotekData._fields})

    section, subsection = 'Meta', ''

    for row in x: 
        
        # skip empty rows
        try:
            row0 = row[0]  
        except IndexError:
            continue
        else:
            row0esc = row0.replace(' ', '_')  # to escape spaces
        
        all_none = all(len(str(item)) == 0 for item in row)

        # First column has section (sub)headings. If there is 
        # section heading in the first column, make it the 
        # current heading.
        if row0esc in data._fields:

            section, subsection = row0esc, 'main'
        
        # Otherwise, if the first column of the row has data, 
        # set this as the subsection and append the rest of
        # the row as data
        elif len(row0) > 0:

            subsection = row0.rstrip(':')
            
            if subsection not in getattr(data, section):

                subsection_n = 1 

            else:
                # handles repeated subsection names without overwriting
                subsection_n += 1  
                subsection = f'{subsection.split("_")[0]}_{subsection_n}'
                
            getattr(data, section)[subsection].append(row[1:])

        # Otherwise, append column 2 onwards as data
        elif not all_none:
            getattr(data, section)[subsection].append(row[1:])

    return data


def _biotek_common(df: pd.DataFrame,
                   data: BiotekData,
                   filename: str) -> pd.DataFrame:
    
    df = df.assign(plate_id=data.Meta['Plate Number'][0][0],
                   data_filename=filename.split(':')[1],
                   data_sheet=filename.split(':')[2])

    for heading in ('Meta', 'Procedure_Details'):

        for subsection, values in getattr(data, heading).items():

            h = 'meta_' + subsection.casefold().strip().replace(' ', '_')
            df[h] = ';'.join(map(lambda x: str(x[0]).strip(), values))

    return df

     
def _parse_read_name(read_name: str, 
                     abs_count: int = 0, 
                     fluor_count: int = 0) -> Tuple[str, str, int, int]:
    
    wavelengths = read_name.split(':')[-1].split(',')

    if len(wavelengths) == 1:

        abs_count += 1
        wv = f'{wavelengths[0]}nm'
        read_type = f'abs_ch{abs_count}'

    elif len(wavelengths) == 2:

        fluor_count += 1
        wv = f'ex:{wavelengths[0]}nm;em:{wavelengths[1]}nm'
        read_type = f'fluor_ch{fluor_count}'

    return read_type, wv, abs_count, fluor_count


def _biotek_plate(data: BiotekData,
                  filename: str,
                  measurement_prefix: str = 'measured_') -> Tuple[pd.DataFrame, Dict[str, set]]:
    
    df, read_types = defaultdict(list), defaultdict(set)

    row_ids = [subsection for subsection in data.Results if subsection != 'main']
    columns = data.Results['main'][0]
    n_cols = len(columns)

    for subsection, values in data.Results.items():

        if subsection in row_ids:

            these_rows = ([subsection] * n_cols)

            df['row_id'] += these_rows
            df['column_id'] += columns
            df['well_id'] += row_col_to_well(these_rows, columns).tolist()

            abs_count, fluor_count = 0, 0

            for value in values:

                (read_type, 
                 wv, 
                 abs_count, 
                 fluor_count) = _parse_read_name(value[-1],  # very last col gives wavelengths
                                                 abs_count,
                                                 fluor_count)

                read_types[read_type].add(wv)

                df[measurement_prefix + read_type] += value[:-1]
                df[read_type + '_wavelength'] += ([wv] * n_cols)

    df = (pd.DataFrame(df))
    df = _biotek_common(df, data, filename)

    return df, read_types


def _biotek_row(data: BiotekData,
                filename: str,
                measurement_prefix: str = 'measured_') -> Tuple[pd.DataFrame, Dict[str, set]]:
    
    read_types = defaultdict(set)

    # Row-wise data has a subheading Actual Temperature before 
    # the actual data table, hence this awkward naming
    data_str = '\n'.join(','.join(str(item) for item in row) 
                         for row in data.Results['Actual Temperature'][1:])
    
    df = (pd.read_csv(StringIO(data_str))
            .rename(columns={'Well': 'well_id',
                             'Well ID': 'well_name'})
            .assign(row_id=lambda x: x['well_id'].str.slice(stop=1),
                    column_id=lambda x: x['well_id'].str.slice(start=1).astype(int)))

    abs_count, fluor_count = 0, 0
    cols_to_drop = [col for col in df if col.startswith('Unnamed')]
    
    for header in df:

        if not header.startswith('Unnamed') and ':' in header:  # this is a data column

            (read_type, wv, 
             abs_count, fluor_count) = _parse_read_name(header,
                                                        abs_count,
                                                        fluor_count)

            read_types[read_type].add(wv)
            df[measurement_prefix + read_type] = df[header].copy()
            df[read_type + '_wavelength'] = wv
            cols_to_drop.append(header)

    df = df.drop(columns=cols_to_drop)
    df = _biotek_common(df, data, filename)

    return df, read_types


def _from_platereader(file: Union[IO, str],
                      shape: str,
                      vendor: str,
                      delimiter: str = None,
                      measurement_prefix: str = 'measured_') -> dict:

    delimiter = delimiter or _sniff(file)
    filename = file.name if isinstance(file, IO) else file

    if delimiter == 'xlsx':

        if isinstance(file, str):

            data_handle = _read_xlsx(file)
        
        elif isinstance(file, TextIOBase):

            data_handle = _read_xlsx(file.name)

        else:

            raise NotImplementedError('Excel "file" argument type '
                                      f'is not supported: {type(file)}')

    elif delimiter in _FORMAT.values():

        data_handle = _read_delim(file, delimiter=delimiter)

    else:

        raise NotImplementedError(f'File format "{format}" not supported.')
    
    if vendor == 'Biotek':

        extracted_data = {n: _biotek_extract(d) 
                          for n, d in data_handle.items()}

        if shape == 'row':

            return {n: _biotek_row(d, 
                                   filename=n, 
                                   measurement_prefix=measurement_prefix) 
                    for n, d in extracted_data.items()}
        
        elif shape == 'plate':

            return {n: _biotek_plate(d, 
                                     filename=n, 
                                     measurement_prefix=measurement_prefix) 
                    for n, d in extracted_data.items()}
        
    raise NotImplementedError( 'Cannot read platereader file type: '
                              f'{filename=}\n\t{vendor=}, {shape=}')


def from_platereader(file: Union[IO, str, List[Union[IO, str]]],
                     shape: str,
                     vendor: str,
                     delimiter: str = None,
                     measurement_prefix: str = 'measured_') -> pd.DataFrame:

    """Convert raw platereader files to columnar format.

    Initially only supports Biotek platereaders.

    Loads files exported from platereader software according to the parameters,
    and does the necessary parsing to extract metadata and measured values
    into a Pandas DataFrame, with one row per well and each column corresponding
    to a variable such as a measurement or metadata. 
    
    Measurement columns can be identified by the `measurement_prefix` value, 
    and wavelengths are annotated in columns starting with "fluor" or "abs" 
    and ending with "_wavelength".

    Parameters
    ----------
    file : str, file-like, or list
        File to parse. Must be CSV, TSV, or XLSX format.
    shape : str
        "plate" or "row", idicating whether the data are in a plate format or 
        row-wise table format.
    vendor : str
        Platereader manufacturer. Currently only "Biotek" is implemented.
    delimiter : str
        Override inference of file format with this delimiter. Must be either
        ",", "\\t", or "xlsx" (to enforce XLSX parsing).
    measurement_prefix : str
        The prefix to add to columns containing raw measured variables.

    Returns
    -------
    pandas.DataFrame
        Parsed data.

    Raises
    ------
    NotImplementedError
        Where the indicated platereader export format is not yet supported.
    
    """
    
    if not isinstance(file, list):

        file = [file]
    
    parsed_files = map(partial(_from_platereader, 
                               shape=shape, 
                               vendor=vendor, 
                               delimiter=delimiter,
                               measurement_prefix=measurement_prefix),
                       file)
    
    return pd.concat((df for parsed_file in parsed_files 
                      for _, (df, _) in parsed_file.items()),
                     axis=0)