# Usage

**hts-tools**  provides command-line utlities to analyse and plot data
from platereaders, starting with the _raw_ exported data with no 
manual conversion or copy-pasting needed. The tools complete specific tasks which 
can be easily composed into analysis pipelines, because the TSV table output goes to
`stdout` by default so they can be piped from one tool to another.

To get a list of commands (tools), do

```bash
hts --help
```

And to get help for a specific command, do

```bash
hts <command> --help
```

For the Python API, [see here](#python).

## First pipeline example

This command takes several exported Excel files (matching the pattern `plate-?.xlsx`) from a Biotek platereader, 
adds annotations on experimental conditions, and normalizes the data based on positive and negative controls. 
Finally, dose-response curves are plotted. 

```bash
hts parse plate-?.xlsx --data-shape row --vendor Biotek \
    | hts join --right layout.xlsx \
    | hts normalize --control compound_name --positive RIF --negative DMSO --grouping plate_id \
    | hts plot-dose -x concentration --facets guide_name --color compound_name \
        --output plt-test
```

### Parsing platereader exports

The command `hts parse` converts Excel or CSV or TSV files exported from platereader software from a specified vendor into a 
uniform columnar table format used by all downstream hts-tools. The `data-shape` option indicates whether the export was in plate 
format or row-wise table format. It should usually be the first command in a pipeline. 

`hts parse` produces a table with at least the following columns (with example entries):

| row_id	| column_id	| well_id	 | plate_id	| data_filename	| data_sheet       | measured_abs_ch1 | abs_ch1_wavelength |
| ------- | --------- | -------- | -------- | ------------- | ---------------- | ---------------- | ------------------ |
| A       | 1         | A01      | Plate 6  | plate-6.xlsx  | Plate6 - Sheet 1 | 0.366            | 600nm              |
| A       | 2         | A02      | Plate 6  | plate-6.xlsx  | Plate6 - Sheet 1 | 0.402	          | 600nm              |

If only fluorescence was measured, the last two columns would be called `measured_fluor_ch1` and `fluor_ch1_wavelength`. If there were multiple
measurements, then they will appear as additional columns with increasing `ch` (channel) numbers, for example,  `measured_fluor_ch2`, `fluor_ch2_wavelength`.

There will also be additional columns using information scraped from the input file. These will usually have headings starting with `meta_`,
such as `meta_protocol_file_path`, `meta_date`, `meta_time`, and `meta_reader_serial_number`.

### Adding experimental conditions

The output from `hts` parse is passed to `hts join`, which combines two tables based on values in shared columns (known as a 
database join). For example, if `layout.xlsx` contains a sheet with this table:

| column_id	| plate_id | compound_name |
| --------- | -------- | ------------- |
| 2         | Plate 6  | trimethoprim  |
| 1         | Plate 6  | moxifloxacin  |

then `hts parse plate-?.xlsx --data-shape row | hts join --right layout.xlsx` will result in 

| row_id	| column_id	| well_id	 | plate_id	| data_filename	| data_sheet       | measured_abs_ch1 | abs_ch1_wavelength | compound_name |
| ------- | --------- | -------- | -------- | ------------- | ---------------- | ---------------- | ------------------ | ------------- |
| A       | 1         | A01      | Plate 6  | plate-6.xlsx  | Plate6 - Sheet 1 | 0.366            | 600nm              | moxifloxacin  |
| A       | 2         | A02      | Plate 6  | plate-6.xlsx  | Plate6 - Sheet 1 | 0.402	          | 600nm              | trimethoprim  |

Since `column_id` and `plate_id` are the only column headings in common, the entries of these columns are used to match the rows of the 
tables. So where `column_id = 2`, `compound_name = trimethoprim` will be added. 

If you join an Excel XLSX file containing multiple sheets, these will each be joined in order. In this way, you can add experimental
conditions easily by, for example, first joining the conditions that vary by plate and row (such as compound), then by column 
(such as concentration). This approach is very flexible, and you can join on any number of columns and add any new ones you like as
long as the column headings aren't repeated. 

### Normalization within batches 

The `hts normalize` command normalizes raw measured data to be between $0$ and $1$ based on posiitve and negative controls, optionally within groups 
(or batches) of measurements. In the example above, the positive and negative controls are defined as `RIF` and `DMSO`, and 
should be found in the column `compound_name` (which may have been added by `hts join`).

The positive and negative controls are averaged within each value in the `--grouping` column. In the example above, they will 
be averaged for each `plate_id`, and these will be used to normalize the measured values of that `plate_id` according to:

$$s = \frac{m - \mu_p}{\mu_n - \mu_p}$$

where $s$ is the normalized value, $m$ is the measured value, and $\mu_p$ and $\mu_n$ are the average positive and negative controls. 

`hts normalize` adds new columns for each measured column. These columns start with `calc_` and end with `_norm`, for example `calc_abs_ch1_norm`
and `calc_abs_ch2_norm`.

### Plotting dose response

`hts plot-dose` is a very flexible command which takes the columnar tables as input and plots the data in almost any breakdown using a color-blind 
palette. The required `-x` option indicates which column to use as the x-axis (usually concentration). The y-axis will be values in all the measured 
and calculated columns (`hts plot-dr` plots them all automatically in seaparte files). The other options allow splitting the plots by file, facet 
(panel) and color according to the values in columns. 

The example above, `hts plot-dose -x concentration --facets guide_name --color compound_name`, will produce plots like this:

<img src="docs/source/_static/plt-test_calc_abs_ch1_norm_.png" alt="" width="400">

The panels each value with the same `guide_name` is in a `facet` (panel), and the lines are colored by `compound_name`.

## Second pipeline example

Here is another example showing the sequential use of `hts join` to join two tables of experimental data, and two other commands: `hts pivot` and 
`hts summarize`.

```bash
hts pivot compounds.xlsx \
      --name compound_name

hts parse plate-*.txt --data-shape plate \
    | hts join --right sample-sheet.csv \
    | hts join --right pivoted-compounds.tsv \
    | hts normalize --control compound_name --positive RIF --negative DMSO --grouping strain_name plate_id \
    | hts summarize --control compound_name --positive RIF --negative DMSO --grouping strain_name compound_name --plot summary \
    > summary.tsv
```

These commands are explained below.

### Converting plate shaped data to columns

Sometimes you will want to use data, such as plate layouts, which are in a plate-shaped layout instead of a column format. For example:

<img src="docs/source/_static/plate-screenshot.png" alt="" width="800">

You can convert this to column format using `hts pivot`, which produces a table in the following format:

| row_id | column_id | compound_name | well_id | plate_id                 | filename                 |
| ------ | --------- | ------------- | ------- | ------------------------ | ------------------------ |
| C	     | 2	       | RIF	         | C02	   | Plate 7	                | compounds.xlsx           |
| D	     | 2	       | LYSINE	       | D02	   | Plate 7	                | compounds.xlsx           |
| E	     | 2	       | DMSO	         | E02	   | Plate 7	                | compounds.xlsx           |
| F	     | 2	       | RIF	         | F02	   | Plate 7	                | compounds.xlsx           |

It is assumed that there is one plate per sheet for Excel files, and one plate per file for TSV and CSV files. The
plate name is taken from the sheet name (Excel) or filename (other formats).

You can prepend the names of the `plate_id` and `filename` columns with the `-x` option. for example, 
`hts pivot compounds.xlsx -x compound_source --name compound_name` would have columns `compound_source_plate_id`
`compound_source_filename`. This is helpful when usign `hts join` later where the plate and filename 
columns would otherwise be shared but have different meanings and values, and you don't want to accidentally 
join on them.

### Statistical testing

Groups of values (such as replicates) can be compared against a negative control for statistical testing using
`hts summarize`. The `--grouping` option indicates the columns whose values together indicate values which are
replicates of a particular condition of interest. For example, `--grouping strain_name compound_name` would
indicate that values which have the same `strain_name` and `compound_name` are replicates.

Statistical tests compare to the `--negative` values, and use all measured and normalized (`calc_*_norm`) columns.
Currently, the Student's t-test and Mann-whitney U-test are implemented. The t-test is best suited to Normal-distributed
data while the MWU is better for other distributions which might not have a nice bell curve distribution.

Although `hts summarize` calcualtes both tests simultaneously, it's not a good idea to look for "significant" $p$-values 
in both. This is called p-hacking, and leads to false positives. Instead, decide which test is most appropriate for your
data and stick with that one.

This command also calculates other summary statistics such as between-replicate mean, variance, and SSMD. If a filename
prefix is provided to the `--plot` option, then volcano and flashlight plots are produced, which may be useful for identifying 
hits of high throughput screens.

<img src="docs/source/_static/summary-counter_fluor_ch1_norm_volcano.png" alt="" width="400">
<img src="docs/source/_static/summary-counter_fluor_ch1_norm_flashlight.png" alt="" width="400">

## Other commands

There are several other commands from **hts-tools** which take as input the output from `hts parse`, joined
to an experimental data table (`layout.xlsx` in these examples).

- `hts qc`

Do quality cpntrol checks by calculating mean, standard deviation, Z\'-factor, and SSMD, and plotting them.

```bash
hts parse plate-?.xlsx --data-shape row  \
  | hts join --right layout.xlsx \
  | hts qc --control compound_name --positive RIF --negative DMSO --grouping strain_name plate_id --plot qc-plot \
  > qc.tsv
```

<img src="docs/source/_static/qc-plot_fluor_ch1_mean_sd.png" alt="" width="400">
<img src="docs/source/_static/qc-plot_fluor_ch1_zprime.png" alt="" width="400">

- `hts plot-hm`

Plot heatmaps of signal intensity arranged by plate well. This can be useful to identify unwanted within-plate variability.

```bash
hts parse plate-?.xlsx --data-shape row \
  | hts join --right layout.xlsx \
  | hts plot-hm --grouping strain_name sample_id plate_id --output hm
```

Here, `--grouping` identifies the columns which indicate values coming from the same plate. One file is produced per
measured and normalized (`calc_*_norm`) column.

<img src="docs/source/_static/hm_calc_fluor_ch1_norm_heatmap.png" alt="" width="400">

- `hts plot-rep`

Plot two replicates against each other for each condition.

```bash
hts parse plate-?.xlsx --data-shape row  \
  | hts join --right layout.xlsx \
  | hts plot-rep --control compound_name --positive RIF --negative DMSO --grouping strain_name compound_name --output rep
```

Here, `--grouping` identifies the unique conditions within which values are treated as replicates. The positives and negatives
are plotted as different colors. One file is produced per measured and normalized (`calc_*_norm`) column.

<img src="docs/source/_static/rep_measured_fluor_ch1_replicates.png" alt="" width="400">

In the plots, the left column is on a linear-linear scale and the right column is on a log-log scale. There is one row
of plots per wavelength set in the dataset.

- `hts plot-hist`

Plot histograms of the data values.

```bash
hts parse plate-?.xlsx --data-shape row \
  | hts join --right layout.xlsx \
  | hts plot-hist --control compound_name --positive RIF --negative DMSO --output hist
```

The positives and negatives are plotted as different colors. One file is produced per measured and normalized (`calc_*_norm`) column.

<img src="docs/source/_static/hist_measured_fluor_ch1_histogram.png" alt="" width="400">

In the plots, the left two columns are on a linear-linear scale and the right two columns are on a log-log scale. There is one row
of plots per wavelength set in the dataset.

