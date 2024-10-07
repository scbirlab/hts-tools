from importlib.metadata import version
__version__ = version('hts-tools')

from .io import from_platereader
from .normalize import normalize
from .plot import (
    plot_dose_response,
    plot_mean_sd, 
    plot_zprime,
    plot_replicates, 
    plot_heatmap, 
    plot_histogram,
    plot_scatter,
)
from .qc import ssmd, z_prime_factor
from .summarize import summarize
from .tables import join, pivot_plate, replicate_table
from .utils import row_col_to_well