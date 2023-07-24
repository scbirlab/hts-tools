#!/usr/bin/env bash

set -e

DATADIR=test/data/biotek-plate-tab
OUTDIR=test/outputs/biotek-plate-tab
SUCCESFILE=test/outputs/SUCCESS2

if [ -e $SUCCESFILE ]
then
    rm $SUCCESFILE
fi

mkdir -p $OUTDIR

hts pivot "$DATADIR/compounds.xlsx" \
      --name compound_name \
      --prefix compound_source \
      > $OUTDIR/pivoted-compounds.tsv

hts parse $DATADIR/170423*.txt --data-shape plate \
    | hts join --right $DATADIR/sample-sheet.csv \
    | hts join --right $OUTDIR/pivoted-compounds.tsv \
    | hts normalize -c compound_name -p RIF -n DMSO -g strain_name sample_id plate_id \
    > $OUTDIR/normalized.tsv

hts plot-hm $OUTDIR/normalized.tsv \
      -g strain_name sample_id plate_id \
      --output $OUTDIR/hm

hts plot-rep $OUTDIR/normalized.tsv \
    -c compound_name -p RIF -n DMSO -g strain_name compound_name \
    --output $OUTDIR/rep

hts plot-hist $OUTDIR/normalized.tsv \
    -c compound_name -p RIF -n DMSO \
    --output $OUTDIR/hist

hts qc $OUTDIR/normalized.tsv -c compound_name \
    -p RIF -n DMSO -g strain_name sample_id plate_id \
    --plot $OUTDIR/qc-plot \
    > $OUTDIR/qc.tsv

hts summarize $OUTDIR/normalized.tsv -c strain_name \
    -p RIF -n WT -g strain_name compound_name \
    --plot $OUTDIR/summary-counter \
    > $OUTDIR/summary-counter.tsv

touch $SUCCESFILE