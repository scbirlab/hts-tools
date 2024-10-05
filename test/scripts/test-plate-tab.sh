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

set -x

hts pivot "$DATADIR/compounds.xlsx" \
    --name compound_name \
    --prefix compound_source \
    -o $OUTDIR/pivoted-compounds.tsv

hts parse $DATADIR/170423*.txt --data-shape plate -o $OUTDIR/parsed.tsv
hts join $OUTDIR/parsed.tsv --right $DATADIR/sample-sheet.csv --o $OUTDIR/parsed-joined.tsv
hts join $OUTDIR/parsed-joined.tsv --right $OUTDIR/pivoted-compounds.tsv -o $OUTDIR/parsed-joined-ann.tsv
hts normalize $OUTDIR/parsed-joined-ann.tsv \
    -c compound_name -p RIF -n DMSO \
    -g strain_name sample_id plate_id \
    -o $OUTDIR/parsed-joined-ann-norm.tsv

normfile=$OUTDIR/parsed-joined-ann-norm.tsv
hts plot-hm $normfile \
      -g strain_name sample_id plate_id \
      --output $OUTDIR/hm

hts plot-rep $normfile \
    -c compound_name -p RIF -n DMSO -g strain_name compound_name \
    --output $OUTDIR/rep

hts plot-hist $normfile \
    -c compound_name -p RIF -n DMSO \
    --output $OUTDIR/hist

hts qc $normfile \
    -c compound_name \
    -p RIF -n DMSO -g strain_name sample_id plate_id \
    --plot $OUTDIR/qc-plot \
    > $OUTDIR/qc.tsv

hts summarize $normfile \
    -c strain_name \
    -p RIF -n WT -g strain_name compound_name \
    --plot $OUTDIR/summary-counter \
    > $OUTDIR/summary-counter.tsv

touch $SUCCESFILE

set +x