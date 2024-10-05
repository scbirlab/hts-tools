#!/usr/bin/env bash

set -e

DATADIR=test/data/biotek-row-xlsx
OUTDIR=test/outputs/biotek-row-xlsx
SUCCESFILE=test/outputs/SUCCESS

if [ -e $SUCCESFILE ]
then
    rm $SUCCESFILE
fi

mkdir -p $OUTDIR

set -x

hts parse $DATADIR/plate-?.xlsx --data-shape row -o $OUTDIR/parsed.tsv
hts join $OUTDIR/parsed.tsv --right $DATADIR/layout.xlsx  -o $OUTDIR/parsed-joined.tsv
hts normalize $OUTDIR/parsed-joined.tsv -c norm_controls -p pos -n neg -g guide_name \
    -o $OUTDIR/parsed-joined-norm.tsv
hts plot-dose $OUTDIR/parsed-joined-norm.tsv \
    -x concentration -p guide_name -c compound_name \
    --x-log \
    -o $OUTDIR/plt-test

touch $SUCCESFILE

set +x