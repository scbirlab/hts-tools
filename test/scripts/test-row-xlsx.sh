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

hts parse $DATADIR/plate-?.xlsx --data-shape row \
    | hts join --right $DATADIR/layout.xlsx \
    | hts normalize -c norm_controls -p pos -n neg -g guide_name \
    | hts plot-dose -x concentration -p guide_name -c compound_name \
        -o $OUTDIR/plt-test

touch $SUCCESFILE