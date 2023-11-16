(python=)
# Python API

**hts-tools** can be imported into Python to help make custom analyses.

```python
>>> import htstools as hts
```

You can read raw exports from platereader software into a columnar Pandas dataframe.

```python
>>> hts.from_platereader("plates.xlsx", shape="plate", vendor="Biotek")
```

Once in the columnar format, you can annotate experimental conditions.

```python
>>> import pandas as pd
>>> a = pd.DataFrame(dict(column=['A', 'B', 'A', 'B'], 
...                       abs=[.1, .2, .23, .11]))
>>> a  
    column   abs
0      A  0.10
1      B  0.20
2      A  0.23
3      B  0.11
>>> b = pd.DataFrame(dict(column=['B', 'A'], 
...                       drug=['TMP', 'RIF']))
>>> b  
    column drug
0      B  TMP
1      A  RIF
>>> shared_cols, data = join(a, b)
>>> shared_cols
('column',)
>>> data 
column   abs drug
0      A  0.10  RIF
1      A  0.23  RIF
2      B  0.20  TMP
3      B  0.11  TMP
```

If the conditions to annotate are in a plate-shaped format, you can melt them into a columnar format before joining.

```python
>>> import pandas as pd
>>> import numpy as np
>>> a = pd.DataFrame(index=list("ABCDEFGH"), 
...                  columns=range(1, 13), 
...                  data=np.arange(1, 97).reshape(8, 12))
>>> a  
    1   2   3   4   5   6   7   8   9   10  11  12
A   1   2   3   4   5   6   7   8   9  10  11  12
B  13  14  15  16  17  18  19  20  21  22  23  24
C  25  26  27  28  29  30  31  32  33  34  35  36
D  37  38  39  40  41  42  43  44  45  46  47  48
E  49  50  51  52  53  54  55  56  57  58  59  60
F  61  62  63  64  65  66  67  68  69  70  71  72
G  73  74  75  76  77  78  79  80  81  82  83  84
H  85  86  87  88  89  90  91  92  93  94  95  96
>>> hts.pivot_plate(a, value_name="well_number")  
    row_id column_id  well_number well_id plate_id
0       A         1            1     A01         
1       B         1           13     B01         
2       C         1           25     C01         
3       D         1           37     D01         
4       E         1           49     E01         
..    ...       ...          ...     ...      ...
91      D        12           48     D12         
92      E        12           60     E12         
93      F        12           72     F12         
94      G        12           84     G12         
95      H        12           96     H12         

[96 rows x 5 columns]
```

This also works on the multi-sheet dictionary output of `pd.read_excel(..., sheet_names=None)`.

```python
>>> hts.pivot_plate({'sheet_1': a}, value_name="well_number")    
row_id column_id  well_number well_id plate_id
0       A         1            1     A01  sheet_1
1       B         1           13     B01  sheet_1
2       C         1           25     C01  sheet_1
3       D         1           37     D01  sheet_1
4       E         1           49     E01  sheet_1
..    ...       ...          ...     ...      ...
91      D        12           48     D12  sheet_1
92      E        12           60     E12  sheet_1
93      F        12           72     F12  sheet_1
94      G        12           84     G12  sheet_1
95      H        12           96     H12  sheet_1

[96 rows x 5 columns]
```

Replicates within condition groups can be annotated.

```python
>>> import pandas as pd
>>> a = pd.DataFrame(dict(group=['g1', 'g1', 'g2', 'g2'], 
...                       control=['n', 'n', 'p', 'p'], 
...                       m_abs_ch1=[.1, .2, .9, .8], 
...                       abs_ch1_wavelength=['600nm'] * 4))
>>> a 
    group control  m_abs_ch1 abs_ch1_wavelength
0    g1       n        0.1              600nm
1    g1       n        0.2              600nm
2    g2       p        0.9              600nm
3    g2       p        0.8              600nm
>>> hts.replicate_table(a, group='group') 
    group control  m_abs_ch1 abs_ch1_wavelength  replicate
0    g1       n        0.1              600nm          1
1    g1       n        0.2              600nm          2
2    g2       p        0.9              600nm          2
3    g2       p        0.8              600nm          1
```

If you prefer, you can get a "wide" output.

```python
>>> hts.replicate_table(a, group='group', wide='m_abs_ch1') 
replicate  rep_1  rep_2
group                  
g1           0.2    0.1
g2           0.8    0.9
```

Values can be normalized to values between 0 and 1 relative to their positive (0%) and negative (100%) controls, optinally within groups or batches.

```python
>>> import pandas as pd
>>> a = pd.DataFrame(dict(control=['n', 'n', '', '', 'p', 'p'], 
...                  m_abs_ch1=[.1, .2, .5, .4, .9, .8], 
...                  abs_ch1_wavelength=['600nm'] * 6))
>>> a 
    control  m_abs_ch1 abs_ch1_wavelength
0       n        0.1              600nm
1       n        0.2              600nm
2                0.5              600nm
3                0.4              600nm
4       p        0.9              600nm
5       p        0.8              600nm
>>> hts.normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1') 
    control  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm
0       n        0.1              600nm                0.15                0.85        1.071429
1       n        0.2              600nm                0.15                0.85        0.928571
2                0.5              600nm                0.15                0.85        0.500000
3                0.4              600nm                0.15                0.85        0.642857
4       p        0.9              600nm                0.15                0.85       -0.071429
5       p        0.8              600nm                0.15                0.85        0.071429
```

The scaling can be reversed with `flip=True`.

```python
>>> hts.normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1', flip=True) 
    control  m_abs_ch1 abs_ch1_wavelength  m_abs_ch1_neg_mean  m_abs_ch1_pos_mean  m_abs_ch1_norm
0       n        0.1              600nm                0.15                0.85       -0.071429
1       n        0.2              600nm                0.15                0.85        0.071429
2                0.5              600nm                0.15                0.85        0.500000
3                0.4              600nm                0.15                0.85        0.357143
4       p        0.9              600nm                0.15                0.85        1.071429
5       p        0.8              600nm                0.15                0.85        0.928571
```

Summary statstics and statsitcial tests relative to the negative controls can be generated.

```python

>>> a = pd.DataFrame(dict(gene=['g1', 'g1', 'g2', 'g2', 'g1', 'g1', 'g2', 'g2'], 
    ...                       compound=['n', 'n', 'n', 'n', 'cmpd1', 'cmpd1', 'cmpd2', 'cmpd2'], 
    ...                       m_abs_ch1=[.1, .2, .9, .8, .1, .3, .5, .45], 
    ...                       abs_ch1_wavelength=['600nm'] * 8))
    >>> a
        gene compound  m_abs_ch1 abs_ch1_wavelength
    0    g1        n       0.10              600nm
    1    g1        n       0.20              600nm
    2    g2        n       0.90              600nm
    3    g2        n       0.80              600nm
    4    g1    cmpd1       0.10              600nm
    5    g1    cmpd1       0.30              600nm
    6    g2    cmpd2       0.50              600nm
    7    g2    cmpd2       0.45              600nm
    >>> hts.summarize(a, measurement_col='m_abs_ch1', control_col='compound', neg='n', group='gene')
      gene abs_ch1_wavelength  m_abs_ch1_mean  m_abs_ch1_std  ...  m_abs_ch1_t.stat  m_abs_ch1_t.p  m_abs_ch1_ssmd  m_abs_ch1_log10fc
    0   g1              600nm          0.1750       0.095743  ...          0.361158       0.742922        0.210042           0.066947
    1   g2              600nm          0.6625       0.221265  ...         -1.544396       0.199787       -0.807183          -0.108233

    [2 rows x 12 columns]
    >>> hts.summarize(a, measurement_col='m_abs_ch1', control_col='compound', neg='n', group=['gene', 'compound'])
    gene compound abs_ch1_wavelength  m_abs_ch1_mean  ...  m_abs_ch1_t.stat  m_abs_ch1_t.p  m_abs_ch1_ssmd  m_abs_ch1_log10fc
    0   g1        n              600nm           0.150  ...          0.000000       1.000000        0.000000           0.000000
    1   g2        n              600nm           0.850  ...          0.000000       1.000000        0.000000           0.000000
    2   g1    cmpd1              600nm           0.200  ...          0.447214       0.711723        0.316228           0.124939
    3   g2    cmpd2              600nm           0.475  ...         -6.708204       0.044534       -4.743416          -0.252725

    [4 rows x 13 columns]
```
