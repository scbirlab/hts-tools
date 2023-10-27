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
>>> pivot_plate(a, value_name="well_number")  
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
>>> pivot_plate({'sheet_1': a}, value_name="well_number")    
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
>>> replicate_table(a, group='group') 
    group control  m_abs_ch1 abs_ch1_wavelength  replicate
0    g1       n        0.1              600nm          1
1    g1       n        0.2              600nm          2
2    g2       p        0.9              600nm          2
3    g2       p        0.8              600nm          1
```

If you prefer, you can get a "wide" output.

```python
>>> replicate_table(a, group='group', wide='m_abs_ch1') 
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
>>> normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1') 
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
>>> normalize(a, control_col='control', pos='p', neg='n', measurement_col='m_abs_ch1', flip=True) 
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
>>> summarize(a, measurement_col='m_abs_ch1', control_col='control', neg='n', group='group')
```