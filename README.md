# miRNA target site prediction Benchmarks

## Installation

```bash
pip install miRBench
```

## Examples

### Get all available datasets

```python
from miRBench.dataset import list_datasets

list_datasets()
```

```python
['AGO2_CLASH_Hejret2023',
 'AGO2_eCLIP_Klimentova2022',
 'AGO2_eCLIP_Manakov2022']
```

Not all datasets are available with all splits and ratios. To get available splits and ratios, use the `full` option.

```python
list_datasets(full=True)
```

```python
{'AGO2_CLASH_Hejret2023': {'splits': {
      'train': {'ratios': ['10']},
      'test': {'ratios': ['1', '10', '100']}}},
 'AGO2_eCLIP_Klimentova2022': {'splits': {
      'test': {'ratios': ['1', '10', '100']}}},
 'AGO2_eCLIP_Manakov2022': {'splits': {
      'train': {'ratios': ['1', '10', '100']},
      'test': {'ratios': ['1', '10', '100']}}}
}
```

### Get dataset

```python
from miRBench.dataset import get_dataset_df

dataset_name = "AGO2_CLASH_Hejret2023"
df = get_dataset_df(dataset_name, split="test", ratio="1")
df.head()
```

|	| noncodingRNA	| gene |	label |
| -------- | ------- | ------- | ------- |
| 0 |	TCCGAGCCTGGGTCTCCCTCTT	 |GGGTTTAGGGAAGGAGGTTCGGAGACAGGGAGCCAAGGCCTCTGTC... |	1 |
|1 |	TGCGGGGCTAGGGCTAACAGCA	|GCTTCCCAAGTTAGGTTAGTGATGTGAAATGCTCCTGTCCCTGGCC...	| 1 |
| 2 |	CCCACTGCCCCAGGTGCTGCTGG	|TCTTTCCAAAATTGTCCAGCAGCTTGAATGAGGCAGTGACAATTCT...	| 1 |
| 3 |	TGAGGGGCAGAGAGCGAGACTTT	|CAGAACTGGGATTCAAGCGAGGTCTGGCCCCTCAGTCTGTGGCTTT...	| 1 |
| 4	 |CAAAGTGCTGTTCGTGCAGGTAG	|TTTTTTCCCTTAGGACTCTGCACTTTATAGAATGTTGTAAAACAGA...	| 1 |

If you want to get just a path to the dataset, use the `get_dataset_path` function:

```python
from miRBench.dataset import get_dataset_path

dataset_path = get_dataset_path(dataset_name, split="test", ratio="1")
dataset_path
```

```python
/home/user/.miRBench/datasets/13909173/AGO2_CLASH_Hejret2023/1/test/dataset.tsv
```

### Get all available tools

```python
from miRBench.predictor import list_predictors

list_predictors()
```
```python
['CnnMirTarget_Zheng2020',
 'RNACofold',
 'miRNA_CNN_Hejret2023',
 'miRBind_Klimentova2022',
 'TargetNet_Min2021',
 'Seed8mer',
 'Seed7mer',
 'Seed6mer',
 'Seed6merBulgeOrMismatch',
 'TargetScanCnn_McGeary2019',
 'InteractionAwareModel_Yang2024']
```

### Encode dataset

```python
from miRBench.encoder import get_encoder

tool = 'miRBind_Klimentova2022'
encoder = get_encoder(tool)

input = encoder(df)
```

### Get predictions

```python
from miRBench.predictor import get_predictor

predictor = get_predictor(tool)

predictions = predictor(input)
predictions[:10]
```

```python
array([0.6899161 , 0.15220629, 0.07301956, 0.43757868, 0.34360734,
       0.20519172, 0.0955029 , 0.79298246, 0.14150576, 0.05329492],
      dtype=float32)
```
