# miRNA target site prediction Benchmarks

## Installation

```bash
pip install git+https://github.com/katarinagresova/miRBench.git
```

## Examples

### Get all available datasets

```python
import miRBench

miRBench.dataset.list_datasets()
```

```python
['Hejret2023']
```

### Get dataset

```python
dataset_name = "Hejret2023"
df = miRBench.dataset.get_dataset(dataset_name, split="test", ratio="1")
df.head()
```

|	| noncodingRNA	| gene |	label |
| -------- | ------- | ------- | ------- |
| 0 |	TCCGAGCCTGGGTCTCCCTCTT	 |GGGTTTAGGGAAGGAGGTTCGGAGACAGGGAGCCAAGGCCTCTGTC... |	1 |
|1 |	TGCGGGGCTAGGGCTAACAGCA	|GCTTCCCAAGTTAGGTTAGTGATGTGAAATGCTCCTGTCCCTGGCC...	| 1 |
| 2 |	CCCACTGCCCCAGGTGCTGCTGG	|TCTTTCCAAAATTGTCCAGCAGCTTGAATGAGGCAGTGACAATTCT...	| 1 |
| 3 |	TGAGGGGCAGAGAGCGAGACTTT	|CAGAACTGGGATTCAAGCGAGGTCTGGCCCCTCAGTCTGTGGCTTT...	| 1 |
| 4	 |CAAAGTGCTGTTCGTGCAGGTAG	|TTTTTTCCCTTAGGACTCTGCACTTTATAGAATGTTGTAAAACAGA...	| 1 |

Data will be downloaded to `$HOME / ".miRBench" / "datasets"` directory, under separate subdirectories for each dataset.

### Get all available tools

```python
miRBench.predictor.list_predictors()
```
```python
['cnnMirTarget',
 'RNACofold',
 'HejretMirnaCnn',
 'miRBind',
 'TargetNet',
 'Seed8mer',
 'Seed7mer',
 'Seed6mer',
 'Seed6merBulgeOrMismatch',
 'TargetScanCnn',
 'YangAttention']
```

### Encode dataset

```python
tool = 'miRBind'
encoder = miRBench.encoders.get_encoder(tool)

input = encoder(df)
```

### Get predictions

```python
predictor = miRBench.predictors.get_predictor(tool)

predictions = predictor(input)
predictions[:10]
```

```python
array([0.6899161 , 0.15220629, 0.07301956, 0.43757868, 0.34360734,
       0.20519172, 0.0955029 , 0.79298246, 0.14150576, 0.05329492],
      dtype=float32)
```

## Benchmark all tools on all datasets

```bash
python benchmark_all.py OUTPUT_FOLDER_PATH
```

The script will run all tools on all datasets and will produce a file with suffix `_predictions.tsv` for each dataset. Predictions from every tool will be saved in separate columns.