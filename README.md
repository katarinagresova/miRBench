# miRNA target site prediction Benchmarks

## Installation

miRBench package can be easily installed using pip:

```bash
pip install miRBench
```

Default installation allows access to the datasets. To use predictors and encoders, you need to install additional dependencies.

### Dependencies for predictors and encoders

To use miRBench with predictors and encoders, install the following dependencies:
- numpy
- biopython
- viennarna
- torch
- tensorflow
- typing-extensions

To install the miRBench package with all dependencies into a virtual environment, you can use the following commands:

```bash
python3.8 -m venv mirbench_venv
source mirbench_venv/bin/activate
pip install miRBench
pip install numpy==1.24.3 biopython==1.83 viennarna==2.7.0 torch==1.9.0 tensorflow==2.13.1 typing-extensions==4.5.0
```

Note: This instalation is for running predictors on the CPU. If you want to use GPU, you need to install version of torch and tensorflow with GPU support.

## Examples

### Get all available datasets

The dataset module is responsible for access to the benchmark datasets described in the [miRBench paper](https://doi.org/10.1101/2024.11.14.623628).

```python
from miRBench.dataset import list_datasets

list_datasets()
```

```python
['AGO2_CLASH_Hejret2023',
 'AGO2_eCLIP_Klimentova2022',
 'AGO2_eCLIP_Manakov2022']
```

Not all datasets are available with all splits. To get available splits, use the `full` option.

```python
list_datasets(full=True)
```

```python
{'AGO2_CLASH_Hejret2023': {'splits': ['train', 'test']},
 'AGO2_eCLIP_Klimentova2022': {'splits': ['test']},
 'AGO2_eCLIP_Manakov2022': {'splits': ['train', 'test', 'leftout']}}
```

### Get dataset

```python
from miRBench.dataset import get_dataset_df

dataset_name = "AGO2_CLASH_Hejret2023"
df = get_dataset_df(dataset_name, split="test")
df.head()
```

|	| gene	| noncodingRNA	| noncodingRNA_name	| noncodingRNA_fam	| feature	| label	| chr	| start	| end	| strand	| gene_cluster_ID |
| -------- | ------- | ------- | ------- | -------- | ------- | ------- | ------- | -------- | ------- | ------- | ------- |
|0	|AAAGCTGTGGAACGCTACCTCTTCCTTTGAG...	|TGAGGTAGTAGGTTGTATAGTT	|hsa-let-7a-5p	|let-7	|exon	|1	|1	|212100882	|212100931	|+	|2391|
|1	|TCACCTCAGACTCTGTCCAACCTCTGCCTCA...	|TGAGGTAGTAGGTTGTGTGGTT	|hsa-let-7a-5p	|let-7	|exon	|1	|1	|35913919	|35913968	|+	|3972|
|2	|TTATATGTGCCCAGTGTGGCAAAACCTTCAA...	|TGAGGTAGTAGGTTGTATAGTT	|hsa-let-7a-5p	|let-7	|exon	|1	|1	|42851209	|42851258	|+	|222|
|3	|TGAGGCCCTCTTCCTGCTCGTCACCTCCGTC...	|TGAGGTAGTAGGTTGTATAGTT	|hsa-let-7a-5p	|let-7	|exon	|1	|1	|43961210	|43961259	|+	|1253|
|4	|ATAAAATTTACGTTTTTAACTATACAATCTAC...	|TGAGGTAGTAGGTTGTATAGTT	|hsa-let-7a-5p	|let-7	|intron	|1	|1	|244661046	|244661095	|+	|1252|

If you want to get just a path to the dataset, use the `get_dataset_path` function:

```python
from miRBench.dataset import get_dataset_path

dataset_path = get_dataset_path(dataset_name, split="test")
dataset_path
```

```python
/home/user/.miRBench/datasets/14501607/AGO2_CLASH_Hejret2023/test/dataset.tsv
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

The encoder module is responsible for encoding data into the format expected by a predictor module. The main function of the module is `get_encoder(predictor_name)` which returns an instance of an encoder object implemented for a specified predictor. The encoder expects data as a Pandas DataFrame with columns named ‘noncodingRNA’ and ‘gene’. Specifying custom column names is possible when calling the encoder. The returned data format differs for every encoder and is specific to the predictor.

```python
from miRBench.encoder import get_encoder

tool = 'miRBind_Klimentova2022'
encoder = get_encoder(tool)

input = encoder(df)
```

### Get predictions

The predictor module is responsible for predicting miRNA-binding site interaction. The main function of the module is `get_predictor(predictor_name)` which returns an instance of a specified predictor object. The predictor object expects data encoded by a corresponding encoder and returns an array of predictions.

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

## Citing miRBench

If you use miRBench in your research, please cite the following article:

Sammut, Stephanie, et al. [miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA Frequency Class Bias.](https://doi.org/10.1101/2024.11.14.623628) bioRxiv (2025): 2025-01.
