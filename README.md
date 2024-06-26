# miRNA Benchmarks

## Clone repo

```bash
git clone https://github.com/katarinagresova/miRBench.git
```

## Setup conda environment

Use prepared script to create conda environment with all necessary dependencies:
```bash
. prepare_conda.sh
```

## Install package

```bash
pip install -e .
```

## Download data

```bash
python src/miRBench/data.py <DATA_FOLDER_PATH> [--helwak] [--hejret] [--klimentova]
```

Data will be downloaded to `DATA_FOLDER_PATH/` directory, under separate subdirectories for each dataset.

## Run tools

Run some tool, for example TargetScan CNN:
```bash
python src/miRBench/tools/TargetScanCnn.py \
    --input [PATH_TO_INPUT_DATA_TSV] \
    --output [PATH_TO_OUTPUT_DATA_TSV]
```

Tool will extend input data with predictions (as a new column) and save it to output file.

All tools have the same interface. You can find available tools in `src/miRBench/tools/` directory.

## Run multiple tools on multiple datasets

```bash
. predict.sh
```

By default, the script will run all tools on all datasets. You can specify tools and datasets changing `TOOLS`, `INPUT_DATA` and `RATIO` variables in the script.

The script will produce a file with suffix `_predictions.tsv` for each dataset. Predictions from every tool will be saved in separate columns.