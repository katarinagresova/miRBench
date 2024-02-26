# miRNA Benchmarks

Setup conda environment
```bash
. prepare_conda.sh
```

Download data
```bash
python data/download_data.py [--helwak] [--hejret] [--klimentova]
```

Data will be downloaded to `data/` directory, under separate subdirectories for each dataset.

Run some tool, for example TargetScan CNN:
```bash
python tools/targetscan.py \
    --input [PATH_TO_INPUT_DATA_TSV] \
    --miRNA_column miRNA \
    --gene_column gene \
    --output [PATH_TO_OUTPUT_DATA_TSV]
```

Tool will extend input data with predictions (as a new column) and save it to output file.
