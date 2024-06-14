import miRBench
import argparse

def main():
    parser = argparse.ArgumentParser(description="Benchmark all available predictors on all available datasets")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for predictions")
    args = parser.parse_args()

    benchmark_all(args.out_dir)

def benchmark_all(out_dir):
    # loop over all available datasets
    for dset in miRBench.dataset.list_datasets():
        for ratio in ["1", "10", "100"]:
            print(f"Downloading {dset} dataset, ratio {ratio}")
            df = miRBench.dataset.get_dataset(dset, split="test", ratio=ratio)

            # loop over all available predictors
            for tool in miRBench.predictor.list_predictors():
                print(f"Running {tool} on {dset} dataset, ratio {ratio}")
                
                encoder = miRBench.encoder.get_encoder(tool)
                predictor = miRBench.predictor.get_predictor(tool)

                input = encoder(df)
                output = predictor(input)

                df[tool] = output

            df.to_csv(f"{out_dir}/{dset}_{ratio}_predictions.tsv", sep="\t")

if __name__ == "__main__":
    main()