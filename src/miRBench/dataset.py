import urllib.request
import pandas as pd
from pathlib import Path

CACHE_PATH = Path.home() / ".miRBench" / "datasets"
DATASET_FILE = "dataset.tsv"

def list_datasets(full=False):

    datasets = {
        "AGO2_CLASH_Hejret2023": {
            "splits": {
                "train": {
                    "ratios": ["10"]
                },
                "test": {
                    "ratios": ["1", "10", "100"]
                }
            }
        },
        "AGO2_eCLIP_Klimentova2022": {
            "splits": {
                "test": {
                    "ratios": ["1", "10", "100"]
                }
            }
        },
        "AGO2_eCLIP_Manakov2022": {
            "splits": {
                "train": {
                    "ratios": ["1", "10", "100"]
                },
                "test": {
                    "ratios": ["1", "10", "100"]
                }
            }
        }
    }

    if full:
        return datasets
    else:
        return list(datasets.keys())

def get_dataset(dataset_name, ratio, split, force_download = False):
    """
    Get dataset from cache or download it if not present
    """

    local_path = Path(CACHE_PATH / dataset_name / ratio / split / DATASET_FILE)
    if not local_path.exists() or force_download:
        print(f"Downloading {dataset_name} dataset, ratio {ratio}, split {split}")
        download_dataset(dataset_name, local_path, ratio, split)

    dataset = pd.read_csv(local_path, sep="\t")
    return dataset

def get_dataset_path(dataset_name, ratio, split, force_download = False):
    """
    Get dataset path from cache or download it if not present
    """

    local_path = Path(CACHE_PATH / dataset_name / ratio / split / DATASET_FILE)
    if not local_path.exists() or force_download:
        print(f"Downloading {dataset_name} dataset, ratio {ratio}, split {split}")
        download_dataset(dataset_name, local_path, ratio, split)

    return local_path

def download_dataset(dataset_name, download_path, ratio, split):

    available_datasets = list_datasets(full=True)
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset {dataset_name} not found")
    if split not in available_datasets[dataset_name]["splits"]:
        raise ValueError(f"Split {split} not found for dataset {dataset_name}")
    if ratio not in available_datasets[dataset_name]["splits"][split]["ratios"]:
        raise ValueError(f"Ratio {ratio} not found for split {split} of dataset {dataset_name}")

    url = f'https://zenodo.org/records/11657638/files/{dataset_name}_{ratio}_{split}_dataset.tsv?download=1'
    
    data_dir = download_path.parent
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    urllib.request.urlretrieve(url, download_path)
