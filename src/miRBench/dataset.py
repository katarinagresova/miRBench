import urllib.request
import pandas as pd
from pathlib import Path
import gzip
import shutil

ZENODO_RECORD_ID = "14501607"
CACHE_PATH = Path.home() / ".miRBench" / "datasets" / ZENODO_RECORD_ID
DATASET_FILE = "dataset.tsv"

def list_datasets(full=False):

    datasets = {
        "AGO2_CLASH_Hejret2023": {
            "splits": ['train', 'test']
        },
        "AGO2_eCLIP_Klimentova2022": {
            "splits": ['test']
        },
        "AGO2_eCLIP_Manakov2022": {
            "splits": ['train', 'test', 'leftout']
        }
    }

    if full:
        return datasets
    else:
        return list(datasets.keys())

def get_dataset_df(dataset_name, split, force_download = False):
    """
    Get dataset from cache or download it if not present.
    Returns dataset as pandas DataFrame.
    """

    local_path = Path(CACHE_PATH / dataset_name / split / DATASET_FILE)
    if not local_path.exists() or force_download:
        print(f"Downloading {dataset_name} dataset, split {split} into {local_path}")
        download_dataset(dataset_name, local_path, split)
    else:
        print(f"Using cached dataset {local_path}")

    dataset = pd.read_csv(local_path, sep="\t")
    return dataset

def get_dataset_path(dataset_name, split, force_download = False):
    """
    Get dataset from cache or download it if not present.
    Returns path to dataset file.
    """

    local_path = Path(CACHE_PATH / dataset_name / split / DATASET_FILE)
    if not local_path.exists() or force_download:
        print(f"Downloading {dataset_name} dataset, split {split} into {local_path}")
        download_dataset(dataset_name, local_path, split)
    else:
        print(f"Using cached dataset {local_path}")

    return local_path

def download_dataset(dataset_name, download_path, split):

    available_datasets = list_datasets(full=True)
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset {dataset_name} not found")
    if split not in available_datasets[dataset_name]["splits"]:
        raise ValueError(f"Split {split} not found for dataset {dataset_name}")

    url = f'https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{dataset_name}_{split}.tsv.gz?download=1'
    
    data_dir = Path(download_path).parent

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    downlaod_path_gz = download_path.with_suffix(".tsv.gz")
    urllib.request.urlretrieve(url, downlaod_path_gz)

    with gzip.open(downlaod_path_gz, 'rb') as f_in:
        with open(download_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

