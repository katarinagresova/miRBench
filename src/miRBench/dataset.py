import urllib.request
import pandas as pd
from pathlib import Path

CACHE_PATH = Path.home() / ".miRBench" / "datasets"
DATASET_FILE = "dataset.tsv"

def list_datasets():
    # TODO: add more information about datasets
    print(["Helwak2013", "Hejret2023", "Klimentova2022", "Manakov2022"])

def get_dataset(dataset_name, ratio, split, force_download = False):
    """
    Get dataset from cache or download it if not present
    """

    local_path = Path(CACHE_PATH / dataset_name / ratio / split / DATASET_FILE)
    if not local_path.exists() or force_download:
        download_dataset(dataset_name, local_path, ratio, split)

    dataset = pd.read_csv(local_path, sep="\t")
    return dataset

def download_dataset(dataset_name, download_path, ratio, split):

    # TODO add verification that combination of name, split and ratio is valid
    if dataset_name == "Helwak2013":
        url = f"https://github.com/ML-Bioinfo-CEITEC/miRBind/raw/main/Datasets/{split}_set_1_{ratio}_CLASH2013_paper.tsv"
    elif dataset_name == "Hejret2023":
        url = f"https://github.com/ML-Bioinfo-CEITEC/HybriDetector/raw/main/ML/Datasets/miRNA_{split}_set_{ratio}.tsv"
    elif dataset_name == "Klimentova2022":
        raise NotImplementedError("Klimentova dataset is not available for download")
    elif dataset_name == "Manakov2022":
        raise NotImplementedError("Manakov dataset is not available for download")
    else:
        raise ValueError("Unknown dataset name")
    
    data_dir = download_path.parent
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    urllib.request.urlretrieve(url, download_path)
