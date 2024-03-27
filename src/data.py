import argparse
import os
import urllib.request
import gdown

def download_helwak(folder_path):
    """
    Download Helwak 2013 Ago1-CLASH dataset
    """

    data_dir = os.path.join(folder_path, "Helwak_2013")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, parents=True)

    for ratio in [1, 10, 100]:
        url = f"https://github.com/ML-Bioinfo-CEITEC/miRBind/raw/main/Datasets/test_set_1_{ratio}_CLASH2013_paper.tsv"
        filename = os.path.join(data_dir, f"miRNA_test_set_{ratio}.tsv")
        urllib.request.urlretrieve(url, filename)

def download_hejret(folder_path):
    """
    Download Hejret 2023 Ago2-CLASH dataset
    """

    data_dir = os.path.join(folder_path, "Hejret_2023")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, parents=True)

    for ratio in [1, 10, 100]:
        for smallRNA in ['miRNA', 'miRNA_real_seq', 'tRNA', 'yRNA']:
            url = f"https://github.com/ML-Bioinfo-CEITEC/HybriDetector/raw/main/ML/Datasets/{smallRNA}_test_set_{ratio}.tsv"
            filename = os.path.join(data_dir, f"{smallRNA}_test_set_{ratio}.tsv")
            urllib.request.urlretrieve(url, filename)

def download_klimentova(folder_path):
    """
    Download Klimentova 2022 Ago2-eCLIP dataset
    """

    data_dir = os.path.join(folder_path, "Klimentova_2022")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, parents=True)

    url = "https://drive.google.com/file/d/18IXpYEFjrWCVF-eJtj28Ezk4HLy-K-FZ/view?usp=sharing"
    filename = os.path.join(data_dir, "miRNA_test_set_1.tsv")
    gdown.download(url, filename, quiet=False, fuzzy=True)

    url = "https://drive.google.com/file/d/1X0PBzLuyh6khzgWwVgMqUeHqrxVfECfD/view?usp=sharing"
    filename = os.path.join(data_dir, "miRNA_test_set_10.tsv")
    gdown.download(url, filename, quiet=False, fuzzy=True)

    url = "https://drive.google.com/file/d/12o72sLUTEtoNLt8t4ubnzyqi_Wd-n6kC/view?usp=sharing"
    filename = os.path.join(data_dir, "miRNA_test_set_100.tsv")
    gdown.download(url, filename, quiet=False, fuzzy=True)

    url = "https://drive.google.com/file/d/12fH7uJgfyhSsod55iFq0wEfEcTNLss74/view?usp=sharing"
    filename = os.path.join(data_dir, "tRNA_test_set_1.tsv")
    gdown.download(url, filename, quiet=False, fuzzy=True)

def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--path", type=str, help="Path to the directory where the datasets will be downloaded")
    parser.add_argument("--helwak", action="store_true", help="Download Helwak 2013 Ago1-CLASH dataset")
    parser.add_argument("--hejret", action="store_true", help="Download Hejret 2023 Ago2-CLASH dataset")
    parser.add_argument("--klimentova", action="store_true", help="Download Klimentova 2022 Ago2-eCLIP dataset")

    args = parser.parse_args()

    if args.helwak:
        download_helwak(args.path)
    if args.hejret:
        download_hejret(args.path)
    if args.klimentova:
        download_klimentova(args.path)
    
if __name__ == "__main__":
    main()
