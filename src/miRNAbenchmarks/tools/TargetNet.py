# Code adapted from https://github.com/seonwoo-min/TargetNet accessed on 2024-03-25

import pandas as pd
import numpy as np
import sys
import torch
from collections import OrderedDict

from targetnet.utils import set_seeds
from targetnet.model.model_utils import get_model
from targetnet.data import extended_seed_alignment, encode_RNA

from miRNAbenchmarks.utils import parse_args, get_model_path

class ModelConfig():
    def __init__(self, cfg, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.type = None
        self.num_channels = None
        self.num_blocks = None
        self.stem_kernel_size = None
        self.block_kernel_size = None
        self.pool_size = None

        for key, value in cfg.items():
            if key == "skip_connection":                self.skip_connection = value
            elif key == "num_channels":                 self.num_channels = value
            elif key == "num_blocks":                   self.num_blocks = value
            elif key == "stem_kernel_size":             self.stem_kernel_size = value
            elif key == "block_kernel_size":            self.block_kernel_size = value
            elif key == "pool_size":                    self.pool_size = value
            else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

class miRNA_CTS_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for miRNA-CTS pair data """
    def __init__(self, X):
        self.X = X 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def prepare_dataset(df, miRNA_col, gene_col, with_esa = True):

    X = []

    for _, row in df.iterrows():

        mirna_seq = row[miRNA_col].upper().replace("T", "U")
        mrna_seq = row[gene_col].upper().replace("T", "U")[:40]

        # align miRNA seed region with mRNA
        mirna_esa, cts_rev_esa, _ = extended_seed_alignment(mirna_seq, mrna_seq)
        # encode aligned miRNA-CTS pair
        X.append(torch.from_numpy(encode_RNA(mirna_seq, mirna_esa,
                                                mrna_seq, cts_rev_esa, with_esa)))

    dataset = miRNA_CTS_dataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    return dataloader

def prepare_model(model_path, device):
    model_cfg = ModelConfig({
        'skip_connection': True,
        'num_channels': [16, 16, 32],
        'num_blocks': [2, 1, 1],
        'stem_kernel_size': 5,
        'block_kernel_size': 3,
        'pool_size': 3
    })

    model, _ = get_model(model_cfg, with_esa=True)
    checkpoint = torch.load(model_path, map_location="cpu")

    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."): k = k[7:]
        state_dict[k] = v

    model.load_state_dict(state_dict)
    model.to(device)

    return model

def predict(model, dataloader):
    predictions = []
    model.eval()
    for X in dataloader:
        preds = model(X)
        preds = torch.sigmoid(preds).cpu().detach().numpy()
        predictions.extend(preds[:,0])
    
    return predictions

if __name__ == '__main__':
    args = parse_args('targetnet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(2020)

    data = pd.read_csv(args.input, sep='\t')
    dataloader = prepare_dataset(data, args.miRNA_column, args.gene_column)
    print("Data prepared")

    model_path = get_model_path(
        folder = 'TargetNet', 
        model_name = 'TargetNet.pt', 
        url = 'https://github.com/katarinagresova/TargetNet/raw/master/pretrained_models/TargetNet.pt'
    )
    model = prepare_model(model_path, device)
    print('Model loaded')

    preds = predict(model, dataloader)
    data['TargetNet'] = preds

    data.to_csv(args.output, sep='\t', index=False)