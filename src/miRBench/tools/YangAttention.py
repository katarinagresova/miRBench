# Code adapted from http://cosbi2.ee.ncku.edu.tw/mirna_binding/download accessed on 2024-03-21

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import os

from miRBench.utils import parse_args, get_model_path

miRNA_MAXLEN = 30
mRNA_MAXLEN = 40
RNA = list('ATCG')
MODEL_TYPE = 'attention'

class FELayer(nn.Module):
    def __init__(self, layer_infos, last_norm=True, norm_type='batch', bias=True):
        super(FELayer, self).__init__()
        
        self.linear_layers = nn.Sequential()
        for idx, li in enumerate(layer_infos):
            self.linear_layers.add_module(f'linear_{idx}', nn.Linear(li[0], li[1], bias=bias))
            if idx != len(layer_infos)-1 or (idx == len(layer_infos)-1 and last_norm):
                self.linear_layers.add_module(f'bn_{idx}', nn.LayerNorm(li[1]) if norm_type != 'batch' else nn.BatchNorm1d(li[1]))
                self.linear_layers.add_module(f'relu_{idx}', nn.PReLU())
                if len(li) == 3:
                    self.linear_layers.add_module(f'dropout_{idx}', nn.Dropout(li[2]))
        
    def forward(self, x):
        return self.linear_layers(x)   

class SEblock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.PReLU()#(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, channels)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        input_x = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1)

        return input_x * x

class SELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=4, add_residual=False, res_dim=16):
        super(SELayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.PReLU()#(inplace=True)
        self.se = SEblock(channels=out_channels, reduction=reduction)#ChannelAttentionBlock()#
        
        if add_residual:
            self.conv2 = nn.Conv1d(in_channels=res_dim, out_channels=out_channels, kernel_size=1)
            self.bn2 = nn.BatchNorm1d(out_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x, residual=None):
        x = self.conv(x)
        x = self.bn1(x)
        if residual is not None:
            x = x + self.bn1(self.conv2(residual))
        x = self.relu(x)
        x = self.se(x)

        return x

class BaseLine7(nn.Module):
    def __init__(self, emb_dim=4, cnn_dim=16, kernel=5, se_reduction=4, cnn_dropout=0.3, mrna_len=31, pirna_len=21, nhead=4, transformer_dropout=0.5, cls_dropout=0.5):
        super(BaseLine7, self).__init__()
        self.mrna_len = mrna_len
        self.pirna_len = pirna_len
        self.emb_dim = emb_dim 
        self.cnn_dim = cnn_dim 
        self.kernel = kernel
        self.se_reduction = se_reduction
        self.cnn_dropout = cnn_dropout
        self.nhead = nhead
        self.transformer_dropout = transformer_dropout
        self.cls_dropout = cls_dropout
        
        # Feature Extraction
        self.mrna_conv = SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
        self.mrna_dropout = nn.Dropout(cnn_dropout)
        self.pirna_conv = SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
        self.pirna_dropout = nn.Dropout(cnn_dropout)
        
        # Transformer Decoder
        self.d_model = cnn_dim
        #self.self_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=transformer_dropout)
        self.multihead_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=transformer_dropout)
        
        # Feedforward model
        self.linear1 = nn.Linear(self.d_model, self.d_model*4)
        self.dropout = nn.Dropout(transformer_dropout)
        self.linear2 = nn.Linear(self.d_model*4, self.d_model)
        #self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        #self.dropout1 = nn.Dropout(transformer_dropout)
        self.dropout2 = nn.Dropout(transformer_dropout)
        self.dropout3 = nn.Dropout(transformer_dropout)
        self.activation = nn.PReLU()#nn.GELU()#

        # Classification
        self.dropout4 = nn.Dropout(cls_dropout)
        self.cls_input_dim = self.d_model*mrna_len
        self.cls_layer = FELayer([
            [self.cls_input_dim, 1024, cls_dropout],
            [1024, 256, cls_dropout],
            [256, 2]
        ], last_norm=False, norm_type='batch')
        # global average pooling 
        self.global_avg_pool = nn.AvgPool1d(mrna_len)
        
    def forward(self, x):
        # Feature Extraction
        mrna_x = x[:, :self.mrna_len].permute(0, 2, 1)
        tgt = self.mrna_dropout(self.mrna_conv(mrna_x)).permute(2, 0, 1)
        self.fm_mrna = tgt.detach()
        pirna_x = x[:, self.mrna_len:].permute(0, 2, 1)
        memory = self.pirna_dropout(self.pirna_conv(pirna_x)).permute(2, 0, 1)
        self.fm_pirna = memory.detach()
        # Transformer Decoder
        #tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=generate_square_subsequent_mask(self.mrna_len, self.mrna_len, avail_device), key_padding_mask=None)[0]
        #tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=None, key_padding_mask=None)[0]
        #tgt = tgt + self.dropout1(tgt2)
        #tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout2(tgt2)
        self.fm_attn = tgt.detach()
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        out = self.norm3(tgt)
        
        # Classification
        out = self.dropout4(out)
        out = out.permute(1, 2, 0)
        # out = self.global_avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.cls_layer(out)
        
        return out


def get_model(model_path, device):

    model = BaseLine7(
                emb_dim=len(RNA), 
                cnn_dim=256,
                kernel=5,
                se_reduction=4,
                cnn_dropout=0.5,
                mrna_len=mRNA_MAXLEN, 
                pirna_len=miRNA_MAXLEN, 
                nhead=16,
                transformer_dropout=0.5,
                cls_dropout=0.75
            )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def data_preprocessing(df, miRNA_col, gene_col, device, batch_size=256, drop_last=False, shuffle=False):

    onehot = {'A':[1,0,0,0],
              'T':[0,1,0,0],
              'U':[0,1,0,0],
              'C':[0,0,1,0],
              'G':[0,0,0,1],
              'L':[0,0,0,0]
             }
    x = []
    for _, row in df.iterrows():

        miRNA = row[miRNA_col][:30]
        # padding miRNA with L to the length of 30
        miRNA = miRNA + 'L'*(30-len(miRNA))

        # taking first 40 characters of gene
        gene = row[gene_col][:40]

        seq = gene + miRNA
        x.append([onehot[c] for c in seq])

    tensor_x = torch.tensor(x).float().to(device=device)
    dataset = TensorDataset(tensor_x)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def predict(model, df, miRNA_col, gene_col, device):

    model.eval()
    data_loader = data_preprocessing(df, miRNA_col, gene_col, device)
    preds = []
    for (batch, ) in data_loader:

        output = model(batch).detach()
        output = F.softmax(output, dim=-1)[:, 1].cpu().numpy()
        preds.extend(output)
        
    preds = np.array(preds) 
    return preds

if __name__ == '__main__':
    args = parse_args('Yang 2023')

    data = pd.read_csv(args.input, sep='\t')

    model_path = get_model_path(
        folder = 'Yang_Attention', 
        model_name = 'attention_model.pkl', 
        url = 'https://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/Yang_Attention/attention_model.pkl'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_path, device)

    preds = predict(model, data, args.miRNA_column, args.gene_column, device)
    data['YangAttention'] = preds

    data.to_csv(args.output, sep='\t', index=False)