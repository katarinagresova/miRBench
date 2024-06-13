import numpy as np
import random
import torch
from Bio import pairwise2
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)

def get_encoder(model_name):
    if model_name == "HejrejMirnaCnn":
        return HejrejMirnaCnnEncoder()
    elif model_name == "cnnMirTarget":
        return cnnMirTargetEncoder()
    elif model_name == "TargetNet":
        return TargetNetEncoder()
    elif model_name == "TargetScanCnn":
        return TargetScanCnnEncoder()
    elif model_name == "YangAttention":
        return YangAttentionEncoder()
    elif model_name == "miRBind":
        return miRBindEncoder()
    elif model_name == "Seed":
        return SeedEncoder()
    elif model_name == "RNAcofold":
        return RNACofoldEncoder()
    else:
        raise ValueError(f"Model {model_name} not found")

class RNACofoldEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col

    def __call__(self, df):
        return self.data_preprocessing(df, self.miRNA_col, self.gene_col)

    def data_preprocessing(self, df, miRNA_col, gene_col):
        def merge_seq(row):
            return row[miRNA_col] + "&" + row[gene_col]

        df["merged_seq"] = df.apply(merge_seq, axis=1)
        return df["merged_seq"].values

class miRBindEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene", tensor_dim=(50, 20, 1)):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col
        self.tensor_dim = tensor_dim

    def __call__(self, df):
        return self.binding_encoding(df, self.miRNA_col, self.gene_col, self.tensor_dim)
    
    def binding_encoding(self, df, miRNA_col, gene_col, tensor_dim=(50, 20, 1)):
        """
        fun encodes miRNAs and mRNAs in df into binding matrices
        :param df: dataframe containing gene_col and miRNA_col columns
        :param tensor_dim: output shape of the matrix. If sequences are longer than tensor_dim, they will be truncated.
        :return: 2D binding matrix with shape (N, *tensor_dim)
        """

        # alphabet for watson-crick interactions.
        alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
        # create empty main 2d matrix array
        N = df.shape[0]  # number of samples in df
        shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
        # initialize dot matrix with zeros
        ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

        # compile matrix with watson-crick interactions.
        for index, row in df.iterrows():
            for bind_index, bind_nt in enumerate(row[gene_col][:tensor_dim[0]].upper()):
                for mirna_index, mirna_nt in enumerate(row[miRNA_col][:tensor_dim[1]].upper()):
                    base_pairs = bind_nt + mirna_nt
                    ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

        return ohe_matrix_2d
    
class HejrejMirnaCnnEncoder(miRBindEncoder):
    def __init__(self):
        super().__init__(miRNA_col="noncodingRNA", gene_col="gene", tensor_dim=(50, 20, 1))

class cnnMirTargetEncoder():

    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col
        self.x_cast = {"A":[1,0,0,0],"U":[0,1,0,0],\
            "T":[0,1,0,0],"G":[0,0,1,0],\
            "C":[0,0,0,1],"N":[0,0,0,0]}

    def __call__(self, df):
        return self.prepare_data(df, self.miRNA_col, self.gene_col)

    # function: randomly select bases for padding without 4 continuous nt pairing with miRNA sequence 
    def selected_padding(self, merged_seq):
        merged_seq_len = len(merged_seq)
        SEQ_LEN = 110
        head_seq = merged_seq[0:10]
        head_reverse_complement = reverse_complement(head_seq)
        
        head_reverse_complement = head_reverse_complement.replace("U","T")
        while True:
            flag = 0
            padding_seq = ""
            for i in range(SEQ_LEN-merged_seq_len):
                temp_base = random.choice("ATGC")
                padding_seq += temp_base
            if len(padding_seq) < 4:
                break
            for j in range(len(padding_seq)-4):
                if head_reverse_complement.find(padding_seq[j:j+4]) >= 0:
                    flag = 1
                    break
            if flag ==0:
                break
        return padding_seq


    # function: padding all the miRNA_target_seq to 110 nt and vectorize with one-hot encoding
    def transform_xdata(self, column):
        x_dataset = []
        for line in column:
            line = line.strip()
            line = line.replace("X","")   # remove "X" in the sequence
            line = line.upper()   # upper case
            # padding sequence to SEQ_LEN
            padding_seq = self.selected_padding(line)   # padding with no-4-pairing sequence
            # padding_seq = "N"*(110-len(line))    # padding with "N"
            line = line + padding_seq
            # vectorization
            temp_list = []
            for base in line:
                temp_list.append(self.x_cast[base])
            x_dataset.append(temp_list)
        return x_dataset

    # function: calculate the result based on the model
    def prepare_data(self, data, miRNA_column, gene_column):

        def merge_seq(row):
            merged_seq = row[miRNA_column] + row[gene_column]
            if len(merged_seq) < 110:
                merged_seq += "N"*(110-len(merged_seq))
            if len(merged_seq) >= 110:
                merged_seq = merged_seq[0:110]
            return merged_seq

        miRNA_target_seq = data.apply(lambda x: merge_seq(x), axis=1)

        # vectorization
        merged_seq_vector = self.transform_xdata(miRNA_target_seq) 

        merged_seq_vector_array = np.array(merged_seq_vector)

        return merged_seq_vector_array

class TargetNetEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene", with_esa=True):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col
        self.with_esa = with_esa
        self.score_matrix = self.get_score_matrix()

    def __call__(self, df):
        return self.prepare_dataset(df, self.miRNA_col, self.gene_col, self.with_esa)

    def get_score_matrix(self):
        score_matrix = {}  # Allow wobble
        for c1 in 'ACGU':
            for c2 in 'ACGU':
                if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
                    score_matrix[(c1, c2)] = 1
                elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
                    score_matrix[(c1, c2)] = 1
                else:
                    score_matrix[(c1, c2)] = 0

        return score_matrix

    def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa, with_esa):
        """ one-hot encoder for RNA sequences with/without extended seed alignments """
        chars = {"A":0, "C":1, "G":2, "U":3}
        if not with_esa:
            x = np.zeros((len(chars) * 2, 40), dtype=np.float32)
            for i in range(len(mirna_seq)):
                x[chars[mirna_seq[i]], 5 + i] = 1
            for i in range(len(cts_rev_seq)):
                x[chars[cts_rev_seq[i]] + len(chars), i] = 1
        else:
            chars["-"] = 4
            x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
            for i in range(len(mirna_esa)):
                x[chars[mirna_esa[i]], 5 + i] = 1
            for i in range(10, len(mirna_seq)):
                x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
            for i in range(5):
                x[chars[cts_rev_seq[i]] + len(chars), i] = 1
            for i in range(len(cts_rev_esa)):
                x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
            for i in range(15, len(cts_rev_seq)):
                x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1

        return x

    def extended_seed_alignment(self, mi_seq, cts_r_seq):
        """ extended seed alignment """
        alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[5:15], self.score_matrix, one_alignment_only=True)[0]
        mi_esa = alignment[0]
        cts_r_esa = alignment[1]
        esa_score = alignment[2]
        return mi_esa, cts_r_esa, esa_score

    def prepare_dataset(self, df, miRNA_col, gene_col, with_esa = True):

        X = []

        for _, row in df.iterrows():

            mirna_seq = row[miRNA_col].upper().replace("T", "U")
            mrna_seq = row[gene_col].upper().replace("T", "U")[:40]

            # align miRNA seed region with mRNA
            mirna_esa, cts_rev_esa, _ = self.extended_seed_alignment(mirna_seq, mrna_seq)
            # encode aligned miRNA-CTS pair
            X.append(torch.from_numpy(self.encode_RNA(mirna_seq, mirna_esa,
                                                    mrna_seq, cts_rev_esa, with_esa)))

        dataset = TargetNetEncoder.miRNA_CTS_dataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        return dataloader
    
    class miRNA_CTS_dataset(torch.utils.data.Dataset):
        """ Pytorch dataloader for miRNA-CTS pair data """
        def __init__(self, X):
            self.X = X 

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i]

class TargetScanCnnEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col

    def __call__(self, df):
        return self.prepare_data(df, self.miRNA_col, self.gene_col)

    def one_hot_encode(self, seq):
        if len(seq) == 0:
            return []
        """ 1-hot encode ATCG sequence """
        nt_dict = {
            'A': 0,
            'T': 1,
            'C': 2,
            'G': 3,
            'X': 4
        }
        targets = np.ones([5, 4]) / 4.0
        targets[:4, :] = np.eye(4)
        seq = [nt_dict[nt] for nt in seq]
        return list(targets[seq].flatten())

    def prepare_data(self, df, miRNA_column, gene_column):
        
        data = []
        
        for seq_id in range(len(df)):
            input_data = []
            for sub_seq in [df.iloc[seq_id][gene_column][i:i+12] for i in range(len(df.iloc[seq_id][gene_column])-11)]:
                seq_one_hot = self.one_hot_encode(sub_seq)
                mirseq_one_hot = self.one_hot_encode(df.iloc[seq_id][miRNA_column][:10])
                input_data.append(np.outer(mirseq_one_hot, seq_one_hot))

            input_data = np.stack(input_data)
            data.append(input_data)

        return data

class YangAttentionEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene", device="cpu"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col
        self.device = device

    def __call__(self, df):
        return self.data_preprocessing(df, self.miRNA_col, self.gene_col, self.device)

    def data_preprocessing(self, df, miRNA_col, gene_col, device, batch_size=256, drop_last=False, shuffle=False):

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
        
class SeedEncoder():
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col

    def __call__(self, df):
        # return array of [miRNA, reverse_complement(gene)] pairs
        return df.apply(lambda x: [x[self.miRNA_col], reverse_complement(x[self.gene_col])], axis=1)

def reverse_complement(self, st):
    nn = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(nn[n] for n in reversed(st))    
