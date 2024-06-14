import numpy as np
import random
import torch
from Bio import pairwise2
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)

def get_encoder(model_name):
    if model_name == "HejretMirnaCnn":
        return HejretMirnaCnnEncoder()
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
    elif model_name == "RNACofold":
        return RNACofoldEncoder()
    else:
        raise ValueError(f"Model {model_name} not found")

class RNACofoldEncoder():
    """
    Based on Lorenz, Ronny, et al. "ViennaRNA Package 2.0." Algorithms for molecular biology 6 (2011): 1-14. https://doi.org/10.1186/1748-7188-6-26.
    Python implementation: https://pypi.org/project/ViennaRNA/

    Encodes miRNA and gene sequences into string for predicting RNAcofold secondary structure.
    Returns list of strings with miRNA and gene sequences separated by "&".
    """

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
    """
    Based on Klimentová, Eva, et al. "miRBind: A deep learning method for miRNA binding classification." Genes 13.12 (2022): 2323. https://doi.org/10.3390/genes13122323.
    Python implementation: https://github.com/ML-Bioinfo-CEITEC/miRBind

    Encodes miRNA and gene sequences into 2D-binding matrix.
    2D-binding matrix has shape (gene_max_len, miRNA_max_len, 1) and contains 1 for Watson-Crick interactions and 0 otherwise.
    Returns array with shape (num_of_samples, gene_max_len, miRNA_max_len, 1).
    """
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
    
class HejretMirnaCnnEncoder(miRBindEncoder):
    """
    Based on Hejret, Vaclav, et al. "Analysis of chimeric reads characterises the diverse targetome of AGO2-mediated regulation." Scientific Reports 13.1 (2023): 22895. https://doi.org/10.1038/s41598-023-49757-z.
    Python implementation: https://github.com/ML-Bioinfo-CEITEC/HybriDetector/tree/main

    Uses the same encoding as miRBindEncoder.
    Encodes miRNA and gene sequences into 2D-binding matrix.
    2D-binding matrix has shape (gene_max_len, miRNA_max_len, 1) and contains 1 for Watson-Crick interactions and 0 otherwise.
    Returns array with shape (num_of_samples, gene_max_len, miRNA_max_len, 1).
    """
    def __init__(self):
        super().__init__(miRNA_col="noncodingRNA", gene_col="gene", tensor_dim=(50, 20, 1))

class cnnMirTargetEncoder():
    """
    Based on Zheng, Xueming, et al. "Prediction of miRNA targets by learning from interaction sequences." Plos one 15.5 (2020): e0232578. https://doi.org/10.1371%2Fjournal.pone.0232578
    Python implementation: https://github.com/zhengxueming/cnnMirTarget

    Encodes miRNA and gene sequences using one-hot encoding, where each nucleotide is represented by a vector of length 4.
    A = [1, 0, 0, 0]
    U = [0, 1, 0, 0]
    T = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    C = [0, 0, 0, 1]
    N = [0, 0, 0, 0]
    miRNA and gene sequences are concatenated and padded to 110 nt. Padding is done with "N" nucleotide. If the sequence is longer than 110 nt, it is truncated.
    Returns numpy array with shape (num_of_samples, 110, 4).
    """

    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col
        self.x_cast = {"A":[1,0,0,0],"U":[0,1,0,0],\
                       "T":[0,1,0,0],"G":[0,0,1,0],\
                       "C":[0,0,0,1],"N":[0,0,0,0]}

    def __call__(self, df):
        return self.prepare_data(df, self.miRNA_col, self.gene_col)

    # function: padding all the miRNA_target_seq to 110 nt and vectorize with one-hot encoding
    def transform_xdata(self, column):
        x_dataset = []
        for line in column:
            line = line.strip()
            line = line.replace("X","")   # remove "X" in the sequence
            line = line.upper()   # upper case

            x_dataset.append([self.x_cast[base] for base in line])
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
    """
    Based on Min, Seonwoo, Byunghan Lee, and Sungroh Yoon. "TargetNet: functional microRNA target prediction with deep neural networks." Bioinformatics 38.3 (2022): 671-677. https://doi.org/10.1093/bioinformatics/btab733.
    Python implementation: https://github.com/seonwoo-min/TargetNet

    Encodes extended seed alignemt of miRNA and gene sequences using one-hot encoding, where each nucleotide is represented by a vector of length 5.
    A = [1, 0, 0, 0, 0]
    C = [0, 1, 0, 0, 0]
    G = [0, 0, 1, 0, 0]
    U = [0, 0, 0, 1, 0]
    T = [0, 0, 0, 1, 0]
    - = [0, 0, 0, 0, 1]
    First, gene sequence is truncated to 40nt nad miRNA sequence is reversed to be in 5' to 3' direction.
    Then, miRNA extended seed (nucleotides 1-10) and gene (nucleotides 6-15) sequences are aligned using global alignment.
    The scoring matrix for the alignment is defined to produce a score of 1 for WC and wobble pairings, and a score of 0 for the other pairings and gaps
    The aligned sequences are encoded using one-hot encoding, producing a 2D matrix with shape (10, 50), where the first 5 rows represent gene and the last 5 rows represent miRNA.
    First 5 columns represent first 5 nucletides from gene, followed by one-hot-encoding of extended seed alignment, and one-hot-encoding of position-wise concatenation of rest of the gene and miRNA sequences.
    Matrix is padded with zeros to have shape (10, 50).
    Returns PyTorch DataLoader with defined batch_size.
    """
    
    def __init__(self):
        self.score_matrix = self.get_score_matrix()

    def __call__(self, df, miRNA_col="noncodingRNA", gene_col="gene", batch_size=64, shuffle=False):
        return self.prepare_dataset(df, miRNA_col, gene_col, batch_size, shuffle)

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

    def encode_RNA(self, mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa, with_esa):
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

    def prepare_dataset(self, df, miRNA_col, gene_col, batch_size=64, shuffle=False, with_esa = True):

        X = []

        for _, row in df.iterrows():

            mirna_seq = reverse(row[miRNA_col].upper().replace("T", "U"))
            mrna_seq = row[gene_col].upper().replace("T", "U")[0:40]

            # align miRNA seed region with mRNA
            mirna_esa, cts_rev_esa, _ = self.extended_seed_alignment(mirna_seq, mrna_seq)
            # encode aligned miRNA-CTS pair
            X.append(torch.from_numpy(self.encode_RNA(mirna_seq, mirna_esa,
                                                    mrna_seq, cts_rev_esa, with_esa)))

        dataset = TargetNetEncoder.miRNA_CTS_dataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
    """
    Based on McGeary, Sean E., et al. "The biochemical basis of microRNA targeting efficacy." Science 366.6472 (2019): eaav1741. https://doi.org/10.1126/science.aav1741.
    Python implementation: https://github.com/kslin/miRNA_models/tree/master

    Encodes miRNA and gene sequences as outer product of one-hot-encoded first 10 nucleotides of miRNA and 12 nucleotides of gene.
    A = [1, 0, 0, 0]
    U = [0, 1, 0, 0]
    T = [0, 1, 0, 0]
    C = [0, 0, 1, 0]
    G = [0, 0, 0, 1]
    X = [0.25, 0.25, 0.25, 0.25]
    First 10 nucleotides of miRNA are one-hot-encoded and flattened to 40-element vector.
    Then, for each 12-nucleotide substring of gene, outer product of miRNA and gene one-hot-encoded sequences is calculated.
    These 40x48 are stacked for each 12-nucleotide substring of gene.
    Returns numpy array with shape (num_of_samples, number_of_12_substrings_in_gene, 40, 48).
    """
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
            mirseq_one_hot = self.one_hot_encode(df.iloc[seq_id][miRNA_column][:10])
            for sub_seq in [df.iloc[seq_id][gene_column][i:i+12] for i in range(len(df.iloc[seq_id][gene_column])-11)]:
                seq_one_hot = self.one_hot_encode(sub_seq)      
                input_data.append(np.outer(mirseq_one_hot, seq_one_hot))

            input_data = np.stack(input_data)
            data.append(input_data)

        return data

class YangAttentionEncoder():
    """
    Based on Yang, Tzu-Hsien, et al. "Identifying Human miRNA Target Sites via Learning the Interaction Patterns between miRNA and mRNA Segments." Journal of Chemical Information and Modeling (2023). https://doi.org/10.1021/acs.jcim.3c01150.
    Python implementation: http://cosbi2.ee.ncku.edu.tw/mirna_binding/download

    Encodes first 30 nucleotides of miRNA padded with "L" to the length of 30 and first 40 nucleotides of gene as one-hot-encoded sequences.
    A = [1, 0, 0, 0]
    U = [0, 1, 0, 0]
    T = [0, 1, 0, 0]
    C = [0, 0, 1, 0]
    G = [0, 0, 0, 1]
    L = [0, 0, 0, 0]
    Returns PyTorch DataLoader with defined batch_size.
    """
    def __call__(self, df, miRNA_col="noncodingRNA", gene_col="gene", device="cpu", batch_size=256, shuffle=False):
        return self.data_preprocessing(df, miRNA_col, gene_col, device, batch_size, shuffle)

    def data_preprocessing(self, df, miRNA_col, gene_col, device, batch_size, shuffle, drop_last=False):

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
    """
    Based on McGeary, Sean E., et al. "MicroRNA 3′-compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position." Elife 11 (2022): e69803. https://doi.org/10.7554/eLife.69803.
    Python implementation: None, made from scratch

    Encodes miRNA and gene sequences as an array of [miRNA, reverse_complement(gene)] pairs.
    Returns array of [miRNA, reverse_complement(gene)] pairs.
    """
    def __init__(self, miRNA_col="noncodingRNA", gene_col="gene"):
        self.miRNA_col = miRNA_col
        self.gene_col = gene_col

    def __call__(self, df):
        # return array of [miRNA, reverse_complement(gene)] pairs
        return df.apply(lambda x: [x[self.miRNA_col], reverse_complement(x[self.gene_col])], axis=1)

def reverse_complement(st):
    nn = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(nn[n] for n in reversed(st))    

def reverse(seq):
    return seq[::-1]
