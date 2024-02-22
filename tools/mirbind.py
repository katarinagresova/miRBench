import argparse
import numpy as np
import pandas as pd
from tensorflow import keras as k
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class ResBlock(layers.Layer):
    """
    Defines a Residual block. For more information refer to the original paper at https://arxiv.org/abs/1512.03385 .
    """

    def __init__(self, downsample=False, filters=16, kernel_size=3):

        super(ResBlock, self).__init__()

        # store parameters
        self.downsample = downsample
        self.filters = filters
        self.kernel_size = kernel_size

        # initialize inner layers
        self.conv1 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=(1 if not self.downsample else 2),
                                   filters=self.filters,
                                   padding="same")
        self.activation1 = layers.ReLU()
        self.batch_norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(kernel_size=self.kernel_size,
                                   strides=1,
                                   filters=self.filters,
                                   padding="same")
        if self.downsample:
          self.conv3 = layers.Conv2D(kernel_size=1,
                                      strides=2,
                                      filters=self.filters,
                                      padding="same")

        self.activation2 = layers.ReLU()
        self.batch_norm2 = layers.BatchNormalization()

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)

        if self.downsample:
            inputs = self.conv3(inputs)

        x = layers.Add()([inputs, x])

        x = self.activation2(x)
        x = self.batch_norm2(x)

        return x

    def get_config(self):
        return {'filters': self.filters, 'downsample': self.downsample, 'kernel_size': self.kernel_size}
    

def one_hot_encoding(df, miRNA_col, gene_col, tensor_dim=(50, 20, 1)):
    """
    fun encodes miRNAs and mRNAs in df into binding matrices
    :param df: dataframe containing 'gene' and 'miRNA' columns
    :param tensor_dim: output shape of the matrix
    :return: numpy array of predictions
    """

    miRNA_length = 20
    gene_length = 50

    # alphabet for watson-crick interactions.
    alphabet = {"AT": 1., "TA": 1., "GC": 1., "CG": 1., "AU": 1., "UA": 1.}
    # create empty main 2d matrix array
    N = df.shape[0]  # number of samples in df
    shape_matrix_2d = (N, *tensor_dim)  # 2d matrix shape
    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(shape_matrix_2d, dtype="float32")

    # compile matrix with watson-crick interactions.
    for index, row in df.iterrows():
        for bind_index, bind_nt in enumerate(row[gene_col][:gene_length].upper()):
            for mirna_index, mirna_nt in enumerate(row[miRNA_col][:miRNA_length].upper()):
                base_pairs = bind_nt + mirna_nt
                ohe_matrix_2d[index, bind_index, mirna_index, 0] = alphabet.get(base_pairs, 0)

    return ohe_matrix_2d

def predict_probs(df, miRNA_col, gene_col, model):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'miRNA' columns
    :param model: Keras model used for predicting
    :param output: output file to write probabilities to
    """

    #orig_len = len(df)
    #mask = (df["miRNA"].str.len() == miRNA_length) & (df["gene"].str.len() == gene_length)
    #df = df[mask]
    #processed_len = len(df)

    #if orig_len != processed_len:
    #    print("Skipping " + str(orig_len - processed_len) + " pairs due to inappropriate length.")

    ohe = one_hot_encoding(df, miRNA_col, gene_col)
    return model.predict(ohe)[:, 1]

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='miRBind prediction.')
    # Path to the input file 
    parser.add_argument('--input', type=str, help='Path to the input file - miRNA and a gene sequence in a tab-separated format.')
    # Name of column containing miRNA sequences
    parser.add_argument('--miRNA_column', type=str, help='Name of the column containing miRNA sequences')
    # Name of column containing gene sequences
    parser.add_argument('--gene_column', type=str, help='Name of the column containing gene sequences')
    # Path to the output file
    parser.add_argument('--output', type=str, help='Path to the output file')
    # Path to the trained model
    parser.add_argument('--model', type=str, help='Path to the trained model')
    # Parse the arguments
    args = parser.parse_args()

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    model = k.models.load_model(args.model)

    preds = predict_probs(data, args.miRNA_column, args.gene_column, model)

    data['mirbind'] = preds

    data.to_csv(args.output, sep='\t', index=False)