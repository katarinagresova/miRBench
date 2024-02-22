import argparse
import numpy as np
import pandas as pd
from tensorflow import keras as k

from mirbind import one_hot_encoding

def predict_probs(df, miRNA_col, gene_col, model):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'noncodingRNA' columns
    :param model: Keras model used for predicting
    :param output: output file to write probabilities to
    """
    #gene_length = 50

    #orig_len = len(df)
    #df = df[df["gene"].str.len() == gene_length]
    #processed_len = len(df)

    #if orig_len != processed_len:
    #    print("Skipping " + str(orig_len - processed_len) + " pairs due to inappropriate target length.")

    ohe = one_hot_encoding(df, miRNA_col, gene_col)
    return model.predict(ohe)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hejret miRNA CNN prediction.')
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
    data['hejret_mirna'] = preds
    data.to_csv(args.output, sep='\t', index=False)