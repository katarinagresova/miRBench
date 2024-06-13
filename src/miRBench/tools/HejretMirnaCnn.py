import os
import urllib.request 
import pandas as pd
from tensorflow import keras as k

from miRBench.utils import parse_args, one_hot_encoding, get_model_path

def predict_probs(df, miRNA_col, gene_col, model):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'noncodingRNA' columns
    :param model: Keras model used for predicting
    """

    ohe = one_hot_encoding(df, miRNA_col, gene_col)
    return model.predict(ohe)

if __name__ == '__main__':
    args = parse_args('Hejret miRNA CNN')

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    model_path = get_model_path(
        folder = 'Hejret_miRNA_CNN', 
        model_name = 'model_miRNA.h5', 
        url = 'https://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/Hejret_miRNA_CNN/model_miRNA.h5'
    )
    model = k.models.load_model(model_path)

    preds = predict_probs(data, args.miRNA_column, args.gene_column, model)
    data['HejretMirnaCnn'] = preds
    data.to_csv(args.output, sep='\t', index=False)