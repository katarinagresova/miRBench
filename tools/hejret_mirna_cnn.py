import os
import urllib.request 
import pandas as pd
from tensorflow import keras as k

from utils import parse_args, one_hot_encoding

def predict_probs(df, miRNA_col, gene_col, model):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'noncodingRNA' columns
    :param model: Keras model used for predicting
    """

    ohe = one_hot_encoding(df, miRNA_col, gene_col)
    return model.predict(ohe)

def get_model_path():
    current_path = os.path.realpath(__file__)
    model_dir_path = os.path.join(os.path.dirname(current_path), '../models/Hejret_miRNA_CNN')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    model_path = os.path.join(model_dir_path, 'model_miRNA.h5')
    if os.path.exists(model_path):
        return model_path

    print('Downloading the model...')
    url = 'https://github.com/ML-Bioinfo-CEITEC/HybriDetector/raw/main/ML/Models/model_miRNA.h5'
    urllib.request.urlretrieve(url, model_path)

    return model_path

if __name__ == '__main__':
    args = parse_args('Hejret miRNA CNN')

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    model_path = get_model_path()
    model = k.models.load_model(model_path)

    preds = predict_probs(data, args.miRNA_column, args.gene_column, model)
    data['hejret_mirna'] = preds
    data.to_csv(args.output, sep='\t', index=False)