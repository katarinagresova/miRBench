import os
import urllib.request 
import pandas as pd
from tensorflow import keras as k
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

from miRNAbenchmarks.utils import parse_args, one_hot_encoding, get_model_path

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
    

def predict_probs(df, miRNA_col, gene_col, model):
    """
    fun predicts the probability of miRNA:target site binding in df file
    :param df: input dataframe with sequences containing 'gene' and 'miRNA' columns
    :param model: Keras model used for predicting
    """

    ohe = one_hot_encoding(df, miRNA_col, gene_col)
    return model.predict(ohe)[:, 1]

if __name__ == '__main__':
    
    args = parse_args('miRBind')

    # Read the input file
    data = pd.read_csv(args.input, sep='\t')

    model_path = get_model_path(
        folder = 'miRBind', 
        model_name = 'miRBind.h5', 
        url = 'hhttps://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/miRBind/miRBind.h5'
    )
    model = k.models.load_model(model_path)

    preds = predict_probs(data, args.miRNA_column, args.gene_column, model)

    data['miRBind'] = preds

    data.to_csv(args.output, sep='\t', index=False)