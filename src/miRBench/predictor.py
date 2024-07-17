import RNA
from pathlib import Path
import urllib.request
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import OrderedDict

CACHE_PATH = Path.home() / ".miRBench" / "models"
MODEL_FILE_NAME = "model.h5"

def list_predictors():
    return ["CnnMirTarget_Zheng2020", 
           "RNACofold", 
           "miRNA_CNN_Hejret2023", 
           "miRBind_Klimentova2022", 
           "TargetNet_Min2021", 
           "Seed8mer", "Seed7mer", "Seed6mer", "Seed6merBulgeOrMismatch", 
           "TargetScanCnn_McGeary2019", 
           "InteractionAwareModel_Yang2024"]

def get_predictor(predictor_name):
    if predictor_name == "CnnMirTarget_Zheng2020":
        return CnnMirTarget()
    elif predictor_name == "RNACofold":
        return RNACofold()
    elif predictor_name == "miRNA_CNN_Hejret2023":
        return HejretMirnaCnn()
    elif predictor_name == "miRBind_Klimentova2022":
        return miRBind()
    elif predictor_name == "TargetNet_Min2021":
        return TargetNet()
    elif predictor_name == "Seed8mer":
        return Seed8mer()
    elif predictor_name == "Seed7mer":
        return Seed7mer()
    elif predictor_name == "Seed6mer":
        return Seed6mer()
    elif predictor_name == "Seed6merBulgeOrMismatch":
        return Seed6merBulgeOrMismatch()
    elif predictor_name == "TargetScanCnn_McGeary2019":
        return TargetScanCnn()
    elif predictor_name == "InteractionAwareModel_Yang2024":
        return InteractionAwareModel()
    else:
        raise ValueError(f"Unknown predictor name: {predictor_name}")

class Predictor():
    def __call__(self, data, **kwargs):
        return self.predict(data, **kwargs)
    
    def predict(self, data, **kwargs):
        raise NotImplementedError()
    
    def get_predictor_name(self):
        raise NotImplementedError()

class CnnMirTarget(Predictor):
    """
    Based on Zheng, Xueming, et al. "Prediction of miRNA targets by learning from interaction sequences." Plos one 15.5 (2020): e0232578. https://doi.org/10.1371%2Fjournal.pone.0232578
    Python implementation: https://github.com/zhengxueming/cnnMirTarget

    Predicts the probability of a miRNA-mRNA interaction.
    Returns a list of probabilities.
    """
    def __init__(self):
        self.predictor_name = "CnnMirTarget_Zheng2020"
        self.model_url = 'https://github.com/katarinagresova/miRBench/raw/main/models/cnnMirTarget/cnn_model_preTrained.h5'
        self.model = get_model(self.predictor_name, self.model_url)

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

class RNACofold(Predictor):
    """
    Based on Lorenz, Ronny, et al. "ViennaRNA Package 2.0." Algorithms for molecular biology 6 (2011): 1-14. https://doi.org/10.1186/1748-7188-6-26.
    Python implementation: https://pypi.org/project/ViennaRNA/

    Predicts the minimum free energy of a miRNA-mRNA sequence.
    Returns a list of negative minimum free energies.
    """
    def predict(self, data):
        return [-1 * RNA.cofold(seq)[1] for seq in data]
    
class HejretMirnaCnn(Predictor):
    """
    Based on Hejret, Vaclav, et al. "Analysis of chimeric reads characterises the diverse targetome of AGO2-mediated regulation." Scientific Reports 13.1 (2023): 22895. https://doi.org/10.1038/s41598-023-49757-z.
    Python implementation: https://github.com/ML-Bioinfo-CEITEC/HybriDetector/tree/main

    Predicts the probability of a miRNA-mRNA interaction.
    Returns a list of probabilities.
    """
    def __init__(self):
        self.predictor_name = "miRNA_CNN_Hejret2023"
        self.model_url = 'https://github.com/katarinagresova/miRBench/raw/main/models/Hejret_miRNA_CNN/model_miRNA.h5'
        self.model = get_model(self.predictor_name, self.model_url)

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)
    
class miRBind(Predictor):
    """
    Based on Klimentová, Eva, et al. "miRBind: A deep learning method for miRNA binding classification." Genes 13.12 (2022): 2323. https://doi.org/10.3390/genes13122323.
    Python implementation: https://github.com/ML-Bioinfo-CEITEC/miRBind

    Predicts the probability of a miRNA-mRNA interaction.
    Returns a list of probabilities.
    """
    def __init__(self):
        self.predictor_name = "miRBind_Klimentova2022"
        self.model_url = 'https://github.com/katarinagresova/miRBench/raw/main/models/miRBind/miRBind.h5'
        self.model = get_model(self.predictor_name, self.model_url)

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)[:, 1]
    
    @register_keras_serializable()
    class ResBlock(layers.Layer):
        """
        Defines a Residual block. For more information refer to the original paper at https://arxiv.org/abs/1512.03385 .
        """

        def __init__(self, downsample=False, filters=16, kernel_size=3):

            super(miRBind.ResBlock, self).__init__()

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
    
class TargetNet(Predictor):
    """
    Based on Min, Seonwoo, Byunghan Lee, and Sungroh Yoon. "TargetNet: functional microRNA target prediction with deep neural networks." Bioinformatics 38.3 (2022): 671-677. https://doi.org/10.1093/bioinformatics/btab733.
    Python implementation: https://github.com/seonwoo-min/TargetNet

    Predicts the probability of a miRNA-mRNA interaction.
    Returns a list of probabilities, conatenated for all batches.
    """
    def __init__(self):
        self.predictor_name = "TargetNet_Min2021"
        self.model_url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetNet/TargetNet.pt'
        self.model = self.prepare_model()

    def predict(self, data, device = "cpu"):
        predictions = []
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for X in data:
                preds = self.model(X)
                preds = torch.sigmoid(preds).cpu().detach().numpy()
                predictions.extend(preds[:,0])

        return predictions
    
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

    def prepare_model(self):

        model_path = Path(CACHE_PATH / self.predictor_name / "model.pt")
        if not model_path.exists():
            download_model(self.predictor_name, self.model_url, model_path)

        model_cfg = TargetNet.ModelConfig({
            'skip_connection': True,
            'num_channels': [16, 16, 32],
            'num_blocks': [2, 1, 1],
            'stem_kernel_size': 5,
            'block_kernel_size': 3,
            'pool_size': 3
        })

        model, _ = self.get_model(model_cfg, with_esa=True)
        checkpoint = torch.load(model_path, map_location="cpu")

        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v

        model.load_state_dict(state_dict)

        return model

    def get_model(self, model_cfg, with_esa, dropout_rate=None):
        """ get model considering model types """
        model = self.TargetNetModel(self, model_cfg, with_esa, dropout_rate)
        params = self.get_params_and_initialize(model)

        return model, params


    def get_params_and_initialize(self, model):
        """
        parameter initialization
        get weights and biases for different weighty decay during training
        """
        params_with_decay, params_without_decay = [], []
        for name, param in model.named_parameters():
            if "weight" in name:
                if "bn" not in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                    params_with_decay.append(param)
                else:
                    nn.init.ones_(param)
                    params_without_decay.append(param)

            else:
                nn.init.zeros_(param)
                params_without_decay.append(param)

        return params_with_decay, params_without_decay
    
    class TargetNetModel(nn.Module):
        """ TargetNet for microRNA target prediction """
        def __init__(self, target_net, model_cfg, with_esa, dropout_rate):
            super(TargetNet.TargetNetModel, self).__init__()
            self.target_net = target_net
            num_channels = model_cfg.num_channels
            num_blocks = model_cfg.num_blocks

            if not with_esa: self.in_channels, in_length = 8, 40
            else:            self.in_channels, in_length = 10, 50
            out_length = np.floor(((in_length - model_cfg.pool_size) / model_cfg.pool_size) + 1)

            self.stem = self._make_layer(model_cfg, num_channels[0], num_blocks[0], dropout_rate, stem=True)
            self.stage1 = self._make_layer(model_cfg, num_channels[1], num_blocks[1], dropout_rate)
            self.stage2 = self._make_layer(model_cfg, num_channels[2], num_blocks[2], dropout_rate)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
            self.max_pool = nn.MaxPool1d(model_cfg.pool_size)
            self.linear = nn.Linear(int(num_channels[-1] * out_length), 1)

        def _make_layer(self, cfg, out_channels, num_blocks, dropout_rate, stem=False):
            layers = []
            for b in range(num_blocks):
                if stem: layers.append(self.target_net.Conv_Layer(self.in_channels, out_channels, cfg.stem_kernel_size, dropout_rate,
                                                post_activation= b < num_blocks - 1))
                else:    layers.append(self.target_net.ResNet_Block(self.in_channels, out_channels, cfg.block_kernel_size, dropout_rate,
                                                    skip_connection=cfg.skip_connection))
                self.in_channels = out_channels

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.dropout(self.relu(x))
            x = self.max_pool(x)
            x = x.reshape(len(x), -1)
            x = self.linear(x)

            return x


    def conv_kx1(in_channels, out_channels, kernel_size, stride=1):
        """ kx1 convolution with padding without bias """
        layers = []
        padding = kernel_size - 1
        padding_left = padding // 2
        padding_right = padding - padding_left
        layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False))
        return nn.Sequential(*layers)


    class Conv_Layer(nn.Module):
        """
        CNN layer with/without activation
        -- Conv_kx1_ReLU-Dropout
        """
        def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, post_activation):
            super(TargetNet.Conv_Layer, self).__init__()
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
            self.conv = TargetNet.conv_kx1(in_channels, out_channels, kernel_size)
            self.post_activation = post_activation

        def forward(self, x):
            out = self.conv(x)
            if self.post_activation:
                out = self.dropout(self.relu(out))

            return out


    class ResNet_Block(nn.Module):
        """
        ResNet Block
        -- ReLU-Dropout-Conv_kx1 - ReLU-Dropout-Conv_kx1
        """
        def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, skip_connection):
            super(TargetNet.ResNet_Block, self).__init__()
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not None else 0)
            self.conv1 = TargetNet.conv_kx1(in_channels, out_channels, kernel_size)
            self.conv2 = TargetNet.conv_kx1(out_channels, out_channels, kernel_size)
            self.skip_connection = skip_connection

        def forward(self, x):
            out = self.dropout(self.relu(x))
            out = self.conv1(out)

            out = self.dropout(self.relu(out))
            out = self.conv2(out)

            if self.skip_connection:
                out_c, x_c = out.shape[1], x.shape[1]
                if out_c == x_c: out += x
                else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

            return out

class SeedPredictor(Predictor):
    def predict(self, data):
        preds = []
        for sample in data:
            seeds = self.get_seed(sample[0])
            preds.append(1 if any([seq in sample[1] for seq in seeds]) else 0)

        return preds

    def get_seed(self, miRNA):
        raise NotImplementedError()

class Seed8mer(SeedPredictor):
    """
    Based on McGeary, Sean E., et al. "MicroRNA 3′-compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position." Elife 11 (2022): e69803. https://doi.org/10.7554/eLife.69803.
    Python implementation: None, made from scratch

    Predicts the occurrence of 8mer seed match.
    Returns a list of 0s and 1s. 0s indicate no 8mer seed match, 1s indicate 8mer seed match.
    """              
    def get_seed(self, miRNA):
        return [
            'A' + miRNA[1:8] # 8mer - full complementarity on positions 2-8 and A on the position 1
        ]
    
class Seed7mer(SeedPredictor):
    """
    Based on McGeary, Sean E., et al. "MicroRNA 3′-compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position." Elife 11 (2022): e69803. https://doi.org/10.7554/eLife.69803.
    Python implementation: None, made from scratch

    Predicts the occurrence of 7mer seed match.
    Returns a list of 0s and 1s. 0s indicate no 7mer seed match, 1s indicate 7mer seed match.
    """
    def get_seed(self, miRNA):
        return [
            miRNA[1:8], # 7mer-m8 - full complementarity on positions 2-8
            'A' + miRNA[1:7] # 7mer-A1 - full complementarity on positions 2-7 and A on the position 1
        ]
    
class Seed6mer(SeedPredictor):
    """
    Based on McGeary, Sean E., et al. "MicroRNA 3′-compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position." Elife 11 (2022): e69803. https://doi.org/10.7554/eLife.69803.
    Python implementation: None, made from scratch

    Predicts the occurrence of 6mer seed match.
    Returns a list of 0s and 1s. 0s indicate no 6mer seed match, 1s indicate 6mer seed match.
    """
    def get_seed(self, miRNA):
        return [
            miRNA[1:7], # 6mer - full complementarity on positions 2-7
            miRNA[2:8], # 6mer-m8 - full complementarity on positions 3-8
            'A' + miRNA[1:6] # 6mer-A1 - full complementarity on positions 2-6 and A on the position 1
        ]
    
class Seed6merBulgeOrMismatch(SeedPredictor):
    """
    Based on McGeary, Sean E., et al. "MicroRNA 3′-compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position." Elife 11 (2022): e69803. https://doi.org/10.7554/eLife.69803.
    Python implementation: None, made from scratch

    Predicts the occurrence of 6mer seed match with bulges or mismatches.
    Returns a list of 0s and 1s. 0s indicate no 6mer seed match with bulges or mismatches, 1s indicate 6mer seed match with bulges or mismatches.
    """
    def get_seed(self, miRNA):
        mers = []
        mers.append(miRNA[1:7])
        for pos in range(1, 7):
            for nt in ['A', 'C', 'G', 'T']:
                # bulges
                mers.append(
                    miRNA[1:7][:pos] + nt + miRNA[1:7][pos:]
                )
                # mismatches
                mers.append(
                    miRNA[1:7][:pos] + nt + miRNA[1:7][pos+1:]
                )
        mers.append(miRNA[2:8])
        for pos in range(2, 8):
            for nt in ['A', 'C', 'G', 'T']:
                mers.append(
                    miRNA[2:8][:pos] + nt + miRNA[2:8][pos:]
                )
                mers.append(
                    miRNA[2:8][:pos] + nt + miRNA[2:8][pos+1:]
                )
        mers.append('A' + miRNA[1:6])
        for pos in range(1, 6):
            for nt in ['A', 'C', 'G', 'T']:
                mers.append(
                    'A' + miRNA[1:6][:pos] + nt + miRNA[1:6][pos:]
                )
                mers.append(
                    'A' + miRNA[1:6][:pos] + nt + miRNA[1:6][pos+1:]
                )

        return list(set(mers))

class TargetScanCnn(Predictor):
    """
    Based on McGeary, Sean E., et al. "The biochemical basis of microRNA targeting efficacy." Science 366.6472 (2019): eaav1741. https://doi.org/10.1126/science.aav1741.
    Python implementation: https://github.com/kslin/miRNA_models/tree/master

    Predicts relative KD values of miRNA-mRNA interactions.
    Returns a list of relative KD values.
    """
    def __init__(self):
        self.predictor_name = "TargetScanCnn_McGeary2019"
        self.model_path = self.prepare_model()

    def predict(self, data):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.import_meta_graph(self.model_path / 'model-100.meta')
            print('Restoring from {}'.format(self.model_path / 'model-100'))
            saver.restore(sess, str(self.model_path) + '/model-100')

            dropout_rate = tf.compat.v1.get_default_graph().get_tensor_by_name('dropout_rate:0')
            phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name('phase_train:0')
            combined_x = tf.compat.v1.get_default_graph().get_tensor_by_name('combined_x:0')
            prediction = tf.compat.v1.get_default_graph().get_tensor_by_name('final_layer/pred_ka:0')

            preds = []
            for input_data in data:
                feed_dict = {
                                dropout_rate: 0.0,
                                phase_train: False,
                                combined_x: input_data
                            }

                pred_kds = sess.run(prediction, feed_dict=feed_dict).flatten()
                preds.append(max(pred_kds))

        return preds

    def prepare_model(self):

        model_path = Path(CACHE_PATH / self.predictor_name)
        if not model_path.exists():
            model_path.mkdir(parents=True)

            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.data-00000-of-00001'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.data-00000-of-00001'))
            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.index'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.index'))
            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.meta'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.meta'))

        return model_path

class InteractionAwareModel(Predictor):
    """
    Based on Yang, Tzu-Hsien, et al. "Identifying Human miRNA Target Sites via Learning the Interaction Patterns between miRNA and mRNA Segments." Journal of Chemical Information and Modeling (2023). https://doi.org/10.1021/acs.jcim.3c01150.
    Python implementation: http://cosbi2.ee.ncku.edu.tw/mirna_binding/download

    Predicts the probability of a miRNA-mRNA interaction.
    Returns a list of probabilities.
    """
    def __init__(self):

        self.predictor_name = "InteractionAwareModel_Yang2024"
        self.model_url = 'https://github.com/katarinagresova/miRBench/raw/main/models/Yang_Attention/attention_model.pkl'
        
        self.miRNA_MAXLEN = 30
        self.mRNA_MAXLEN = 40
        self.RNA = list('ATCG')
        self.MODEL_TYPE = 'attention'
        
        self.model = self.prepare_model()

    def predict(self, data, device = "cpu"):
        self.model.to(device)
        self.model.eval()
        preds = []
        for (batch, ) in data:

            output = self.model(batch).detach()
            output = F.softmax(output, dim=-1)[:, 1].cpu().numpy()
            preds.extend(output)
            
        preds = np.array(preds) 
        return preds
    
    def prepare_model(self):
        model_path = Path(CACHE_PATH / self.predictor_name / "model.pkl")
        if not model_path.exists():
            download_model(self.predictor_name, self.model_url, model_path)

        model = self.BaseLine7(
                emb_dim=len(self.RNA), 
                cnn_dim=256,
                kernel=5,
                se_reduction=4,
                cnn_dropout=0.5,
                mrna_len=self.mRNA_MAXLEN, 
                pirna_len=self.miRNA_MAXLEN, 
                nhead=16,
                transformer_dropout=0.5,
                cls_dropout=0.75
            )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        return model
    
    class FELayer(nn.Module):
        def __init__(self, layer_infos, last_norm=True, norm_type='batch', bias=True):
            super(InteractionAwareModel.FELayer, self).__init__()
            
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
            super(InteractionAwareModel.SEblock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(channels, channels//reduction)
            self.relu = nn.PReLU()
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
            super(InteractionAwareModel.SELayer, self).__init__()
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
            
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.PReLU()
            self.se = InteractionAwareModel.SEblock(channels=out_channels, reduction=reduction)#ChannelAttentionBlock()#
            
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
            super(InteractionAwareModel.BaseLine7, self).__init__()
            
            #super(BaseLine7, self).__init__()
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
            self.mrna_conv = InteractionAwareModel.SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
            self.mrna_dropout = nn.Dropout(cnn_dropout)
            self.pirna_conv = InteractionAwareModel.SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
            self.pirna_dropout = nn.Dropout(cnn_dropout)
            
            # Transformer Decoder
            self.d_model = cnn_dim
            self.multihead_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=transformer_dropout)
            
            # Feedforward model
            self.linear1 = nn.Linear(self.d_model, self.d_model*4)
            self.dropout = nn.Dropout(transformer_dropout)
            self.linear2 = nn.Linear(self.d_model*4, self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            self.norm3 = nn.LayerNorm(self.d_model)
            self.dropout2 = nn.Dropout(transformer_dropout)
            self.dropout3 = nn.Dropout(transformer_dropout)
            self.activation = nn.PReLU()#nn.GELU()#

            # Classification
            self.dropout4 = nn.Dropout(cls_dropout)
            self.cls_input_dim = self.d_model*mrna_len
            self.cls_layer = InteractionAwareModel.FELayer([
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
            out = out.reshape(out.shape[0], -1)
            out = self.cls_layer(out)
            
            return out

def get_model(model_name, model_url, force_download = False):

    local_path = Path(CACHE_PATH / model_name / MODEL_FILE_NAME)
    if not local_path.exists() or force_download:
        download_model(model_name, model_url, local_path)

    model = k.models.load_model(local_path)
    
    return model

def download_model(model_name, model_url, local_path):

    model_dir_path = Path(CACHE_PATH / model_name)
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    urllib.request.urlretrieve(model_url, local_path)

        