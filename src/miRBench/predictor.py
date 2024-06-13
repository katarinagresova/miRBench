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

def get_predictor(predictor_name):
    if predictor_name == "cnnMirTarget":
        return cnnMirTarget()
    elif predictor_name == "RNAcofold":
        return RNAcofold()
    elif predictor_name == "HejretMirnaCnn":
        return HejretMirnaCnn()
    elif predictor_name == "miRBind":
        return miRBind()
    elif predictor_name == "TargetNet":
        return TargetNet()
    elif predictor_name == "Seed8mer":
        return Seed8mer()
    elif predictor_name == "Seed7mer":
        return Seed7mer()
    elif predictor_name == "Seed6mer":
        return Seed6mer()
    elif predictor_name == "Seed6merBulgeOrMismatch":
        return Seed6merBulgeOrMismatch()
    elif predictor_name == "TargetScanCnn":
        return TargetScanCnn()
    else:
        raise ValueError("Unknown predictor name")

class Predictor():
    def __call__(self, data):
        return self.predict(data)
    
    def predict(self, data):
        raise NotImplementedError()

class cnnMirTarget(Predictor):
    def __init__(self):
        self.model = get_model("cnnMirTarget")

    def predict(self, data):
        return self.model.predict(data)

class RNAcofold(Predictor):
    def predict(self, data):
        return [-1 * RNA.cofold(seq)[1] for seq in data]
    
class HejretMirnaCnn(Predictor):
    def __init__(self):
        self.model = get_model("HejretMirnaCnn")

    def predict(self, data):
        return self.model.predict(data)
    
class miRBind(Predictor):
    def __init__(self):
        self.model = get_model("miRBind")

    def predict(self, data):
        return self.model.predict(data)[:, 1]
    
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
    def __init__(self, device = "cpu"):
        self.model = self.prepare_model()
        self.device = device

    def predict(self, data):
        predictions = []
        self.model.eval()
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

        model_path = Path(CACHE_PATH / TargetNet / "model.pt")
        if not model_path.exists():
            download_model("TargetNet", model_path)

        model_cfg = TargetNet.ModelConfig({
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
        model.to(self.device)

        return model

class SeedPredictor(Predictor):
    def predict(self, data):
        preds = []
        for sample in data:
            seeds = self.get_seed(sample[0])
            preds.append(1 if any([seq in sample[1] for seq in seeds]) else 0)

    def get_seed(self, miRNA):
        raise NotImplementedError()

class Seed8mer(SeedPredictor):               
    def get_seed(self, miRNA):
        return [
            'A' + miRNA[1:8] # 8mer - full complementarity on positions 2-8 and A on the position 1
        ]
    
class Seed7mer(SeedPredictor):
    def get_seed(self, miRNA):
        return [
            miRNA[1:8], # 7mer-m8 - full complementarity on positions 2-8
            'A' + miRNA[1:7] # 7mer-A1 - full complementarity on positions 2-7 and A on the position 1
        ]
    
class Seed6mer(SeedPredictor):
    def get_seed(self, miRNA):
        return [
            miRNA[1:7], # 6mer - full complementarity on positions 2-7
            miRNA[2:8], # 6mer-m8 - full complementarity on positions 3-8
            'A' + miRNA[1:6] # 6mer-A1 - full complementarity on positions 2-6 and A on the position 1
        ]
    
class Seed6merBulgeOrMismatch(SeedPredictor):
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
    def __init__(self):
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

        model_path = Path(CACHE_PATH / "TargetScanCnn")
        if not model_path.exists():
            model_path.mkdir(parents=True)

            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.data-00000-of-00001'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.data-00000-of-00001'))
            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.index'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.index'))
            url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetScan_CNN/model-100.meta'
            urllib.request.urlretrieve(url, Path(model_path / 'model-100.meta'))

        return model_path

class YangAttention(Predictor):
    def __init__(self, device = "cpu", miRNA_MAXLEN = 30, mRNA_MAXLEN = 40):
        
        self.miRNA_MAXLEN = miRNA_MAXLEN
        self.mRNA_MAXLEN = mRNA_MAXLEN
        self.RNA = list('ATCG')
        self.MODEL_TYPE = 'attention'
        
        self.model = self.prepare_model("YangAttention", device)

    def predict(self, data):
        self.model.eval()
        preds = []
        for (batch, ) in data:

            output = self.model(batch).detach()
            output = F.softmax(output, dim=-1)[:, 1].cpu().numpy()
            preds.extend(output)
            
        preds = np.array(preds) 
        return preds
    
    def prepare_model(self, model_name, device):
        model_path = Path(CACHE_PATH / model_name / "model.pkl")
        if not model_path.exists():
            download_model(model_name, model_path)

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
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model
    
    class FELayer(nn.Module):
        def __init__(self, layer_infos, last_norm=True, norm_type='batch', bias=True):
            super(YangAttention.FELayer, self).__init__()
            
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
            super(YangAttention.SEblock, self).__init__()
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
            super(YangAttention.SELayer, self).__init__()
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=(kernel_size//2))
            
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.PReLU()
            self.se = YangAttention.SEblock(channels=out_channels, reduction=reduction)#ChannelAttentionBlock()#
            
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
            super(YangAttention.BaseLine7, self).__init__()
            
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
            self.mrna_conv = YangAttention.SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
            self.mrna_dropout = nn.Dropout(cnn_dropout)
            self.pirna_conv = YangAttention.SELayer(emb_dim, cnn_dim, kernel, stride=1, reduction=se_reduction)
            self.pirna_dropout = nn.Dropout(cnn_dropout)
            
            # Transformer Decoder
            self.d_model = cnn_dim
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
            self.cls_layer = YangAttention.FELayer([
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

def get_model(model_name, force_download = False):

    local_path = Path(CACHE_PATH / model_name / MODEL_FILE_NAME)
    if not local_path.exists() or force_download:
        download_model(model_name, local_path)

    model = k.models.load_model(local_path)
    
    return model

def download_model(model_name, local_path):

    if model_name == "cnnMirTarget":
        url = 'https://github.com/katarinagresova/miRBench/raw/main/models/cnnMirTarget/cnn_model_preTrained.h5'
    elif model_name == "miRBind":
        url = 'https://github.com/katarinagresova/miRBench/raw/main/models/miRBind/miRBind.h5'
    elif model_name == "HejretMirnaCnn":
        url = 'https://github.com/katarinagresova/miRBench/raw/main/models/Hejret_miRNA_CNN/model_miRNA.h5'
    elif model_name == "TargetNet":
        url = 'https://github.com/katarinagresova/miRBench/raw/main/models/TargetNet/TargetNet.pt'
    elif model_name == "YangAttention":
        url = 'https://github.com/katarinagresova/miRBench/raw/main/models/Yang_Attention/attention_model.pkl'
    else:
        raise ValueError("Unknown model name")

    model_dir_path = Path(CACHE_PATH / model_name)
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    urllib.request.urlretrieve(url, local_path)

        