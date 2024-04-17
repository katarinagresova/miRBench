import numpy as np
import pandas as pd
import tensorflow as tf
import os
import urllib.request

from miRNAbenchmarks.utils import parse_args

def one_hot_encode(seq):
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

def predict(load_model, df, miRNA_col, gene_col):
    # load trained model 
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.import_meta_graph(load_model + '.meta')
        print('Restoring from {}'.format(load_model))
        saver.restore(sess, load_model)

        _dropout_rate = tf.compat.v1.get_default_graph().get_tensor_by_name('dropout_rate:0')
        _phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name('phase_train:0')
        _combined_x = tf.compat.v1.get_default_graph().get_tensor_by_name('combined_x:0')
        _prediction = tf.compat.v1.get_default_graph().get_tensor_by_name('final_layer/pred_ka:0')

        #num_batches = 64
        #batch_size = 1

        #num_batches = len(seqs) // batch_size
        #print("Number of batches: ", num_batches)

        #results = []
        preds = []

        for seq_id in range(len(df)):
            #print("Processing {}/{}...".format((batch+1)*batch_size, 438784))
            #seqs = kmers[batch*batch_size: (batch+1) * batch_size]

            #batch_seqs = seqs[batch*batch_size: (batch+1) * batch_size]

            input_data = []
            for sub_seq in [df.iloc[seq_id][gene_col][i:i+12] for i in range(len(df.iloc[seq_id][gene_col])-11)]: # sliding window
                seq_one_hot = one_hot_encode(sub_seq)
                mirseq_one_hot = one_hot_encode(df.iloc[seq_id][miRNA_col][:10])
                input_data.append(np.outer(mirseq_one_hot, seq_one_hot))

            input_data = np.stack(input_data)

            feed_dict = {
                            _dropout_rate: 0.0,
                            _phase_train: False,
                            _combined_x: input_data
                        }

            #pred_kds = -1 * sess.run(_prediction, feed_dict=feed_dict).flatten()
            pred_kds = sess.run(_prediction, feed_dict=feed_dict).flatten()

            #results.append(pred_kds)
            preds.append(max(pred_kds))

    return preds

def get_model_path():
    current_path = os.path.realpath(__file__)
    model_dir_path = os.path.join(os.path.dirname(current_path), '../../../models/TargetScan_CNN')
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path, parent = True)

    model_path = os.path.join(model_dir_path, 'model-100')
    if os.path.exists(model_path + '.meta'):
        return model_path

    print('Downloading the model...')
    url = 'https://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/TargetScan_CNN/model-100.data-00000-of-00001'
    urllib.request.urlretrieve(url, model_path + '.data-00000-of-00001')
    url = 'https://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/TargetScan_CNN/model-100.index'
    urllib.request.urlretrieve(url, model_path + '.index')
    url = 'https://github.com/katarinagresova/miRNA_benchmarks/raw/main/models/TargetScan_CNN/model-100.meta'
    urllib.request.urlretrieve(url, model_path + '.meta')

    return model_path

if __name__ == '__main__':
    args = parse_args('TargetScan CNN')

    data = pd.read_csv(args.input, sep='\t')

    model_path = get_model_path()
    preds = predict(model_path, data, args.miRNA_column, args.gene_column)
    data['TargetScanCnn'] = preds
    data.to_csv(args.output, sep='\t', index=False)