import tensorflow as tf
import numpy as np
import h5py
import os
import time
from keras.preprocessing.sequence import pad_sequences
s = time.time()

tf.reset_default_graph()
datadir = 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data'


def get_embedding_and_label(emb_file, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    return embedding


def reshape_input_embeddings(input_embed, n_sent1, first_reshaping):
    tf.reset_default_graph()
    global reshaped_1, reshaped_2, reshaped_3, fin_reshape
    x = tf.placeholder(tf.float32, [3, None, 1024])
    x_ = tf.placeholder(tf.float32, [3, None])
    n_sent = np.array(input_embed).shape[1]
    new_shape = n_sent * 1024

    if first_reshaping:
        reshaped_1_layer = tf.reshape(input_embed[0], [new_shape])
        reshaped_2_layer = tf.reshape(input_embed[1], [new_shape])
        reshaped_3_layer = tf.reshape(input_embed[2], [new_shape])
        with tf.Session() as sess:
            res = sess.run([reshaped_1_layer, reshaped_2_layer, reshaped_3_layer], feed_dict={x: input_embed})
            res_ = [list(res[i]) for i in range(3)]
    else:
        final_reshape = tf.reshape(input_embed, [3, n_sent1, 1024])
        with tf.Session() as sess:
            res_ = sess.run([final_reshape], feed_dict={x_: input_embed})
    return res_, n_sent


def sum_all_layers(reshaped_embed, n_sent):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [3, None, 1024])
    sum_layers = tf.reduce_sum(x, 0)
    # reshaped_embed = tf.reshape(reshaped_embed, [-1, n_sent, 1024])
    with tf.Session() as sess:
        res = sess.run([sum_layers], feed_dict={x: reshaped_embed})
    return res


def model(input_embed):

    reshaped_, n_sent_ = reshape_input_embeddings(input_embed, ' ', first_reshaping=True)
    pad_embedding = pad_sequences(reshaped_, maxlen=1024 * n_sent_ + 5, dtype='float32', padding='post')
    # print(pad_embedding, np.array(pad_embedding).shape)
    pad_embedding = pad_sequences(pad_embedding, maxlen=1024 * n_sent_, dtype='float32', padding='post')
    # print(pad_embedding, np.array(pad_embedding).shape)
    reshaped_embedding = reshape_input_embeddings(pad_embedding, n_sent_, first_reshaping=False)[0]
    sum = sum_all_layers(reshaped_embedding[0], n_sent_)
    # print(sum[0], np.array(sum[0]).shape)
    return sum, np.array(sum[0].shape)


def run_for_file(data, i, part_num):
    new_embeddings_file = os.path.join(datadir, 'reshaped_data/not_fixed_new/{}_{}_review_embeddings_nf.hdf5'.format(data, i))
    embeddings_file = os.path.join(datadir, '{}_{}_review_embeddings.hdf5'.format(data, i))
    with h5py.File(new_embeddings_file, 'w') as fout:
        for idx in range(part_num):
            review_embed = get_embedding_and_label(embeddings_file, idx)
            embed, shape = model(review_embed)
            ds = fout.create_dataset(
                '{}'.format(idx),
                np.array(embed).shape[:], dtype='float32',
                data=embed[:]
            )


for i in range(4):
    run_for_file('train', str(i+1), part_num=5000)

for i in range(5):
    run_for_file('test', str(i + 1), part_num=5000)

run_for_file('train_left', str(5), part_num=2500)
run_for_file('valid', str(0), part_num=2500)


fin = time.time()
print('\n', 'time: %s' % str(fin - s))
