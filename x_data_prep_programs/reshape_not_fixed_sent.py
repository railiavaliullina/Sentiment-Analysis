import tensorflow as tf
import numpy as np
import h5py
import os
import time
s = time.time()

tf.reset_default_graph()
datadir = 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data'


def get_embedding_and_label(emb_file, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    return embedding


def sum_all_layers(reshaped_embed):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [3, None, 1024])
    sum_layers = tf.reduce_sum(x, 0)
    with tf.Session() as sess:
        res = sess.run([sum_layers], feed_dict={x: reshaped_embed})
    return res


def model(input_embed):
    sum = sum_all_layers(input_embed)
    return sum, np.array(sum[0].shape)


def run_for_file(data, i, part_num):
    new_embeddings_file = os.path.join(datadir, 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data/'
                                                'reshaped_data/not_fixed_sent/{}_{}_review_embeddings_not_fixed.hdf5'.format(data, i))
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
