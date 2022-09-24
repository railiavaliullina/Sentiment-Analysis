import tensorflow as tf
import numpy as np
import h5py
import os
import time
from keras.preprocessing.sequence import pad_sequences
s = time.time()

tf.reset_default_graph()
datadir = 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data'
max_len = 12


def get_embedding_and_label(emb_file, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    # with open(lab_file, 'r', encoding='utf-8') as f:
    #     labels = f.readlines()
    return embedding


# print(label, np.array(review_embed[0]).shape, review_embed, review_embed[0])
# print(review_embed[0], review_embed[1], review_embed[2], np.array(review_embed).shape)


def reshape_input_embeddings(input_embed, first_reshaping):
    tf.reset_default_graph()
    global reshaped_1, reshaped_2, reshaped_3, fin_reshape
    x = tf.placeholder(tf.float32, [3, None, 1024])
    x_ = tf.placeholder(tf.float32, [3, max_len*1024])
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
        final_reshape = tf.reshape(input_embed, [3, max_len, 1024])
        with tf.Session() as sess:
            res_ = sess.run([final_reshape], feed_dict={x_: input_embed})
    return res_


def model(input_embed):

    reshaped_ = reshape_input_embeddings(input_embed, first_reshaping=True)
    pad_embedding = pad_sequences(reshaped_, maxlen=1024 * max_len, dtype='float32', padding='post')
    reshaped_embedding = reshape_input_embeddings(pad_embedding, first_reshaping=False)[0]
    # sum = sum_all_layers(reshaped_embedding)
    # print(sum[0], np.array(sum[0]).shape)
    return reshaped_embedding, np.array(reshaped_embedding.shape)


def run_for_file(data, i, part_num):
    new_embeddings_file = os.path.join(datadir, '{}_{}_review_embeddings_3_12.hdf5'.format(data, i))
    embeddings_file = os.path.join(datadir, '{}_{}_review_embeddings.hdf5'.format(data, i))
    with h5py.File(new_embeddings_file, 'w') as fout:
        for idx in range(part_num):
            review_embed = get_embedding_and_label(embeddings_file, idx)
            embed, shape = model(review_embed)
            ds = fout.create_dataset(
                '{}'.format(idx),
                np.array(embed).shape[:], dtype='float32',
                data=embed[:, :, :]
            )


# for i in range(5):
#     run_for_file('train', str(i+1))
#
# for i in range(5):
#     run_for_file('test', str(i + 1))

run_for_file('valid', str(0), part_num=2500)
run_for_file('train_left', str(5), part_num=2500)

# embeddings_file = os.path.join(datadir, '{}_{}_review_embeddings.hdf5'.format('train', 1))
# review_embed = get_embedding_and_label(embeddings_file, 0)
# print(model(review_embed))
fin = time.time()
print('\n', 'time: %s' % str(fin - s))
