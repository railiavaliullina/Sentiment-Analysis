import tensorflow as tf
import numpy as np
import h5py
import os
import time
s = time.time()

tf.reset_default_graph()

datadir = 'precomputed_data'
part_num = 5000
count_of_Elmo_layers = 3


def read_info_about_review(labels_file, review_file, embedding_file, sent_idx, rev_idx, fl_lbl):
    with h5py.File(embedding_file, 'r') as f:
        embedding = f[str(sent_idx)][...]
    if fl_lbl:
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        with open(review_file, 'r', encoding='utf-8') as f:
            review = f.readlines()
        return embedding, labels[rev_idx], review[sent_idx]
    else:
        return embedding


def get_count_of_sent(counts_file, idx):
    with open(counts_file, 'r', encoding='utf-8') as f:
        counts_file = f.readlines()
    if idx == 0:
        global prev_sent_count
        prev_sent_count = 0
    else:
        prev_sent_count += int(counts_file[idx - 1])
    return int(counts_file[idx]), prev_sent_count


def average_of_word_embeddings(data, i, idx):
    review_embedding = []
    embedding_file = os.path.join(datadir, '{}/{}_{}_elmo_embeddings.hdf5'.format(data, i, data))
    count_of_sent = os.path.join(datadir, '{}/count_of_sent_{}_{}.txt'.format(data, data, i))
    sent_count, prev_sent_count = get_count_of_sent(count_of_sent, idx)
    for sent_idx in range(sent_count):  # для каждого предложения
        print(idx, sent_count, sent_idx, prev_sent_count, sent_idx + prev_sent_count)
        review_embed = read_info_about_review('', '', embedding_file, sent_idx + prev_sent_count, idx, fl_lbl=False)
        sentence_embedding = change_to_sent_embeddings(review_embed)
        review_embedding.append(sentence_embedding)
    prev_sent_count += sent_count
    return review_embedding


def change_to_sent_embeddings(review_embed):  # changes tensors of n_tokens to tensor of one sentence
    x = tf.placeholder(tf.float32, [3, None, 1024])
    mean = tf.reduce_mean(x, 1)
    with tf.Session() as sess:
        res = sess.run([mean], feed_dict={x: review_embed})
    return res


def review_embeddings(data, i, idx):  # (3, n_sent, 1024)
    review_embed = average_of_word_embeddings(data, i, idx)
    n_sent = np.array(review_embed).shape[0]  # count of sent-s
    all_layers = []
    for j in range(3):
        lst = []
        for i in range(n_sent):
            lst.append(review_embed[i][0][j])
        layer_1 = stack_sentences_in_one_layer__stack_all_layers(lst, n_sent, stack_sent=True)
        all_layers.append(layer_1)
    stacked_layers = stack_sentences_in_one_layer__stack_all_layers(all_layers, n_sent, stack_sent=False)
    return stacked_layers, np.array(stacked_layers).shape


def stack_sentences_in_one_layer__stack_all_layers(lst, n_sent,
                                                   stack_sent):  # result : review repr-n = tensor (3, n_sent, 1024)
    tf.reset_default_graph()
    if stack_sent:
        x1 = tf.placeholder(tf.float32, [n_sent, 1024])
    else:  # stack layers
        x1 = tf.placeholder(tf.float32, [3, n_sent, 1024])
    concat = tf.stack(x1)
    with tf.Session() as sess:
        res = sess.run(concat, feed_dict={x1: lst})
    return res


def write_to_hdf5(file, data, i):
    with h5py.File(file, 'w') as fout:

        for idx in range(part_num):
            review_emb, shape = review_embeddings(data, i, idx)
            ds = fout.create_dataset(
                '{}'.format(idx),
                review_emb.shape[:], dtype='float32',
                data=review_emb[:, :, :]
            )


def run_for_set(data, i):
    review_embeddings_file = os.path.join(datadir, 'review_emb/{}_{}_review_embeddings.hdf5'.format(data, i))
    write_to_hdf5(review_embeddings_file, data, i)


# for i in range(5):
#     run_for_set('train', str(i+1))

# for i in range(5):
#     run_for_set('test', str(i+1))

fin = time.time()
print('\n', 'time: %s' % str(fin - s))
