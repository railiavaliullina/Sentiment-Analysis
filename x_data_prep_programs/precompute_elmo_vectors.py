import numpy as np
import os
import tensorflow as tf
from elmo_model import dump_bilm_embeddings

path_to_dataset = 'prepared_dataset'
path_to_splitted_dataset = 'prepared_dataset/splitted_dataset'
path_to_elmo_model = 'pretrained_elmo'
path_to_elmo_vectors = 'precomputed_data'
path_to_splitted_into_sent_reviews = 'prepared_dataset/splitted_dataset/sentences'

options_file = os.path.join(path_to_elmo_model, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
weight_file = os.path.join(path_to_elmo_model, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
vocab_file = os.path.join(path_to_dataset, 'vocab.txt')

num_parts = 5


# pre-compute Elmo for all dataset


def pre_compute_elmo_vectors(dataset_file, data, options_file, weight_file, vocab):
    tf.reset_default_graph()
    embedding_file = os.path.join(path_to_elmo_vectors, '{}elmo_embeddings.hdf5'.format(data))
    dump_bilm_embeddings(vocab, dataset_file, options_file, weight_file, embedding_file)


# computing Elmo for all parts of train set

# for i in range(num_parts):
#     dataset_file = os.path.join(path_to_splitted_into_sent_reviews,
#                                 'splitted_into_sent_{}_{}.txt'.format('train', str(i + 1)))
#     print('Computing Elmo for file: ', dataset_file, '\n')
#     pre_compute_elmo_vectors(dataset_file, 'train', options_file, weight_file, vocab_file)

# dataset_file = 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data/splitted_into_sent_valid.txt'
# print('Computing Elmo for file: ', dataset_file, '\n')
# pre_compute_elmo_vectors(dataset_file, 'valid', options_file, weight_file, vocab_file)
