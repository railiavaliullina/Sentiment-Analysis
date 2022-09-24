import h5py
import os
import numpy as np
import time

s = time.time()
path_to_dataset = 'prepared_dataset'
path_to_elmo_vectors = 'precomputed_data'
path_to_splitted_into_sent_reviews = 'prepared_dataset/splitted_dataset/sentences'
path = 'precomputed_data/files'
count_of_reviews = 5000
num_parts = 5


def get_embedding(file):
    f = h5py.File(file, 'r')
    return f


def convert_to_int_list(list_):
    return list(map(int, list_))


def keys(f):
    with h5py.File(f, 'r') as f:
        return [key for key in f.keys()]


def read_txt_files(file):
    with open(file, 'r', encoding='utf-8') as f:
        t = f.readlines()
    return t


def get_each_reviews_sent_idxs(data, i):
    embeddings_file = os.path.join(path_to_elmo_vectors, '{}/{}_{}_elmo_embeddings.hdf5'.format(data, i, data))
    sent_counts_file = os.path.join(path, 'count_of_sent_{}_{}.txt'.format(data, i))
    print('Opened files : \n', embeddings_file, '\n', sent_counts_file, '\n')
    counts_train = np.array(convert_to_int_list(read_txt_files(sent_counts_file)))
    embed = get_embedding(embeddings_file)
    keys = np.array(list(embed.keys()))   # индексы предложений
    count_of_sent = np.array(keys).shape  # 62 551
    count_of_sent = count_of_sent[0]
    list_of_idx = []
    for j in range(count_of_reviews):
        list_of_idx.append([])
    for i in range(count_of_sent):
        sent_idx = keys[i]
        list_of_idx[counts_train[int(sent_idx)]].append(sent_idx)
    return list_of_idx, embed


def get_embeddings_of_all_sents_of_review(list_of_sent_idx_lists, embeddings):
    max_counts_of_sent_of_all_reviews = []
    k = 0
    for lst in list_of_sent_idx_lists:
        count_of_sent = []
        if k % 100 == 0:
            print('review %s' % str(k))
        for sent_idx in lst:
            count_of_sent.append(np.array(list(embeddings[sent_idx])).shape[1])
        k += 1
        max_counts_of_sent_of_all_reviews.append(max(count_of_sent))
    return max_counts_of_sent_of_all_reviews


def write_to_file(file, max_counts):
    with open(file, 'w', encoding='utf-8') as fout:
        for max_count in max_counts:
            fout.write(str(max_count) + '\n')


def run_functions_for_file(data, i):
    reviews_sentences_lst, embeddings = get_each_reviews_sent_idxs(str(data), str(i))
    max_counts = get_embeddings_of_all_sents_of_review(reviews_sentences_lst, embeddings)
    file_ = os.path.join(path_to_elmo_vectors, '{}/max_tokens_count/max_tokens_count_{}_{}.txt'.format(data, data, i))
    write_to_file(file_, max_counts)


# write to files max_count of tokens in sentences in each review

# for i in range(num_parts):
#    run_functions_for_file('train', str(i+1))
#    run_functions_for_file('test', str(i+1))


# for taking average of sentences
# for getting embedding of review

# def reshape_sentences_in_reviews():

def read_embed_and_inf(data, i):
    file = os.path.join(path_to_elmo_vectors, '{}/max_tokens_count/max_tokens_count_{}_{}.txt'.format(data, data, i))
    max_counts = read_txt_files(file)
    return max_counts


# print(read_embed_and_inf('train', '1'))


fin = time.time()
print('\n', 'time: %s' % str(fin - s))
