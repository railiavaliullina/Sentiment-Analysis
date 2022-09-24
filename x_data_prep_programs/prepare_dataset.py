import numpy as np
import os
import random
from collections import Counter
import re

path_to_dataset = 'C:/Users/Admin/Desktop/ML/IMDB'
path_to_new_dataset = 'prepared_dataset'
path_to_splitted_dataset = 'prepared_dataset/splitted_dataset/reviews'
num_parts = 5


def get_dataset_files_to_prepare(folder, path):
    path = os.path.join(path, folder)
    files_list = np.array(os.listdir(path))
    return files_list, path


def get_list_of_reviews_with_added_labels(files_list, path, label_value):
    list_of_reviews = []
    for file in files_list:
        file = os.path.join(path, file)
        with open(file, 'r', encoding='utf-8') as f:
            t = str(label_value) + ' '
            t += f.read()
        t = t.replace('<br /><br />', ' ')
        list_of_reviews.append(t)
    return list_of_reviews


def write_reviews_into_two_txt_file(reviews, txt_file):
    with open(txt_file, 'a', encoding='utf-8') as fout:
        for review in reviews:
            fout.write(review + '\n')


def shuffle_txt_files(txt_file, new_txt_file):
    lines = open(txt_file, encoding='utf-8').readlines()
    random.shuffle(lines)
    open(new_txt_file, 'w', encoding='utf-8').writelines(lines)


def read_shuffled_reviews(file, path):
    file = os.path.join(path, file)
    with open(file, 'r', encoding='utf-8') as f:
        t = f.readlines()
    return t


def tokenize_all_reviews(reviews):
    tokenized_reviews = []
    for review in reviews:
        tokenized_reviews.append(review.split())
    return tokenized_reviews


def make_list_with_reviews_labels(y_file, x_file):
    for x in x_file:
        label = int(x[0])
        y_file.append(label)
        x.remove(x[0])
    return y_file, x_file


def create_dataset_and_label_file(dataset_txt, tokenized_reviews_file, labels_txt, labels_list):
    dataset_file = os.path.join(path_to_new_dataset, dataset_txt)
    with open(dataset_file, 'w', encoding='utf-8') as fout:
        for review in tokenized_reviews_file:
            fout.write(' '.join(review) + '\n')

    labels_file = os.path.join(path_to_new_dataset, labels_txt)
    with open(labels_file, 'w', encoding='utf-8') as fout:
        for label in labels_list:
            fout.write(' '.join(str(label) + '\n'))


def make_lists_of_dataset_files(dataset_file):
    file_reviews_list = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            file_reviews_list.append(line)
    return file_reviews_list


def split_data_into_parts(file, num_parts):
    data = make_lists_of_dataset_files(file)
    data = np.array(data)
    reviews_num = data.shape[0]
    part_num = int(reviews_num / num_parts)
    k = 0
    all_parts_reviews = []
    while k < num_parts:
        one_part_reviews = []
        for i in range(part_num):
            one_part_reviews.append(data[i + k * part_num])
        all_parts_reviews.append(one_part_reviews)
        k += 1
    return np.array(all_parts_reviews)


def write_splitted_dataset_to_files(file, num_parts, data, path_to_splitted_dataset):
    splitted_data = split_data_into_parts(file, num_parts)
    for i in range(num_parts):
        file = os.path.join(path_to_splitted_dataset, '{}_{}.txt'.format(str(i + 1), data))
        with open(file, 'w', encoding='utf-8') as fout:
            for review in splitted_data[i]:
                fout.write(review)


def make_vocab(train_set_file):
    with open(train_set_file, 'r', encoding='utf-8') as f:
        t = f.read()
    t = re.findall(r'\w+|[^\w\s]', t)
    vocab = []
    for token in t:
        vocab.append(token)
    vocab = Counter(vocab)
    sorted_tokens = []
    for el in vocab.most_common():
        sorted_tokens.append(el[0])
    vocab_file = os.path.join(path_to_new_dataset, 'vocab.txt')
    with open(vocab_file, 'w', encoding='utf8') as fout:
        fout.write('</S>' + '\n')
        fout.write('<S>' + '\n')
        fout.write('<UNK>' + '\n')
        for token in sorted_tokens:
            fout.write(token + '\n')


# get reviews from data set and add labels
# one line = label + review

files_neg_train, path = np.array(get_dataset_files_to_prepare('neg_train', path_to_dataset))
reviews_neg_train = get_list_of_reviews_with_added_labels(files_neg_train, path, 0)

files_pos_train, path = np.array(get_dataset_files_to_prepare('pos_train', path_to_dataset))
reviews_pos_train = get_list_of_reviews_with_added_labels(files_pos_train, path, 1)

files_neg_test, path = np.array(get_dataset_files_to_prepare('neg_test', path_to_dataset))
reviews_neg_test = get_list_of_reviews_with_added_labels(files_neg_test, path, 0)

files_pos_test, path = np.array(get_dataset_files_to_prepare('pos_test', path_to_dataset))
reviews_pos_test = get_list_of_reviews_with_added_labels(files_pos_test, path, 1)

# write all train reviews in one file, all test reviews in another

write_reviews_into_two_txt_file(reviews_neg_train, os.path.join(path_to_new_dataset, 'train.txt'))
write_reviews_into_two_txt_file(reviews_pos_train, os.path.join(path_to_new_dataset, 'train.txt'))
write_reviews_into_two_txt_file(reviews_neg_test, os.path.join(path_to_new_dataset, 'test.txt'))
write_reviews_into_two_txt_file(reviews_pos_test, os.path.join(path_to_new_dataset, 'test.txt'))

# shuffle reviews(lines) in train.txt, test.txt

shuffle_txt_files(os.path.join(path_to_new_dataset, 'train.txt'), os.path.join(path_to_new_dataset,
                                                                               'shuffled_train_file.txt'))
shuffle_txt_files(os.path.join(path_to_new_dataset, 'test.txt'), os.path.join(path_to_new_dataset,
                                                                              'shuffled_test_file.txt'))

# tokenize reviews in train and test files

tokenized_train_reviews = tokenize_all_reviews(read_shuffled_reviews('shuffled_train_file.txt', path_to_new_dataset))
tokenized_test_reviews = tokenize_all_reviews(read_shuffled_reviews('shuffled_test_file.txt', path_to_new_dataset))

# extract labels to label files

Y_train = []
Y_test = []

y_train, x_train = make_list_with_reviews_labels(Y_train, tokenized_train_reviews)
y_test, x_test = make_list_with_reviews_labels(Y_test, tokenized_test_reviews)

create_dataset_and_label_file('new_train_dataset.txt', tokenized_train_reviews, 'labels_train.txt', y_train)
create_dataset_and_label_file('new_test_dataset.txt', tokenized_test_reviews, 'labels_test.txt', y_test)

# split train and test sets into parts (dataset files, labels files)

write_splitted_dataset_to_files(os.path.join(path_to_new_dataset, 'new_train_dataset.txt'), num_parts,
                                'train_dataset', path_to_splitted_dataset)
write_splitted_dataset_to_files(os.path.join(path_to_new_dataset, 'new_test_dataset.txt'), num_parts,
                                'test_dataset', path_to_splitted_dataset)
write_splitted_dataset_to_files(os.path.join(path_to_new_dataset, 'labels_train.txt'), num_parts,
                                'train_labels', path_to_splitted_dataset)
write_splitted_dataset_to_files(os.path.join(path_to_new_dataset, 'labels_test.txt'), num_parts,
                                'test_labels', path_to_splitted_dataset)

# making vocabulary of tokens from train set reviews
# result : 93 997 tokens in vocab.txt file

make_vocab(os.path.join(path_to_new_dataset, 'new_train_dataset.txt'))
