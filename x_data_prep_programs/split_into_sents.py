import os
import nltk

path_to_dataset = 'prepared_dataset'
path_to_splitted_dataset_files = 'prepared_dataset/splitted_dataset/reviews'
path_to_splitted_into_sent_reviews = 'prepared_dataset/splitted_dataset/sentences'
num_parts = 5


def split_reviews_into_sentences(line_sentences, file):
    with open(file, 'r', encoding='utf-8') as f:
        t = f.readlines()
    for line in t:
        line_sentences.append(nltk.tokenize.sent_tokenize(line))
    return line_sentences


def write_to_file_sentences(line_sentences, txt_file):
    with open(txt_file, 'w', encoding='utf-8') as fout:
        for review in line_sentences:
            for sentence in review:
                fout.write(sentence + '\n')


def write_to_file_count_of_sentences(line_sentences, txt_file):
    with open(txt_file, 'w', encoding='utf-8') as fout:
        for review in line_sentences:
            fout.write(str(len(review)) + '\n')


def get_name_of_splitted_file(data, i):
    file = os.path.join(path_to_splitted_dataset_files, '{}_{}.txt'.format(str(i + 1), data))
    return file


def make_result_files(i, new_file_name_data, new_file_name_count, data):
    file = get_name_of_splitted_file(data, i)
    line_sentences = []
    list = split_reviews_into_sentences(line_sentences, file)
    txt_file = os.path.join(path_to_splitted_into_sent_reviews, new_file_name_data)
    write_to_file_sentences(list, txt_file)
    count_of_sent_train_set = os.path.join(path_to_splitted_into_sent_reviews, new_file_name_count)
    write_to_file_count_of_sentences(list, count_of_sent_train_set)


# for train set files
for i in range(num_parts):
    make_result_files(i, 'splitted_into_sent_{}_{}.txt'.format('train', str(i+1)),
                      'count_of_sent_{}_{}.txt'.format('train', str(i+1)), data='train_dataset')

# for test set files
for i in range(num_parts):
    make_result_files(i, 'splitted_into_sent_{}_{}.txt'.format('test', str(i+1)),
                      'count_of_sent_{}_{}.txt'.format('test', str(i+1)), data='test_dataset')
