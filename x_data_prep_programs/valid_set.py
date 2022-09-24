import numpy as np
from collections import Counter
import random

file = 'C:/Users/Admin/PycharmProjects/Elmo/precomputed_data/reshaped_data/5_train_labels.txt'
txt_file = 'C:/Users/Admin/PycharmProjects/Elmo/prepared_dataset/splitted_dataset/reviews/5_train_dataset.txt'
valid_x = 'C:/Users/Admin/PycharmProjects/Elmo/prepared_dataset/splitted_dataset/reviews/valid_dataset.txt'
valid_y = 'C:/Users/Admin/PycharmProjects/Elmo/prepared_dataset/splitted_dataset/reviews/valid_labels.txt'
train_x = 'C:/Users/Admin/PycharmProjects/Elmo/prepared_dataset/splitted_dataset/reviews/5_train_dataset_left.txt'
train_y = 'C:/Users/Admin/PycharmProjects/Elmo/prepared_dataset/splitted_dataset/reviews/5_train_labels_left.txt'


with open(file, 'r') as f:
    t = f.readlines()
lst = []
lst2 = []
for el in t:
    lst.append(int(el.split()[0]))
    lst2.append(int(el.split()[0]))
# print(lst)
# print(Counter(lst))
# # 1250 0, 1250 1
# ones_idxs = []
# zeros_idxs = []
# for i, el in enumerate(lst):
#     if el == 1 and len(ones_idxs) < 1250:
#         ones_idxs.append(i)
#         lst[i] = 2
#     elif el == 0 and len(zeros_idxs) < 1250:
#         zeros_idxs.append(i)
#         lst[i] = 2
#
# ost = []  # train
# for i, el in enumerate(lst):
#     if el != 2:
#         ost.append(i)
# random.shuffle(ost)
#
# with open(txt_file, 'r', encoding='utf-8') as f:
#     tt = f.readlines()
#
# train_labels = []
# train_reviews = []
# for i in ost:
#     train_labels.append(lst2[i])
#     train_reviews.append(tt[i])
#
# labels = ones_idxs + zeros_idxs
# random.shuffle(labels)
#
# valid_labels = []
# valid_reviews = []
#
# for i in labels:
#     valid_labels.append(lst2[i])
#     valid_reviews.append(tt[i])
# print(len(valid_labels), valid_labels)
# print(len(valid_reviews))


# with open(valid_x, 'w', encoding='utf-8') as fout:
#     for rev in valid_reviews:
#         fout.write(''.join(rev))
#
# with open(valid_y, 'w', encoding='utf-8') as fout:
#     for label in valid_labels:
#         fout.write(' '.join(str(label) + '\n'))
#
# with open(train_x, 'w', encoding='utf-8') as fout:
#     for rev in train_reviews:
#         fout.write(''.join(rev))
#
# with open(train_y, 'w', encoding='utf-8') as fout:
#     for label in train_labels:
#         fout.write(' '.join(str(label) + '\n'))

