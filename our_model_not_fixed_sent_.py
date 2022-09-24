import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

s = time.time()
tf.reset_default_graph()

datadir = 'precomputed_data'
files_num = 1
train_set_size = 5000
valid_set_size = 2500
test_set_size = 5000
vec_dim = 1024
num_classes = 2

batch_size = 64
learning_rate = 1e-5
decay_rate = 0.5
units = 1024
epochs = 10


# Elmo -> biLSTM -> fc


def get_embedding_and_label(emb_file, lab_file, count_f, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    with open(lab_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    with open(count_f, 'r', encoding='utf-8') as f:
        counts = f.readlines()
    return embedding[0], labels[idx].split()[0], counts[idx].split()[0]


def load_data(data, fl, valid):
    embeddings = []
    labels = []
    counts = []
    if valid:
        part_num = 2500
        embeddings_file = os.path.join(datadir, 'reshaped_data/not_fixed_new/{}_review_embeddings_nf.hdf5'.format(data))
        labels_file = os.path.join(datadir, 'reshaped_data/{}_labels.txt'.format(data))
        counts_file = os.path.join(datadir, 'reshaped_data/not_fixed_sent/count_of_sent_{}.txt'.format(data))
        print('Loading data from files:\n{}\n{}'.format(embeddings_file, labels_file))
        for rev_idx in range(part_num):
            review_embed, label, count = get_embedding_and_label(embeddings_file, labels_file, counts_file, rev_idx)
            embeddings.append(review_embed)
            labels.append(int(label))
            counts.append(int(count))
    else:
        part_num = 5000
        for i in range(1, files_num + 1):
            embeddings_file = os.path.join(datadir,
                                           'reshaped_data/not_fixed_new/{}_{}_review_embeddings_nf.hdf5'.format(data, i))
            labels_file = os.path.join(datadir, 'reshaped_data/{}_{}_labels.txt'.format(i, data))
            counts_file = os.path.join(datadir, 'reshaped_data/not_fixed_sent/count_of_sent_{}_{}.txt'.format(data, i))
            print('Loading data from files:\n{}\n{}'.format(embeddings_file, labels_file))
            if i == files_num and fl:
                part_num = 2500
            for rev_idx in range(part_num):
                review_embed, label, count = get_embedding_and_label(embeddings_file, labels_file, counts_file, rev_idx)
                embeddings.append(review_embed)
                labels.append(int(label))
                counts.append(int(count))
    return np.array(embeddings), np.array(labels), np.array(counts)


def one_hot_encoding(labels):
    batch_one_hot = np.zeros(shape=(len(labels), num_classes))
    mask = labels == 1
    batch_one_hot[mask, 1] = 1
    batch_one_hot[~mask, 0] = 1
    return batch_one_hot


def generate_batch(x, y, c, b):
    return x[b: (b + batch_size)], y[b: (b + batch_size)], c[b: (b + batch_size)]


def pad(batch_input, max_sent_count):
    cur_sent_count = np.array(batch_input).shape[0]
    dif = max_sent_count - cur_sent_count
    if dif > 0:
        z = np.zeros([dif, vec_dim], dtype=np.float32)
        batch_input = np.concatenate([batch_input, z], axis=0)
    return batch_input


def model(x_train, y_train, x_test, y_test, x_valid, y_valid, c_train, c_test, c_valid):
    x = tf.placeholder(tf.float32, shape=[None, None, vec_dim], name='x')
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

    # Variables
    global_step = tf.Variable(0, trainable=False, name='global_step')
    weights_fc = tf.Variable(tf.truncated_normal([2 * units, num_classes]), name='weights_fc')
    biases_fc = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='biases_fc')

    # Model
    forward_lstm = tf.nn.rnn_cell.LSTMCell(units)
    backward_lstm = tf.nn.rnn_cell.LSTMCell(units)
    cur_sent_len = tf.count_nonzero(tf.reduce_max(tf.abs(x), axis=2), axis=1, dtype=tf.int32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, x, dtype=tf.float32, sequence_length=cur_sent_len)
    outputs = tf.concat(outputs, axis=2)

    batch_size_ = tf.shape(outputs)[0]
    max_l = tf.shape(outputs)[1]
    cells_out_size = int(outputs.get_shape()[2])

    indices = tf.range(0, batch_size_) * max_l + (cur_sent_len - 1)
    params = tf.reshape(outputs, [-1, cells_out_size])
    res = tf.gather(params, indices)

    fc_layer_outputs = tf.nn.relu(tf.add(tf.matmul(res, weights_fc), biases_fc))
    y_ = tf.nn.softmax(fc_layer_outputs)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1)), tf.float32))

    # training
    train_losses, train_iterations, train_accuracies, test_accuracies, valid_accuracies = [], [], [], [], []
    train_losses_1, train_iterations_1, train_accuracies_1 = [], [], []
    best_valid_acc = 0
    num_iter = int(train_set_size / batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(epochs):
        print('Epoch: {}'.format(i + 1))
        for j in range(num_iter):
            batch_x, batch_y, batch_c = generate_batch(x_train, y_train, c_train, j * batch_size)
            batch_x = np.array([pad(x, max(batch_c)) for x in batch_x])
            # batch_x = pad(batch_x, batch_c)
            _, cur_step, tr_loss, tr_acc = sess.run([optimizer, global_step, loss, accuracy],
                                                    feed_dict={x: batch_x,
                                                               y: batch_y})

            train_losses.append(tr_loss), train_iterations.append(cur_step), train_accuracies.append(tr_acc)

            if j % 100 == 0:
                tr_loss_mean = np.mean(np.array(train_losses))
                train_losses_1.append(tr_loss_mean)
                tr_acc_mean = np.mean(np.array(train_accuracies))
                train_accuracies_1.append(tr_acc_mean)
                train_iterations_1.append(j + i * num_iter)

                for k in range(valid_set_size//batch_size):
                    batch_x_valid, batch_y_valid, batch_c_valid = generate_batch(x_valid, y_valid, c_valid, k * batch_size)
                    batch_x_valid = np.array([pad(x, max(batch_c_valid)) for x in batch_x_valid])

                    valid_acc = sess.run(accuracy, feed_dict={x: batch_x_valid, y: batch_y_valid})
                    valid_accuracies.append(valid_acc)

                print('iteration: {}, train_loss_mean: {}, train_accuracy_mean: {}, '
                      'valid_accuracy_cur: {}'.format(j + 1, tr_loss_mean, tr_acc_mean, np.mean(valid_accuracies)))

                if np.mean(valid_accuracies) > best_valid_acc:
                    saver.save(sess=sess, save_path='saved_models/best_model.ckpt')
                    text = 'epoch: {}, iteration: {}, train_loss_mean: {}, train_loss: {}, train_accuracy_mean: {}, ' \
                           'train_accuracy: {}, valid_accuracy: {}'.format(i + 1, j + 1, tr_loss_mean, tr_loss,
                                                                           tr_acc_mean,
                                                                           tr_acc,
                                                                           np.mean(valid_accuracies))

                    best_valid_acc = np.mean(valid_accuracies)
                    with open('saved_models/info.txt', 'w') as fout:
                        fout.write(text)

    print('Calculating test_accuracy...')
    for j in range(test_set_size//batch_size):
        batch_x_test, batch_y_test, batch_c_test = generate_batch(x_test, y_test, c_test, j * batch_size)
        batch_x_test = np.array([pad(x, max(batch_c_test)) for x in batch_x_test])
        test_acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test})
        test_accuracies.append(test_acc)

    print('\nTest accuracy : {}'.format(np.mean(test_accuracies)))
    print('\nBest validation accuracy : {}'.format(best_valid_acc))

    sess.close()
    fin = time.time()
    print('\nTime: {} min'.format((fin - s) / 60))

    plt.subplot(1, 2, 1)
    plt.plot(train_iterations_1, train_losses_1)
    plt.title('Error')
    plt.subplot(1, 2, 2)
    plt.plot(train_iterations_1, train_accuracies_1)
    plt.title('Accuracy')
    plt.show()


X_train, Y_train, C_train = load_data('train', fl=True, valid=False)
X_test, Y_test, C_test = load_data('test', fl=False, valid=False)
X_valid, Y_valid, C_valid = load_data('valid', fl=False, valid=True)
Y_train = one_hot_encoding(Y_train)
Y_test = one_hot_encoding(Y_test)
Y_valid = one_hot_encoding(Y_valid)
model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid, C_train, C_test, C_valid)
