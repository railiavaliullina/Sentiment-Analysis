import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

s = time.time()
tf.reset_default_graph()

# Elmo -> biLSTM -> MaxPool -> 2 layer fc (512)

datadir = 'precomputed_data'
files_num = 1
vec_dim = 1024
train_set_size = 5000
num_classes = 2
num_layers = 3
max_len = 4

batch_size = 2
learning_rate = 0.00001
decay_steps = 10000
decay_rate = 0.95
lambda_ = 5
units = 100
epochs = 5


def get_embedding_and_label(emb_file, lab_file, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    with open(lab_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    return embedding, labels[idx].split()[0]


def load_data(data, fl, valid):
    embeddings = []
    labels = []
    if valid:
        part_num = 2500
        embeddings_file = os.path.join(datadir, 'reshaped_data/3_12_sent/'
                                                '{}_review_embeddings_3_12.hdf5'.format(data))
        labels_file = os.path.join(datadir, 'reshaped_data/{}_labels.txt'.format(data))
        print('Loading data from files:\n{}\n{}'.format(embeddings_file, labels_file))
        for rev_idx in range(part_num):
            review_embed, label = get_embedding_and_label(embeddings_file, labels_file, rev_idx)
            embeddings.append(review_embed)
            labels.append(int(label))
    else:
        part_num = 5000
        for i in range(1, files_num + 1):
            embeddings_file = os.path.join(datadir, 'reshaped_data/3_12_sent/'
                                                    '{}_{}_review_embeddings_3_12.hdf5'.format(data, i))
            labels_file = os.path.join(datadir, 'reshaped_data/{}_{}_labels.txt'.format(i, data))
            print('Loading data from files:\n{}\n{}'.format(embeddings_file, labels_file))
            if i == files_num and fl:
                part_num = 2500
            for rev_idx in range(part_num):
                review_embed, label = get_embedding_and_label(embeddings_file, labels_file, rev_idx)
                embeddings.append(review_embed)
                labels.append(int(label))
    return embeddings, labels


def one_hot_encoding(labels):
    labels = np.array(labels)
    batch_one_hot = np.zeros(shape=(len(labels), num_classes))
    mask = labels == 1
    batch_one_hot[mask, 1] = 1
    batch_one_hot[~mask, 0] = 1
    return batch_one_hot


def generate_batch(x, y, b):
    return x[b: (b + batch_size)], y[b: (b + batch_size)]


def reshape_input(input_x):
    cur_sent_count = input_x.shape[1]
    dif = max_len - cur_sent_count
    if dif > 0:
        z = np.zeros([num_layers, dif, vec_dim], dtype=np.float32)
        input_x = np.concatenate([input_x, z], axis=1)
    elif dif < 0:
        input_x = np.stack([x[:max_len][:] for x in input_x])
    return input_x


def model(x_train, y_train, x_test, y_test, x_valid, y_valid):
    x = tf.placeholder(tf.float32, shape=[None, num_layers, max_len, vec_dim], name='x')
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool, name='training')

    # Variables
    global_step = tf.Variable(0, trainable=False, name='global_step')
    weights_fc = tf.Variable(tf.truncated_normal([2*units, num_classes]), name='weights_fc')
    biases_fc = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='biases_fc')
    # weights_fc_2 = tf.Variable(tf.truncated_normal([512, num_classes]), name='weights_fc_2')
    # biases_fc_2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='biases_fc_2')

    # Training scheme
    fl_dim = max_len * vec_dim
    s_1 = tf.nn.softmax(tf.Variable(tf.truncated_normal([fl_dim, fl_dim])))
    s_2 = tf.nn.softmax(tf.Variable(tf.truncated_normal([fl_dim, fl_dim])))
    s_3 = tf.nn.softmax(tf.Variable(tf.truncated_normal([fl_dim, fl_dim])))
    gamma = tf.Variable(tf.constant(0.1))

    # Model
    x_input = tf.reshape(x, [-1, num_layers, fl_dim])  # (-1, 3, 12*1024)
    x_input = tf.unstack(x_input, num_layers, axis=1)  # (3(-1, 12*1024))

    # batch normalization for each layer
    x_input_1 = tf.layers.batch_normalization(inputs=x_input[0], training=training)
    x_input_2 = tf.layers.batch_normalization(inputs=x_input[1], training=training)
    x_input_3 = tf.layers.batch_normalization(inputs=x_input[2], training=training)

    x_input_1 = tf.matmul(x_input_1, s_1)  # (3(-1, 12*1024))
    x_input_2 = tf.matmul(x_input_2, s_2)  # (3(-1, 12*1024))
    x_input_3 = tf.matmul(x_input_3, s_3)  # (3(-1, 12*1024))

    x_input_1 = tf.multiply(x_input_1, gamma)
    x_input_2 = tf.multiply(x_input_2, gamma)
    x_input_3 = tf.multiply(x_input_3, gamma)

    x_input = tf.stack([x_input_1, x_input_2, x_input_3], axis=1)  # (-1, 3, 12*1024)
    x_input = tf.reshape(x_input, [-1, num_layers, max_len, vec_dim])  # (-1, 3, 12, 1024)
    x_input = tf.reduce_sum(x_input, axis=1)  # (-1, 12, 1024)

    # Model

    dropout = tf.nn.dropout(x_input, keep_prob)
    lstm_layer_input = tf.unstack(dropout, max_len, axis=1)
    forward_lstm = tf.nn.rnn_cell.LSTMCell(units)
    backward_lstm = tf.nn.rnn_cell.LSTMCell(units)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm,  lstm_layer_input, dtype=tf.float32)
    outputs = tf.concat(outputs, axis=2)
    dropout_ = tf.nn.dropout(outputs, keep_prob)
    fc_layer_outputs = tf.nn.relu(tf.add(tf.matmul(dropout_, weights_fc), biases_fc))
    y_ = tf.nn.softmax(fc_layer_outputs)

    reg_L2 = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables()]) * lambda_

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y) + reg_L2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate, decay_rate).minimize(loss, global_step=global_step)
    optimizer = tf.group([optimizer, update_ops])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1)), tf.float32))

    # training
    train_losses = []
    train_iterations = []
    train_accuracies = []

    train_losses_1 = []
    train_iterations_1 = []
    train_accuracies_1 = []
    test_accuracies = []

    saver = tf.train.Saver()
    best_valid_acc = 0
    num_iter = int(train_set_size / batch_size)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        # print('Epoch: {}'.format(i + 1))
        for j in range(num_iter):
            # print(x_train[0])
            batch_x, batch_y = generate_batch(x_train, y_train, j * batch_size)
            # print(batch_x)
            # print(np.shape(batch_x), batch_x)
            # print([x for x in batch_x])
            batch_x = np.array([reshape_input(x) for x in batch_x])
            # print(np.shape(batch_x), batch_x)

            _, cur_step, tr_loss, tr_acc, _ = sess.run([optimizer, global_step, loss, accuracy, update_ops],
                                                       feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 0.5,
                                                                  training: True})

            train_losses.append(tr_loss)
            train_iterations.append(cur_step)
            train_accuracies.append(tr_acc)

            if j % 100 == 0:
                tr_loss_mean = np.mean(np.array(train_losses))
                train_losses_1.append(tr_loss_mean)
                tr_acc_mean = np.mean(np.array(train_accuracies))
                train_accuracies_1.append(tr_acc_mean)
                train_iterations_1.append(j + i * num_iter)

                x_valid = np.array([reshape_input(x) for x in x_valid])
                valid_acc = sess.run(accuracy, feed_dict={x: x_valid, y: y_valid, keep_prob: 1.0, training: False})

                print('iteration: {}, train_loss_mean: {}, train_accuracy_mean: {}, '
                      'valid_accuracy_cur: {}'.format(j + 1, tr_loss_mean, tr_acc_mean, valid_acc))

                if valid_acc > best_valid_acc:
                    saver.save(sess=sess, save_path='saved_models/700_bilstm_mp/best_model.ckpt')
                    text = 'epoch: {}, iteration: {}, train_loss_mean: {}, train_loss: {}, train_accuracy_mean: {}, ' \
                           'train_accuracy: {}, valid_accuracy: {}'.format(i + 1, j + 1, tr_loss_mean, tr_loss,
                                                                           tr_acc_mean,
                                                                           tr_acc,
                                                                           valid_acc)

                    best_valid_acc = valid_acc
                    with open('saved_models/700_bilstm_mp/info.txt', 'w') as fout:
                        fout.write(text)

    print('Calculating test_accuracy...')
    for i in range(epochs):
        for j in range(num_iter):
            batch_x_test, batch_y_test = generate_batch(x_test, y_test, j * batch_size)
            batch_x_test = np.array([reshape_input(x) for x in batch_x_test])
            test_acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0, training: False})
            test_accuracies.append(test_acc)
    print('\nTest accuracy : {}'.format(np.mean(test_accuracies)))
    print('\nBest validation accuracy : {}'.format(best_valid_acc))

    sess.close()

    plt.subplot(1, 2, 1)
    plt.plot(train_iterations_1, train_losses_1)
    plt.title('Error')
    plt.subplot(1, 2, 2)
    plt.plot(train_iterations_1, train_accuracies_1)
    plt.title('Accuracy')
    plt.show()


X_train, Y_train = load_data('train', fl=True, valid=False)
X_test, Y_test = load_data('test', fl=False, valid=False)
X_valid, Y_valid = load_data('valid', fl=False, valid=True)
Y_train = one_hot_encoding(Y_train)
Y_test = one_hot_encoding(Y_test)
Y_valid = one_hot_encoding(Y_valid)
tr_time_s = time.time()
model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid)
fin = time.time()
print('\nTime: {} min'.format((fin - s) / 60))
# 0.8507199883460999
