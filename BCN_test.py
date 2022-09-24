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
max_len = 2
vec_dim = 1024
train_set_size = 5000
valid_set_size = 2500
test_set_size = 25000
num_classes = 2

batch_size = 2
learning_rate = 0.0001
lambda_ = 0.001
units = 2
epochs = 5
# Maxout network
k, m = 2, 2
maxout_units = 1

# Test accuracy : 0.593029797077179
# Best validation accuracy : 0.545192301273346

# with dropouts:
#  Test accuracy : 0.5355313420295715
#  Best validation accuracy : 0.5089476704597473

# Xavier init:
#  Test accuracy : 0.795134425163269
#  Best validation accuracy : 0.7527243494987488

#
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


def get_embedding_and_label(emb_file, lab_file, idx):
    with h5py.File(emb_file, 'r') as f:
        embedding = f[str(idx)][...]
    with open(lab_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    return embedding[0], labels[idx].split()[0]


def load_data(data, fl, valid):
    embeddings = []
    labels = []
    if valid:
        part_num = 2500
        embeddings_file = os.path.join(datadir,
                                       'reshaped_data/{}_review_embeddings.hdf5'.format(data))
        labels_file = os.path.join(datadir, 'reshaped_data/{}_labels.txt'.format(data))
        print('Loading data from files:\n{}\n{}'.format(embeddings_file, labels_file))
        for rev_idx in range(part_num):
            review_embed, label = get_embedding_and_label(embeddings_file, labels_file, rev_idx)
            embeddings.append(review_embed)
            labels.append(int(label))
    else:
        part_num = 5000
        for i in range(1, files_num + 1):
            embeddings_file = os.path.join(datadir, 'reshaped_data/'
                                                    '{}_{}_review_embeddings.hdf5'.format(data, i))
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
    cur_sent_count = input_x.shape[0]
    dif = max_len - cur_sent_count
    if dif > 0:
        z = np.zeros([dif, vec_dim], dtype=np.float32)
        input_x = np.concatenate([input_x, z], axis=0)
    elif dif < 0:
        input_x = np.array(input_x[:max_len])
    return input_x


def MLP_maxout(input_vector, d, m_, maxout_layer_num):
    weights_maxout = [tf.get_variable(shape=[d, m_],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      name='weights_maxout{}_{}'.format(str(maxout_layer_num), str(i))) for i in
                      range(k)]
    biases_maxout = [tf.Variable(tf.constant(0.1, shape=[m_]),
                                 name='biases_maxout_{}_{}'.format(str(maxout_layer_num), str(i))) for i in range(k)]
    res = [tf.add(tf.matmul(input_vector, weights_maxout[i]), biases_maxout[i]) for i in range(k)]
    return tf.stack(res, axis=1)


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def model(x_train, y_train, x_test, y_test, x_valid, y_valid):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, max_len, 1024], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    training = tf.placeholder(tf.bool, name='training')

    # Variables
    flatten_dim = max_len * 1024
    global_step = tf.Variable(0, trainable=False, name='global_step')
    weights_fc_x = tf.get_variable(shape=[flatten_dim, flatten_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   name='weights_fc_x')
    biases_fc_x = tf.Variable(tf.constant(0.1, shape=[flatten_dim]), name='biases_fc_x')
    weights_fc_y = tf.get_variable(shape=[flatten_dim, flatten_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   name='weights_fc_y')
    biases_fc_y = tf.Variable(tf.constant(0.1, shape=[flatten_dim]), name='biases_fc_y')

    flatten_dim_2 = 3 * max_len * 2 * units
    W_at_x = tf.get_variable(shape=[flatten_dim_2, flatten_dim_2],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='weights_at_x')
    b_at_x = tf.Variable(tf.constant(0.1, shape=[flatten_dim_2]), name='biases_at_x')
    W_at_y = tf.get_variable(shape=[flatten_dim_2, flatten_dim_2],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             name='weights_at_y')
    b_at_y = tf.Variable(tf.constant(0.1, shape=[flatten_dim_2]), name='biases_at_y')

    tf.summary.histogram('weights_fc_x_sum', weights_fc_x)
    tf.summary.histogram('weights_fc_y_sum', weights_fc_y)
    tf.summary.histogram('weights_at_x_sum', W_at_x)
    tf.summary.histogram('weights_at_y_sum', W_at_y)

    # Model

    # ReLu network
    with tf.name_scope('ReLu_network_X'):
        f_c_x = tf.nn.relu(tf.add(tf.matmul(tf.reshape(x, [-1, flatten_dim]), weights_fc_x), biases_fc_x))
        f_c_x = tf.reshape(f_c_x, [-1, max_len, 1024])

    with tf.name_scope('ReLu_network_Y'):
        f_c_y = tf.nn.relu(tf.add(tf.matmul(tf.reshape(x, [-1, flatten_dim]), weights_fc_y), biases_fc_y))
        f_c_y = tf.reshape(f_c_y, [-1, max_len, 1024])

    # Encoder

    # bilstm 1
    with tf.name_scope('Encoder_X'):
        outputs_x, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units),
                                                       tf.contrib.rnn.LSTMCell(units),
                                                       f_c_x,
                                                       dtype=tf.float32)
        X = tf.concat(outputs_x, axis=2)

    # bilstm 2
    with tf.variable_scope('Encoder_Y'):
        outputs_y, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units),
                                                       tf.contrib.rnn.LSTMCell(units),
                                                       f_c_y,
                                                       dtype=tf.float32)
        Y = tf.concat(outputs_y, axis=2)

    # biattention mechanism
    with tf.variable_scope('Biattention'):
        A = tf.matmul(X, tf.transpose(Y, perm=[0, 2, 1]))
        # A_x = tf.unstack(A, max_len, axis=2)
        A_x = tf.nn.softmax(A)
        # A_x = tf.stack(A_x, axis=2

        A_y = tf.transpose(A, perm=[0, 2, 1])
        A_y = tf.nn.softmax(A_y)
        # A_y = tf.stack(A_y, axis=2)

        C_x = tf.matmul(tf.transpose(A_x, perm=[0, 2, 1]), X)
        C_y = tf.matmul(tf.transpose(A_y, perm=[0, 2, 1]), Y)

    # integrate
    with tf.variable_scope('Integrate_X'):
        dif_context_sum_X = tf.subtract(X, C_y)
        element_w_mult_X = tf.multiply(X, C_y)
        concat_X = tf.concat([X, dif_context_sum_X, element_w_mult_X], axis=1)
        X_y, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units),
                                                 tf.contrib.rnn.LSTMCell(units),
                                                 concat_X,
                                                 dtype=tf.float32)
        X_y = tf.concat(X_y, axis=2)

    with tf.variable_scope('Integrate_Y'):
        dif_context_sum_Y = tf.subtract(Y, C_x)
        element_w_mult_Y = tf.multiply(Y, C_x)
        concat_Y = tf.concat([Y, dif_context_sum_Y, element_w_mult_Y], axis=1)
        Y_x, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units),
                                                 tf.contrib.rnn.LSTMCell(units),
                                                 concat_Y,
                                                 dtype=tf.float32)
        Y_x = tf.concat(Y_x, axis=2)

    # self - attentive pooling weights for each time step of the sequence
    with tf.variable_scope('Pool_X'):
        B_x = tf.reshape(tf.add(tf.matmul(tf.reshape(X_y, [-1, flatten_dim_2]), W_at_x), b_at_x), [-1, max_len, ve])
        # B_x = tf.unstack(tf.reshape(B_x, [-1, 3 * max_len, 2 * units]), 3 * max_len, axis=1)
        B_x = tf.nn.softmax(B_x)
        # weighted summations of each sequence (x_self_pool, y_self_pool)
        x_self = tf.matmul(tf.transpose(X_y, perm=[0, 2, 1]), B_x)
        # Pooling along the time dimension
        min_pool_X = tf.reduce_min(X_y, axis=2)
        max_pool_X = tf.reduce_max(X_y, axis=2)
        mean_pool_X = tf.reduce_mean(X_y, axis=2)
        concat_pool_X = tf.concat(
            [min_pool_X, max_pool_X, mean_pool_X, tf.reshape(x_self, [-1, 2 * units * 2 * units])],
            axis=1)

    with tf.variable_scope('Pool_Y'):
        B_y = tf.add(tf.matmul(tf.reshape(Y_x, [-1, flatten_dim_2]), W_at_y), b_at_y)
        # B_y = tf.unstack(tf.reshape(B_y, [-1, 3 * max_len, 2 * units]), 3 * max_len, axis=1)
        B_y = tf.nn.softmax(B_y)
        # weighted summations of each sequence (x_self_pool, y_self_pool)
        y_self = tf.matmul(tf.transpose(Y_x, perm=[0, 2, 1]), B_y)
        # Pooling along the time dimension
        min_pool_Y = tf.reduce_min(Y_x, axis=2)
        max_pool_Y = tf.reduce_max(Y_x, axis=2)
        mean_pool_Y = tf.reduce_mean(Y_x, axis=2)
        concat_pool_Y = tf.concat(
            [min_pool_Y, max_pool_Y, mean_pool_Y, tf.reshape(y_self, [-1, 2 * units * 2 * units])],
            axis=1)

    concat = tf.concat([concat_pool_X, concat_pool_Y], axis=1)

    # 3-layer maxout network
    with tf.variable_scope('Maxout_network'):
        d_1 = 2 * (2 * units * 2 * units + max_len * 3 * 3)
        with tf.variable_scope('Maxout_layer_1'):
            maxout_layer_1 = MLP_maxout(concat, d_1, m, maxout_layer_num=1)
            batch_norm_1 = tf.layers.batch_normalization(inputs=maxout_layer_1, training=training)
            maxout_layer_1 = max_out(batch_norm_1, maxout_units, axis=1)

            # drop_1 = tf.nn.dropout(maxout_layer_1, keep_prob)

        d = maxout_units * m
        with tf.variable_scope('Maxout_layer_2'):
            maxout_layer_2 = MLP_maxout(tf.reshape(maxout_layer_1, [-1, d]), d, m, maxout_layer_num=2)
            batch_norm_2 = tf.layers.batch_normalization(inputs=maxout_layer_2, training=training)
            maxout_layer_2 = max_out(batch_norm_2, maxout_units, axis=1)

            # drop_2 = tf.nn.dropout(maxout_layer_2, keep_prob)
        with tf.variable_scope('Maxout_layer_3'):
            maxout_layer_3 = MLP_maxout(tf.reshape(maxout_layer_2, [-1, d]), d, 2, maxout_layer_num=3)
            batch_norm_3 = tf.layers.batch_normalization(inputs=maxout_layer_3, training=training)
            maxout_layer_3 = tf.reshape(max_out(batch_norm_3, 1, axis=1), [-1, 2])

    with tf.variable_scope('Softmax_layer'):
        y_ = tf.nn.softmax(maxout_layer_3)

    with tf.name_scope('cross_entropy'):
        reg_L2 = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables()]) * lambda_
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y) + reg_L2)
        tf.summary.scalar('cross_entropy', loss)

    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_losses, train_iterations, train_accuracies, test_accuracies, valid_accuracies = [], [], [], [], []
    train_losses_1, train_iterations_1, train_accuracies_1 = [], [], []

    best_valid_acc = 0
    num_iter = int(train_set_size / batch_size)

    sess = tf.Session()
    train_writer = tf.summary.FileWriter('C:/Users/Admin/PycharmProjects/Elmo/tensorboard/BCN/maxout', sess.graph)
    # valid_writer = tf.summary.FileWriter('C:/Users/Admin/PycharmProjects/Elmo/tensorboard/BCN/valid', sess.graph)
    # test_writer = tf.summary.FileWriter('C:/Users/Admin/PycharmProjects/Elmo/tensorboard/BCN/test', sess.graph)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('saved_models/BCN/last'))

    for i in range(epochs):
        print('Epoch: {}'.format(i + 1))
        for j in range(num_iter):
            batch_x, batch_y = generate_batch(x_train, y_train, j * batch_size)
            batch_x = np.array([reshape_input(x) for x in batch_x])

            summary, _, cur_step, tr_loss, _, tr_acc = sess.run(
                [merged, optimizer, global_step, loss, update_ops, accuracy],
                feed_dict={x: batch_x,
                           y: batch_y,
                           keep_prob: 0.5,
                           training: True})

            train_losses.append(tr_loss)
            train_iterations.append(cur_step)
            train_accuracies.append(tr_acc)
            train_writer.add_summary(summary, j + i * num_iter)

            if j % 100 == 0:
                tr_loss_mean = np.mean(np.array(train_losses))
                train_losses_1.append(tr_loss_mean)
                tr_acc_mean = np.mean(np.array(train_accuracies))
                train_accuracies_1.append(tr_acc_mean)
                train_iterations_1.append(j + i * num_iter)

                for z in range(valid_set_size // batch_size):
                    batch_x_valid, batch_y_valid = generate_batch(x_valid, y_valid, z * batch_size)
                    batch_x_valid = np.array([reshape_input(x) for x in batch_x_valid])
                    summary_val, valid_acc = sess.run([merged, accuracy], feed_dict={x: batch_x_valid,
                                                                                     y: batch_y_valid,
                                                                                     keep_prob: 1.0,
                                                                                     training: False})
                    valid_accuracies.append(valid_acc)

                print('iteration: {}, train_loss_mean: {}, train_accuracy_mean: {}, '
                      'valid_accuracy_cur: {}'.format(j + 1, tr_loss_mean, tr_acc_mean, np.mean(valid_accuracies)))

                if np.mean(valid_accuracies) > best_valid_acc:
                    saver.save(sess=sess, save_path='saved_models/BCN_maxout/best_model.ckpt')
                    text = 'epoch: {}, iteration: {}, train_loss_mean: {}, train_loss: {}, train_accuracy_mean: {}, ' \
                           'train_accuracy: {}, valid_accuracy: {}'.format(i + 1, j + 1, tr_loss_mean, tr_loss,
                                                                           tr_acc_mean,
                                                                           tr_acc,
                                                                           np.mean(valid_accuracies))

                    best_valid_acc = np.mean(valid_accuracies)
                    with open('saved_models/BCN_maxout/info.txt', 'w') as fout:
                        fout.write(text)

    saver.save(sess=sess, save_path='saved_models/BCN_maxout/last/last_checkpoint.ckpt')
    print('Calculating test_accuracy...')
    for j in range(test_set_size // batch_size):
        batch_x_test, batch_y_test = generate_batch(x_test, y_test, j * batch_size)
        batch_x_test = np.array([reshape_input(x) for x in batch_x_test])
        test_acc = sess.run(accuracy, feed_dict={x: batch_x_test,
                                                 y: batch_y_test,
                                                 keep_prob: 1.0,
                                                 training: False})
        test_accuracies.append(test_acc)
        # test_writer.add_summary(summary_test, j)

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


X_train, Y_train = load_data('train', fl=True, valid=False)
X_test, Y_test = load_data('test', fl=False, valid=False)
X_valid, Y_valid = load_data('valid', fl=False, valid=True)
Y_train = one_hot_encoding(Y_train)
Y_test = one_hot_encoding(Y_test)
Y_valid = one_hot_encoding(Y_valid)
model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid)
