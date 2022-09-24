import tensorflow as tf
import numpy as np

def length_calc(input):
  used = tf.sign(tf.reduce_max(tf.abs(input), axis=2))
  length = tf.reduce_sum(used, axis=1)
  length = tf.cast(length, tf.int32)
  return length

def rnn_output(input):
    length=length_calc(input)
    output, state = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(512),
        input,
        dtype=tf.float32,
        sequence_length=length
    )
    return output, length

def last_rnn_output(rnn_output, length):
    batch_size = tf.shape(rnn_output)[0]
    max_length = tf.shape(rnn_output)[1]
    out_size = tf.shape(rnn_output)[2]
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(rnn_output, [-1, out_size])
    last = tf.gather(flat, index)
    return last

batch_size=2
feature_dim=1024

input_batch = tf.placeholder(dtype=tf.float32, shape=(None, None, feature_dim))
length = length_calc(input_batch)
output_rnn, length = rnn_output(input_batch)
last_rnn = last_rnn_output(output_rnn, length)
init=tf.global_variables_initializer()

for i in range(100):
    timestamp = np.random.randint(5) + 3
    batch=np.zeros((batch_size,timestamp,feature_dim))
    for i, elem in enumerate(batch):
        fill=np.random.randint(timestamp)+1
        batch[i][:fill]=np.ones((fill,feature_dim))

    with tf.Session() as sess:
        init.run()
        length_mask_np = sess.run(length, feed_dict={input_batch:batch})
        print(f"Shape: {length_mask_np.shape}")
        # print(length_mask_np)
        output_rnn_np = sess.run(output_rnn, feed_dict={input_batch:batch})
        print(f"Shape: {output_rnn_np.shape}")
        # print(output_rnn_np)
        last_rnn_np = sess.run(last_rnn, feed_dict={input_batch:batch})
        print(f"Shape: {last_rnn_np.shape}")
        # print(last_rnn_np)
