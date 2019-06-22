from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step1 import MNIST data
print('Step1: import mnist data')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)

# Step2 setting Parameters
print('Step2: Setting parameters')
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
n_input = 784
X = tf.placeholder('float', [None, n_input])
# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2
weights = {
    'encoder_h1':tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2':tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3':tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4':tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
    'decoder_h1':tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2':tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3':tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4':tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4':tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4':tf.Variable(tf.random_normal([n_input])),
}
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),biases['encoder_b4']))
    return layer_4

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),biases['decoder_b4']))
    return layer_4

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
# prediction
print('Step3:Prediction')
y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# launch the graph
print('Step4: launch the graph')
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c  =sess.run([optimizer, cost], feed_dict={X:batch_xs})
        if epoch % display_step == 0:
            print('Epoch:', '%04d' %(epoch+1),
                  'cost=','{:.9f}'.format(c))
    print('Optimization Finished!')
    encoder_result = sess.run(encoder_op, feed_dict={X:mnist.test.images})
    plt.scatter(encoder_result[:,0], encoder_result[:,1], c=mnist.test.labels)
    plt.colorbar()
    plt.savefig('mnist2_0427.png', dpi=500)