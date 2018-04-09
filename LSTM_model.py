""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 500
display_step = 10

# Network Parameters
num_input = 9 # 90 subcarriers
timesteps = 10 # timesteps
num_hidden = 400 # hidden layer num of features as specified 
#TODO: change to one hot encdoing!
num_classes = 2 # binary classification of 

#import datasets as numpy arrays that can be easily loaded into tensorflow Dataset
features = np.array(list(csv.reader(open("hallway_data_combined.csv", "rt"), delimiter=","))).astype("float")
labels = np.array(list(csv.reader(open("hallway_labels_combined.csv", "rt"), delimiter=","))).astype("float")

# this operation builds batches from the training data
def batching(size):
    data_rows = np.shape(features)[0]
    #print(data_rows)
    batch_data = np.zeros([size,90])
    batch_labels = np.zeros([size,num_classes])
    for i in range(size):
        index = np.random.randint(data_rows)
        #print(index)
        batch_data[i] = features[index]
        batch_labels[i] = labels[index]
    return batch_data,batch_labels

test_data, test_labels = batching(50)
print (test_data)
print (test_labels)


# Assume that each row of `features` corresponds to the same row as `labels`.
#assert features.shape[0] == labels.shape[0]

#dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Shuffle, repeat, and batch the examples.
#dataset = dataset.shuffle(1000).repeat().batch(batch_size)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
#use get_variable
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    #or dynamic -> high dimension tensor, batch or time major
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

#1 input no output for train node 
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        #print (dataset.batch(batch_size))
        #batch = dataset.batch(batch_size)
        batch_x,batch_y = batching(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128

    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #TODO: check if this has correct shape
    #test_set_size = 1000
    #np.random.shuffle(arr)
    #test_y = 
    #batch_x = batch_x.reshape((test_set_size, timesteps, num_input))
    #TODO: check if this has correct shape
    #test_label = 
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
