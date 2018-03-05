""" Simple LSTM Implementation with Basic LSTM CELL and Static RNN witha hidden size taking
    runs on radom single input and random outputs between 0-10
    Change the logits layer place holder size to match the required label 
    Change the inputs placeholder and X_split to match the sequence you wish to give input
    let the iterator to run through as inputs are passed in as time steps 
    Same skelton can be upgraded required application changing 
    Optimizer, Loss function, Input and Output Place holders, and static or dynamic RNN
    ----------------------------------------------------------------------------------------
    Ramachandra Vikas Chamarthi
    Graduate Research Assistant @ The UNC Charlotte (www.uncc.edu)
    vikaschamarthi240@gmail.com
    -----------------------------------------------------------------------------------------
""" 

#Import Required modules

import numpy as np 
import random 
import tensorflow as tf
import tensorflow.contrib.layers as layers
from random import *
from tensorflow.contrib import rnn

map_fn = tf.map_fn

#Placeholder for feeding in values to Network

inputs = tf.placeholder(tf.float32, shape = (1,1))
outputs = tf.placeholder(tf.float32, shape = (1,10))

# Cell to be Used
cell = rnn.BasicLSTMCell(10, forget_bias=1.0, state_is_tuple=True)

batch_size = 1 

initial_state = cell.zero_state(batch_size, tf.float32)

# Split it in to time steps were 1 in below splits them in to 1 time step
X_split = tf.split(inputs, 1, 0)
Y_split = tf.split(outputs, 1, 0)

#Declare the type of RNN 
rnn_outputs, rnn_states = tf.nn.static_rnn(cell, X_split, initial_state = initial_state)

# Optimizer Creation
final_projection = lambda x: layers.linear(x, num_outputs=1, activation_fn=tf.nn.sigmoid)

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_outputs, labels=outputs))

train_fn = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)


ITERATIONSPEREPOCH = 100
# Training starts
session = tf.Session()
session.run(tf.initialize_all_variables()) 
for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        epoch_error += session.run([error, train_fn], {
            inputs : np.random.rand(1, 1),
            outputs : np.random.rand(1, 10),
        })[0]
    epoch_error /= ITERATIONSPEREPOCH
    valid_accuracy = session.run(error, {
        inputs: np.random.rand(1, 1),
        outputs: np.random.rand(1, 10),
    })
    print "epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)
