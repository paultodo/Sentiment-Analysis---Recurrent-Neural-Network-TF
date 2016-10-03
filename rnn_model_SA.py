import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class rnn_model(object):
    """
    A RNN for predicting output based on long term dependancies.
    """
    def __init__(self, data_size, num_layers,
        stddev_init, batch_size, state_size, num_steps, vocab_size, num_classes, cell_type = 'LSTM'):

        self.batch_size = batch_size

        if cell_type == 'BASIC':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif cell_type == 'GRU':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif cell_type == 'LSTM':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(cell_type))

        self.x = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name="input_x")
        self.y = tf.placeholder(tf.int32, shape=[batch_size], name='labels_placeholder')
        cell = cell_fn(state_size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
     
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, state_size], -0.5, 0.5))
            rnn_inputs = [tf.squeeze(i) for i in tf.split(1, num_steps, tf.nn.embedding_lookup(embedding, self.x))]

        rnn_outputs, last_state = tf.nn.rnn(cell, rnn_inputs, dtype=tf.float32)

   
        W_softmax = tf.Variable(tf.random_normal([state_size, num_classes]))
        b_softmax = tf.Variable(tf.random_normal([num_classes]))
        self.logits = tf.matmul(rnn_outputs[-1], W_softmax) + b_softmax 
        
      
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y)
        self.loss = tf.reduce_mean(self.losses)

        # Accuracy
        self.probs = tf.nn.softmax(self.logits)
        self.predictions_label = tf.argmax(self.probs, 1, name="predictions")
        self.correct_predictions = tf.equal(self.predictions_label, tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name="accuracy")
