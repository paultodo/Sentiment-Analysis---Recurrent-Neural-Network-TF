# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class nn_model(object):
    """
    A deep NN for text classif
    """
    def __init__(self, embedding_size, vocab_size, sequence_length, stdev_init, num_classes, batch_size, test_dataset, data_set_size, num_hidden1, num_hidden2, num_hidden3, dp1, dp2, dp3, l2_reg_lambda):

        self.input = tf.placeholder(tf.int32, shape=(batch_size, sequence_length))
        self.labels = tf.placeholder(tf.int32, shape=(batch_size))
        self.test_dataset = tf.constant(test_dataset)
        
        embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
        
        # Variables
        w1 = tf.Variable(tf.truncated_normal([embedding_size, num_hidden1], stddev = stdev_init[0]))
        b1 = tf.Variable(tf.zeros([num_hidden1]))
      
        w2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev = stdev_init[1]))
        b2 = tf.Variable(tf.zeros([num_hidden2]))
        
        w3 = tf.Variable(tf.truncated_normal([num_hidden2, num_hidden3], stddev = stdev_init[2]))
        b3 = tf.Variable(tf.zeros([num_hidden3])) 
        
        w4 = tf.Variable(tf.truncated_normal([num_hidden3, num_classes], stddev = stdev_init[3]))
        b4 = tf.Variable(tf.zeros([num_classes]))
    


        def get_nn_model(dataset, use_dropout, batch_normalization) :
                
            inputs = tf.nn.embedding_lookup(embedding, dataset)
            output = tf.reduce_sum(inputs, 1)

            if batch_normalization:
                h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(output, w1) + b1))
            else :
                h1 = tf.nn.relu(tf.matmul(output, w1) + b1)
            if use_dropout:
                logits_h1 = tf.nn.dropout(h1, dp1)
            else : 
                logits_h1 = h1
            
            if batch_normalization:
                h2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h1,w2) + b2))
            else:
                h2 = tf.nn.relu(tf.matmul(logits_h1, w2) + b2)
            if use_dropout:
                logits_h2 = tf.nn.dropout(h2, dp2)
            else :
                logits_h2 = h2
            
            if batch_normalization:
                h3 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h2, w3) + b3))
            else:
                h3 = tf.nn.relu(tf.matmul(logits_h2, w3) + b3)
            if use_dropout:
                logits_h3 = tf.nn.dropout(h3,dp3)
            else:
                logits_h3 = h3
            
            if batch_normalization:
                logits = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(logits_h3, w4) + b4))
            else:
                logits = tf.matmul(logits_h3, w4) + b4
            return logits

        self.logits_train = get_nn_model(self.input, False, True)

        self.reg = l2_reg_lambda * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
        self.loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_train, self.labels) 
        self._cost = cost = tf.reduce_sum(self.loss_) / batch_size + self.reg
    
        # Predictions for the training, validation, and test data.
        self.probs = tf.nn.softmax(self.logits_train)
        self.train_preds = tf.argmax(self.logits_train, 1, name="predictions")
       
        self.test_probs = tf.nn.softmax(get_nn_model(self.test_dataset, False, True))
        self.test_preds = tf.argmax(self.test_probs, 1)

   