# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn.ptb import reader
from utils import imdb_data

def accuracy_numpy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.array(labels))
          / predictions.shape[0])

class nn_model(object):
    """
    A NN for text classif.
    Todo : feed in the dataset in small batches so that it can run in GPU (memory issue)
    So far : restricted size to 2K examples
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, batch_size, test_dataset):

        self.batch_size = batch_size
        self.input = tf.placeholder(tf.int32, shape=(batch_size, sequence_length))
        self.labels = tf.placeholder(tf.int32, shape=(batch_size))
        self.test_dataset = tf.constant(test_dataset)

        embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
        
        with tf.variable_scope('softmax'):
            W_softmax = tf.get_variable('W', [embedding_size, num_classes])
            b_softmax = tf.get_variable('b', [num_classes],initializer=tf.constant_initializer(0.0))
      
        def get_logits(x):      
            inputs = tf.nn.embedding_lookup(embedding, x)
            output = tf.reduce_sum(inputs, 1)
            logits = tf.matmul(output, W_softmax) + b_softmax
            return logits

        logits = get_logits(self.input)        
        self.probs = tf.nn.softmax(logits)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels)
        self._cost = cost = tf.reduce_sum(loss_) / batch_size

        self.test_probs = tf.nn.softmax(get_logits(self.test_dataset))
        self.test_preds = tf.argmax(self.test_probs, 1, name="predictions")

nb_epochs = 5
MAXLEN = 100
print "Loading data..."
train_data, valid_data, test_data = imdb_data.load_data()
test_dataset, max_len_seqs, test_labels = imdb_data.prepare_data(test_data[0], test_data[1], MAXLEN)
print "...Data loaded !"
with tf.Graph().as_default():
    with tf.Session() as session:
        dnn = nn_model(test_dataset = test_dataset[:2000], sequence_length = 100, num_classes = 2, vocab_size = 100000, embedding_size = 100, batch_size = 32)
        global_step = tf.Variable(0)
        init_lr = 0.05
        optimizer = tf.train.AdagradOptimizer(init_lr)
        train_step = optimizer.minimize(dnn._cost)

        tf.initialize_all_variables().run()

        for epoch in range(nb_epochs):
            print "epoch %d" % epoch
            epoch_size = len(train_data[0])//dnn.batch_size
            start_time = time.time()
            costs = 0.0
            correct_answers = 0.0
            seqs, labels = train_data
            MAXLEN = 100
            for step in range(epoch_size):
                x = seqs[step*dnn.batch_size:(step+1)*dnn.batch_size]
                y = labels[step*dnn.batch_size:(step+1)*dnn.batch_size]
                x, max_len_seqs, y = imdb_data.prepare_data(x, y, MAXLEN)
                x = x[:,:MAXLEN]
                cost, prediction, _ = session.run([dnn._cost, dnn.probs, train_step],
                                             {dnn.input: x,
                                              dnn.labels: y})
                correct_answers += (np.argmax(prediction, 1) == np.array(y)).sum()
                costs += cost

                if step % 300 == 0 and step > 0 :
                    print "At step %d - Loss : %.3f  - Accuracy : %.3f " % (step, costs / step, correct_answers / (step * dnn.batch_size))
                    print "Test accuracy : %.3f :" % accuracy_numpy(dnn.test_probs.eval(), test_labels[:2000])