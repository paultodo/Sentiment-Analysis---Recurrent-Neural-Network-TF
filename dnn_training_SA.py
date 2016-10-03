# -*- coding: utf-8 -*-

### Training ###
import tensorflow as tf
import numpy as np
from dnn_model_SA import nn_model
import datetime

from sklearn.metrics import confusion_matrix
from six.moves import cPickle as pickle
import math
import time
import os

import time
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn.ptb import reader
from utils import imdb_data



def accuracy_numpy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.array(labels))
          / predictions.shape[0])

# Model parameters
tf.flags.DEFINE_integer("nb_hidden_1", 100, "Number of hidden nodes for layer 1 (default: 1024)")
tf.flags.DEFINE_integer("nb_hidden_2", 200, "Number of hidden nodes for layer 2 (default: 800)")
tf.flags.DEFINE_integer("nb_hidden_3", 100, "Number of hidden nodes for layer 3 (default: 512)")
tf.flags.DEFINE_float("keep_prob_layer1", 0.5, "Probability to keep nodes in layer 1 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer2", 0.5, "Probability to keep nodes in layer 2 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer3", 0.5, "Probability to keep nodes in layer 3 (default: 0.6)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("nb_epochs", 5, "Number of training step (default: 5)")
tf.flags.DEFINE_integer("eval_every", 100, "Number of steps between every eval print (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.05, "Initial learning rate (default: 0.0005)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0001)")
tf.flags.DEFINE_integer("embedding_size", 50, "Number of embedding dim (default: 50)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

### LOAD DATA ###

MAXLEN = 100
vocab_size = 100000
print "Loading data..."
train_data, valid_data, test_data = imdb_data.load_data()
test_dataset, max_len_seqs, test_labels = imdb_data.prepare_data(test_data[0], test_data[1], MAXLEN)
print "...Data loaded !"

# Compute var for init 
sqrt_0  = math.sqrt(2.0 / float(FLAGS.embedding_size))
sqrt_1 = math.sqrt(2.0 / float(FLAGS.nb_hidden_1))
sqrt_2 = math.sqrt(2.0 / float(FLAGS.nb_hidden_2))
sqrt_3 = math.sqrt(2.0 / float(FLAGS.nb_hidden_3))

print ("...Start training !")


with tf.Graph().as_default():
    with tf.Session() as session:
        dnn = nn_model(embedding_size = FLAGS.embedding_size,
            vocab_size = vocab_size,
            batch_size = FLAGS.batch_size,
            sequence_length = MAXLEN,
            dp1 = FLAGS.keep_prob_layer1,
            dp2 = FLAGS.keep_prob_layer2,
            dp3 = FLAGS.keep_prob_layer3,
            num_hidden1 = FLAGS.nb_hidden_1,
            num_hidden2 = FLAGS.nb_hidden_2,
            num_hidden3 = FLAGS.nb_hidden_3,
            data_set_size = len(train_data[0]),
            num_classes = 2,
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            test_dataset = test_dataset,
            stdev_init = [sqrt_0, sqrt_1, sqrt_2, sqrt_3])

        global_step = tf.Variable(0)
        init_lr = 0.05
        optimizer = tf.train.AdagradOptimizer(init_lr)
        train_step = optimizer.minimize(dnn._cost)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.scalar_summary("loss", dnn._cost)

        train_summary_op = loss_summary
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, session.graph_def)

        saver = tf.train.Saver(tf.all_variables())
        
        tf.initialize_all_variables().run()
        print("Initialized")
        for epoch in range(FLAGS.nb_epochs):
            print "epoch %d" % epoch
            epoch_size = len(train_data[0])//FLAGS.batch_size
            start_time = time.time()
            costs = 0.0
            correct_answers = 0.0
            seqs, labels = train_data
            MAXLEN = 100
            for step in range(epoch_size):
                x = seqs[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size]
                y = labels[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size]
                x, max_len_seqs, y = imdb_data.prepare_data(x, y, MAXLEN)
                x = x[:,:MAXLEN]
                summaries, cost, prediction, _ = session.run([train_summary_op, dnn._cost, dnn.probs, train_step],
                                             {dnn.input: x,
                                              dnn.labels: y})
                train_summary_writer.add_summary(summaries, step)
                correct_answers += (np.argmax(prediction, 1) == np.array(y)).sum()
                costs += cost

                if step % 300 == 0 and step > 0 :
                    print "At step %d - Loss : %.3f  - Accuracy : %.3f " % (step, costs / step, correct_answers / (step * FLAGS.batch_size))
                    print "Test accuracy : %.3f :" % accuracy_numpy(dnn.test_probs.eval(), test_labels)
                    # print costs / step
