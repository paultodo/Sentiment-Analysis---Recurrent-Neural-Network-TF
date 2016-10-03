import tensorflow as tf
import numpy as np
import datetime
from rnn_model_SA import rnn_model
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn

from tensorflow.models.rnn.ptb import reader
from sklearn.metrics import confusion_matrix
from six.moves import cPickle
import math
import time
import os
import matplotlib.pyplot as plt
from utils import imdb_data
from utils import data_helpers


#todo : FAIRE AVEC BONNES DATA IMDB DONC PADDING.


tf.flags.DEFINE_integer("state1_size", 50, "Number of hidden nodes for cell 1 (default: 100)")
tf.flags.DEFINE_integer("state2_size", 50, "Number of hidden nodes for cell 2 (default: 100)")
tf.flags.DEFINE_integer("state3_size", 50, "Number of hidden nodes for cell 3 (default: 100)")
tf.flags.DEFINE_float("keep_prob_layer1", 0.9, "Probability to keep nodes in layer 1 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer2", 0.8, "Probability to keep nodes in layer 2 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer3", 0.7, "Probability to keep nodes in layer 3 (default: 0.6)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("nb_epochs", 10, "Number of training step (default: 2)")
tf.flags.DEFINE_integer("eval_every", 50, "Number of steps between every eval print (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.005, "Initial learning rate (default: 0.0005)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00000, "L2 regularizaion lambda (default: 0.0001)")
tf.flags.DEFINE_integer("num_steps", 56, "Num steps (default: 10)")
tf.flags.DEFINE_integer("num_classes", 2, "Num classes (default: 2)")
tf.flags.DEFINE_boolean("verbose", True, "Use verbose (default: True)")
tf.flags.DEFINE_integer("num_layers", 1, "Num layers (default: 1)")
tf.flags.DEFINE_string("cell_type", 'BASIC', "Cell type (default: 'BASIC')")
tf.flags.DEFINE_string("is_training", 'TRUE', "us dropout or not")
tf.flags.DEFINE_string("save_dir", 'save', "Where to save models / params (default: 'save')")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

MAXLEN = 30
vocab_size = 10000

train_data, valid_data, test_data = imdb_data.load_data(n_words = vocab_size)
test_dataset, max_len_seqs, test_labels = imdb_data.prepare_data(test_data[0], test_data[1], MAXLEN)

data_size = len(train_data[0])


config=tf.ConfigProto(log_device_placement=True)
with tf.Graph().as_default():
    with tf.Session(config=config) as sess:
 
        sqrt_0 = math.sqrt(1.0 / float(FLAGS.state1_size))
        sqrt_1 = math.sqrt(1.0 / float(FLAGS.state1_size))
        sqrt_2 = math.sqrt(1.0 / float(FLAGS.state1_size))
        sqrt_3 = math.sqrt(1.0 / float(FLAGS.state1_size))
        list_stddev = [sqrt_0, sqrt_1, sqrt_2, sqrt_3]
        current_list = [sqrt for sqrt in list_stddev[:FLAGS.num_layers]]

        rnn = rnn_model(batch_size = FLAGS.batch_size,
                        state_size = FLAGS.state1_size,
                        num_steps = MAXLEN,
                        num_classes = 2,
                        stddev_init = [sqrt_0, sqrt_1],
                        num_layers = FLAGS.num_layers,
                        cell_type = FLAGS.cell_type,
                        data_size = data_size,
                        vocab_size = vocab_size)
  
        
        training_losses = []

        global_step = tf.Variable(0)
        init_lr = FLAGS.learning_rate

        optimizer = tf.train.AdagradOptimizer(init_lr)
        train_step = optimizer.minimize(rnn.loss)

        tf.initialize_all_variables().run()

        saver = tf.train.Saver(tf.all_variables())

        start_t = time.time()

        for epoch in range(FLAGS.nb_epochs):
            print "epoch %d" % epoch
            epoch_size = data_size // rnn.batch_size
            start_time = time.time()
            costs = 0.0
            correct_answers = 0.0
            seqs, labels = train_data
            for step in range(epoch_size):
                x = seqs[step*rnn.batch_size:(step+1)*rnn.batch_size]
                y = labels[step*rnn.batch_size:(step+1)*rnn.batch_size]
                x, max_len_seqs, y = imdb_data.prepare_data(x, y, MAXLEN)
                x = x[:,:MAXLEN]
                cost, prediction, _ = sess.run([rnn.loss, rnn.probs, train_step],
                                             {rnn.x: x,
                                              rnn.y: y})
                correct_answers += (np.argmax(prediction, 1) == np.array(y)).sum()
                costs += cost

                if step % 300 == 0 and step > 0 :
                    print "At step %d - Loss : %.3f  - Accuracy : %.3f " % (step, costs / step, correct_answers / (step * rnn.batch_size))



