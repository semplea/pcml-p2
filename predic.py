# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from functions_helpers import *
from TextCNN import *
import sys # to be removed at the end

# ============
# Parameters
# ============

# Files
tf.flags.DEFINE_string("data", "twitter-datasets/", "Folder containing the data files")
tf.flags.DEFINE_boolean("full", False, "Use the full dataset instead of the reduced one")
tf.flags.DEFINE_string("name", "", "Name of the run to use")
tf.flags.DEFINE_integer("size", 63, "Maximal number of words in the tweets")

# Misc Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# File paths
data_folder = FLAGS.data
vocab_file = data_folder + 'vocab' + ('_full' if FLAGS.full else '') + '.pkl'
test_file = data_folder + 'test_data.txt'
checkpoint_dir = 'runs/' + FLAGS.name + '/checkpoints/'

#============
# Load data
#============
vocab = load_pickle(vocab_file)
x_test = load_test(test_file)
x_test = map_data(x_test, vocab, max_size=FLAGS.max_size)
print('Test set shape', x_test.shape)

#======================
# Creation of the Graph
#======================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

    # Get the placeholders from the graph by name
    input_x = sess.graph.get_operation_by_name("input_x").outputs[0]
    #input_y = graph.get_operation_by_name("input_y").outputs[0]
    dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    # Tensors we want to evaluate
    predictions = sess.graph.get_operation_by_name("output/predictions").outputs[0]

    # Generate batches for one epoch
    batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []

    for x_test_batch in batches:
        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])

    all_predictions = np.where(all_predictions == 0,1, -1)
    
    # Save the evaluation to a csv
    output_path = data_folder + 'predictions.csv'
    all_predictions = np.column_stack((list(range(1,len(all_predictions) + 1)), all_predictions))
    all_predictions = np.row_stack((['Id','Prediction'], all_predictions))

    print("Saving evaluation to {0}".format(output_path))
    np.savetxt(output_path, all_predictions, delimiter=',', fmt='%s')
