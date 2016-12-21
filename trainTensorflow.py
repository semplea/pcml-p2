# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from functions_helpers import *
from TextCNN import *
import sklearn.preprocessing as preprocessing #used to normalized word_embeddings

# ============
# Parameters
# ============

# Files
tf.flags.DEFINE_string("data", "twitter-datasets/", "Folder containing the data files")
tf.flags.DEFINE_string("embeddings", "", "Word embeddings file")
tf.flags.DEFINE_boolean("full", False, "Use the full dataset instead of the reduced one")

# Model parameters
tf.flags.DEFINE_float("dev_sample_percentage", 0.05, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-5, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 2)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("load_existing_padded_file", True, "Load the existing padded tweet file")

# Parse flags
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

"""
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")
"""

# File paths
data_folder = FLAGS.data
embeddings_file = data_folder + "embeddings" + (("_" + FLAGS.embeddings) if FLAGS.embeddings else "") + ".npy"
x_train_file = data_folder + 'x_train_padded' + ('_full' if FLAGS.full else '') + '.npy'
y_train_file = data_folder + 'y_train' + ('_full' if FLAGS.full else '') + '.npy'
vocab_file = data_folder + 'vocab' + ('_full' if FLAGS.full else '') + '.pkl'

# Run name
timestamp = str(int(time.time()))
run_name = FLAGS.embeddings + ('_full_' if FLAGS.full else '_') + timestamp

#============
# Load data
#============

# Training data
print("Loading data...")
x_train = np.load(x_train_file)
y_train = np.load(y_train_file)
assert x_train.shape[0] == y_train.shape[0]
num_data = y_train.shape[0]
tweet_len = x_train.shape[1]
num_classes = y_train.shape[1]
print(num_data, "tweets of length", tweet_len, "to split between", num_classes, "classes")

# Vocabulary
vocab = load_pickle(vocab_file)

# Load and normalize embeddings
embeddings = load_embeddings(embeddings_file)
embeddings == preprocessing.normalize(embeddings, norm='l2') #TODO not sur if needed or not
print("All data loaded.")

# ==================
# Data Preparation
# ==================

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

#======================
# Creation of the Graph
#======================
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
	  allow_soft_placement = FLAGS.allow_soft_placement,
	  log_device_placement = FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			sequence_length = tweet_len,
			num_classes     = num_classes,
			vocab_size      = len(vocab) + 1, # the + 1 is for the <pad> token
			embedding       = embeddings,
			filter_sizes    = list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters     = FLAGS.num_filters,
			l2_reg_lambda   = FLAGS.l2_reg_lambda)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Keep track of gradient values and sparsity
		"""
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)
		"""

		# Output directory for models and summaries
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", run_name))
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", cnn.loss)
		acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

		# Train Summaries
		train_summary_op = tf.summary.merge([loss_summary, acc_summary])
		# train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables())
		
		# Initialize all variables
		sess.run(tf.global_variables_initializer())
		# sess.run(cnn.W_static.assign(embeddings))
		# sess.run(cnn.W_dynamic.assign(embeddings))

		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			# Run CNN
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
			}
			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
				feed_dict)

			# Log
			if step %10 == 0:
				time_str = datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_batch, y_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			# Run CNN
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: 1.0,
			}
			step, summaries, loss, accuracy = sess.run(
				[global_step, dev_summary_op, cnn.loss, cnn.accuracy],
				feed_dict)

			# Log
			if step %10 == 0:
				time_str = datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			if writer:
				writer.add_summary(summaries, step)

		# Generate batches
		batches = batch_iter(
			list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
		
		# Training loop. For each batch...
		for batch in batches:
			# Train on batch
			x_batch, y_batch = zip(*batch)
			train_step(x_batch, y_batch)
			current_step = tf.train.global_step(sess, global_step)

			# Evaluate
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_step(x_dev, y_dev, writer=dev_summary_writer)
				print("")

			# Checkpoint
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))

		# Final checkpoint
		path = saver.save(sess, checkpoint_prefix, global_step=current_step)
		print("Saved model checkpoint to {}\n".format(path))
