# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from functions_helpers import *
from TextCNN import *

data_folder = 'twitter-datasets/'
embeddings_dim = 20
pos_train_file = data_folder + 'pos_train.txt'
neg_train_file = data_folder + 'neg_train.txt'
vocab_pickle = data_folder + 'vocab.pkl'
cooc_pickle = data_folder + 'cooc.pkl'
embeddings_file = data_folder + 'embeddings.npy'
filter_sizes = "3,4,5" # must be a string, not array of int
num_filters = 128

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", pos_train_file, "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", neg_train_file, "Data source for the positive data.")
tf.flags.DEFINE_string("vocab_file", vocab_pickle, "Data source for the positive data.")
tf.flags.DEFINE_string("embeddings_file", embeddings_file, "Data source for the positive data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# Load data
print("Loading data...")
x_train, y_train = load_data_label(FLAGS.positive_data_file, FLAGS.negative_data_file)
print('X_train shape', x_train.shape,'Y_train shape',  y_train.shape)
#np.savetxt('x_train_padded.txt', x_train)
#np.savetxt('y_train_padded.txt', y_train)

# load_vocab
vocab = load_pickle(FLAGS.vocab_file)
embeddings = np.load(FLAGS.embeddings_file)

#add one fake word embedding for the random
embeddings = np.vstack([embeddings, np.ones([1,embeddings.shape[1]])*-999])

print(embeddings.shape)
print(len(vocab))

# In[ ]:

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# Load data
print("Loading data...")
x_train, y_train = load_data_label(FLAGS.positive_data_file, FLAGS.negative_data_file)
print('X_train shape', x_train.shape,'Y_train shape',  y_train.shape)
#np.savetxt('x_train_padded.txt', x_train)
#np.savetxt('y_train_padded.txt', y_train)

# load_vocab
vocab = load_pickle(FLAGS.vocab_file)

embeddings = np.load(FLAGS.embeddings_file)
#add one fake word embedding for the random
embeddings = np.vstack([embeddings, np.ones([1,embeddings.shape[1]])*-999])

print(embeddings.shape)
print(len(vocab))


x_train = map_data(x_train, vocab)
print('==============')
print(x_train.shape)
print('==============')
# In[ ]:

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", embeddings_dim, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", filter_sizes, "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", num_filters, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices]


x_shuffled = x_shuffled
y_shuffled = y_shuffled


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index =-1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

vocab_size = len(vocab) + 1
embedding_dim
dummy_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length= x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab) + 1, # the + 1 is for the <pad> token
            embedding_dim= embeddings.shape[1],
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
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

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab")) vocab already exists

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.now().isoformat()
            if step %10 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.now().isoformat()
            if step %10 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
