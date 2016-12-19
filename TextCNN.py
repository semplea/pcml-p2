import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(
        self, sequence_length, num_classes, vocab_size,
        embedding_dim, filter_sizes, num_filters, embedding_vectors):
        """
        sequence_length: length of our sentences (all must have the same length: pad all sentences)
        num_classes: number of classes in the output layer (2: positiv and negative)
        vocab_size : the vocab dictionnary
        embedding_dim: the embedding vectors matrix
        filter_sizes: the number of words we want our convolutional to cover. we will have num_filters
                    specified size ex: [3,4,5]
        num_filters: the number of filter per filter size
        """
        #create Variable
        #embedding_dim = embedding_vectors.shape[1]
        #vocab_size = len(vocab)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #First layer: Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W_static = tf.get_variable(
                name="W_static",
                shape=[vocab_size, embedding_dim],
                initializer=tf.constant_initializer(np.array(embedding_vectors)),
                trainable = False)
            self.embedded_chars_static = tf.nn.embedding_lookup(W_static, self.input_x)
            self.embedded_chars_expanded_static = tf.expand_dims(self.embedded_chars_static, -1)

            W_dynamic = tf.get_variable(
                name="W_dynamic",
                shape=[vocab_size, embedding_dim],
                initializer=tf.constant_initializer(np.array(embedding_vectors)),
                trainable = True)
            self.embedded_chars_dynamic = tf.nn.embedding_lookup(W_dynamic, self.input_x)
            self.embedded_chars_expanded_dynamic = tf.expand_dims(self.embedded_chars_dynamic, -1)

            self.W_concat = tf.concat(3, [self.embedded_chars_expanded_static, self.embedded_chars_expanded_dynamic], name='W_concat')

        # with tf.device('/cpu:0'), tf.name_scope("embedding_dynamic"):
        #    W_dynamic = tf.get_variable(
        #        name="W",
        #        shape=[vocab_size, embedding_dim],
        #        initializer=tf.constant_initializer(np.array(embedding_vectors)),
        #        trainable = True)
        #    self.embedded_chars_dynamic = tf.nn.embedding_lookup(self.W_dynamic, self.input_x)
        #    self.embedded_chars_expanded_dynamic = tf.expand_dims(self.embedded_chars_dynamic, -1)


        # Convolution and max pooling layers
        pooled_outputs = []
        embedded_chars_expandeds = [self.W_concat]

        for i, filter_size in enumerate(filter_sizes):
            for j,emb in enumerate(embedded_chars_expandeds):
                with tf.name_scope("conv-maxpool-{}-{}".format(filter_size, j)):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_dim, 2, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        emb,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) * len(embedded_chars_expandeds) #TODO *2 is because of the 2 channels. Not sure
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

            # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
