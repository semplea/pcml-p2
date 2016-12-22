import numpy as np
import pickle
import re
from datetime import datetime

data_folder = 'twitter-datasets/'

def load_data_label(pos, neg):
    with open(pos) as f:
        pos_data = f.read().splitlines()
    with open(neg) as f:
        neg_data = f.read().splitlines()
    train_data = np.append(pos_data, neg_data)
    positive_labels = [[1, 0] for _ in pos_data]
    negative_labels = [[0, 1] for _ in neg_data]
    label_data = np.concatenate([positive_labels, negative_labels], 0)
    return train_data, label_data


def map_data(data, vocab, max_size=None, save=False):
    size_vocab = len(vocab)
    if max_size == None:
        max_size = find_max_length(data, vocab)
    output = np.empty([len(data),max_size])
    print('Total number of data: ', len(data))
    for i,tweet in enumerate(data):
        if (i % 10000) == 0:
            print(i)
        idx = [vocab.get(token,-1) for token in tweet.strip().split() if vocab.get(token,-1)>=0]
        output[i] = pad_tweet(idx, max_size, size_vocab)

    if save:
        np.save(data_folder + 'x_train_padded', output)
    return output

def find_max_length(data, vocab):
    """
    data: list of string
    """
    max_size = 0
    for tweet in data:
        max_size = max(max_size, length_tweet(tweet, vocab))
    return max_size

def pad_tweet(tweet, max_size, dummy_idx):
    size = len(tweet)
    if size < max_size:
        tweet = np.append(tweet, [dummy_idx] * (max_size-size))
    else:
        tweet = tweet[:max_size]       
    return tweet

def length_tweet(tweet, vocab):
    return len([vocab.get(token,-1) for token in tweet.strip().split() if vocab.get(token,-1)>=0])

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_embeddings(file):
    """
    load word embeddings and add a dummy word that is the mean of all word embeddings.
    That dummy word will correspond to the <pad> word
    """
    embeddings = np.load(file)
    return np.vstack([embeddings, np.mean(embeddings, axis=0)])

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_test(file):
    with open(file) as test:
        test = test.read().splitlines()
        output = [None] * len(test)
        for i, tweet in enumerate(test):
            regex =  re.match( r'(\d+)(,)(.+)', tweet )

            tweet = regex.group(3)
            output[i] = tweet
        return output


def convert_prediction(predictions):
    def convert(x):
        print(x)
        if x[0] == 1:
            return 1
        else:
            return -1

    output = np.empty([len(predictions), 2])
    for i, l in enumerate(predictions):
        output[i] = [i, convert(l)]

    return output
