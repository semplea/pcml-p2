import numpy as np
import pickle
import re
from datetime import datetime


def load_data_label(pos, neg):
    with open(pos) as f:
        pos_data = f.read().splitlines()

        f = open(pos)
        pos_data = f.read().splitlines()
    with open(neg) as f:
        neg_data = f.read().splitlines()
    train_data = np.append(pos_data, neg_data)
    positive_labels = [[1, 0] for _ in pos_data]
    negative_labels = [[0, 1] for _ in neg_data]
    label_data = np.concatenate([positive_labels, negative_labels], 0)
    #return train_data, label_data
    return train_data, label_data

def map_data(data, vocab):
    size_vocab = len(vocab)
    max_size = find_max_length(data, vocab)
    output = np.empty([len(data),max_size])
    print('Total number of data: ', len(data))
    for i,tweet in enumerate(data):
        if (i % 10000) == 0:
            print(i)
        idx = [vocab.get(token,-1) for token in tweet.strip().split() if vocab.get(token,-1)>=0]
        output[i] = pad_tweet(idx, max_size, size_vocab)

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
    return tweet

def length_tweet(tweet, vocab):
    return len([vocab.get(token,-1) for token in tweet.strip().split() if vocab.get(token,-1)>=0])


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


#def padTweet(twitterSet):
#    #get the max size among all tweet
#    tweet_lengths = {}
#    for i, l in enumerate(twitterSet):
#        tweet_lengths[i] = len(re.findall(r'\s', l)) + 1
#    max_size = max(tweet_lengths.items(), key=lambda x: x[1])[1]
#    #pad each tweet
#    twitterPadded = []
#
#    print('Totel number of tweets: ', len(twitterSet))
#    print('Max_size: ', max_size)
#    for i, l in enumerate(twitterSet):
#        if i % 1000 == 0:
#            print(i, 'tweet padded')
#        l_size = tweet_lengths[i]
#        twitterPadded = np.append(twitterPadded, l + ' <pad>' * (max_size - l_size))
#
#    print('Padding done')
#    return twitterPadded

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
