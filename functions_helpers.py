import numpy as np
import pickle
import re

def load_data_label(pos, neg):
    with open(pos) as f:
        pos_data = f.read().splitlines()

        f = open(pos)
        pos_data = f.read().splitlines()
    with open(neg) as f:
        neg_data = f.read().splitlines()
    train_data = np.append(pos_data, neg_data)
    label_data = np.append(np.ones(len(pos_data)), np.ones(len(neg_data)) * -1 )
    return padTweet(train_data), label_data

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def padTweet(twitterSet):
    #get the max size among all tweet
    tweet_lengths = {}
    for i, l in enumerate(twitterSet):
        tweet_lengths[i] = len(re.findall(r'\s', l)) + 1

    max_size = max(tweet_lengths.items(), key=lambda x: x[1])[1]

    #pad each tweet
    twitterPadded = []

    for i, l in enumerate(twitterSet):
        l_size = tweet_lengths[i]
        twitterPadded = np.append(twitterPadded, l + ' <pad>' * (max_size - l_size))

    return twitterPadded
