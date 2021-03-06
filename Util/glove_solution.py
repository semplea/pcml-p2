#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] == 'small':
        cooc_file = 'cooc.pkl'
        embeddings = 'embeddings_glove-basic'
    elif sys.argv[1] == 'full':
        cooc_file =  'cooc_full.pkl'
        embeddings = 'embeddings_glove-basic_full'
    else:
        print("Give as argument 'small' to use only 10% of the data set of 'full' to run over the whole set")
        return 


    print("loading cooccurrence matrix")
    with open(cooc_file, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 25
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(embeddings, xs)


if __name__ == '__main__':
    main()
