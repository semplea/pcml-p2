#!/usr/bin/env python3
import pickle
import sys

def main(full=False):
    vocab = dict()
    with open('vocab_cut' + ('_full' if full else '') + '.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab' + ('_full' if full else '') + '.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(len(sys.argv) >= 2 and sys.argv[1] == "full")
