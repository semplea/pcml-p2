import numpy as np
import sys
from functions_helpers import load_data_label, load_pickle, map_data

assert len(sys.argv) >= 2
data_folder = sys.argv[1]
full = len(sys.argv) >= 3 and sys.argv[2] == 'full'
positive_data_file = data_folder + 'pos_train' + ('_full' if full else '') + '.txt'
negative_data_file = data_folder + 'neg_train' + ('_full' if full else '') + '.txt'
vocab_file = data_folder + 'vocab' + ('_full' if full else '') + '.pkl'

# Load data
print("Loading data...")
x_train, y_train = load_data_label(positive_data_file, negative_data_file)
print('X_train shape', x_train.shape,'Y_train shape',  y_train.shape)
vocab = load_pickle(vocab_file)
print('Vocab shape', len(vocab))

#pad and transform x_train
x_train_padded = map_data(x_train, vocab)

# Save padded x and y
np.save(data_folder + "x_train_padded" + ("_full" if full else ''), x_train_padded)
np.save(data_folder + "y_train" + ("_full" if full else ''), y_train)