import numpy as np

def load_data_label(pos, neg):
    with open(pos) as f:
        pos_data = f.read().splitlines()
    with open(neg) as f:
        neg_data = f.read().splitlines()
    train_data = np.append(pos_data, neg_data)
    label_data = np.append(np.ones(len(pos_data)), np.ones(len(neg_data)) * -1 )
    return train_data, label_data

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
