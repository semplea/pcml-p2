# PCML Project 2
Loïc Ottet, Baptiste Raemy, Alexis Semple

## Initial data
In folder `twitter-datasets/`
- `pos_train.txt`
- `neg_train.txt`

The steps described below are analoguous for the full dataset (`pos_train_full.txt` and `neg_train_full.txt`). All scripts act on the reduced datasets, but they can take `full` as their last argument to act to the full dataset.

## Preprocessing steps (execute while in `twitter-datasets/` folder)

1. Build the vocabulary (`build_vocab.sh`). Yields `vocab.txt`
2. Cut the vocabulary (`cut_vocab.sh`). Yields `vocab_cut.txt`
3. Convert the vocabulary to a python dictionary mapping words to ids (`pickle_vocab.py`). Yields `vocab.pkl`

## Word embedding computation (execute while in `twitter-datasets/` folder)

### Provided GloVe algorithm (`glove-basic`)
1. Compute the cooccurence matrix (`cooc.py`). Yields `cooc.pkl`
2. Compute the GloVe matrix (`glove_solution.py`). Yields `embeddings_glove-basic.npy`

### Stanford GloVe vectors (`glove`)
In what follows, `**` represents the number of features of the word vector (25, 50, 100 or 200)

1. The vectors are in `twitter-datasets/glove.twitter.27B.**d.txt`
2. Compute the word embedding for a given dimension (`filterVocab.py **`). Yields `embeddings_glove**.npy`)

### FastText (`fasttext`)
1. Compute the fastText vectors (`./fasttext skipgram -input data.txt -output model`). Yields `model.vec` (`data.txt` is a concatenation of the positive and negative train sets)
2. Compute the word embedding for our vocabulary (`filterVocabFastText.py`). Yields `embeddings_fasttext.npy`

## Network training (execute while in main folder)
To apply `trainTensorflow.py` and `predic.py` to the full dataset, use `--full`

1. Load and pad the training data (`loadData.py`). Yields `x_train_padded.npy` and `y_train.npy`
2. Train the neural network (`trainTensorflow.py --embeddings=***`), where `***` is the name of the chosen embedding (`glove-basic`, `glove**` or `fasttext`). Yields detailed run data in `uns/***_****/`, where `****` is the timestamp of the run
3 . Generate predictions from the test set (`predic.py --name=***_****`). Yields `predictions.csv`

## External libraries uses

- [TensorFlow](https://www.tensorflow.org)
- [FastText](https://research.fb.com/projects/fasttext/)

## External datasets used

- [Stanford GloVe Twitter data](http://nlp.stanford.edu/data/glove.twitter.27B.zip "Download link")