from keras_preprocessing.text import Tokenizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def load_pkl(src):
    with open(src, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(src, obj):
    with open(src, 'wb') as f:
        pickle.dump(obj, f)


class MyTokenizer(object):
    def __init__(self, **kwargs):
        self.tokenizer = Tokenizer(**kwargs)

    def fit_on_seq(self, processed_docs_train):
        return self.tokenizer.fit_on_sequences()

    def fit_on_texts(self, processed_docs_train):
        return self.tokenizer.fit_on_texts(processed_docs_train)

    def decode_vectors(self, vectors):
        return self.tokenizer.sequences_to_texts(vectors)

    def vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def texts_to_sequences(self, processed_docs_train):
        return self.tokenizer.texts_to_sequences(processed_docs_train)

    def word_index(self):
        return self.tokenizer.word_index

    def load_tokenizer(self, src):
        with open(src, "rb") as f:
            self.tokenizer = pickle.load(f)

    def save_tokenizer(self, src):
        with open(src, "wb") as f:
            pickle.dump(self.tokenizer, f)


def load_data(src):
    """
        Load text
        :param src:
        :return:
        """
    data_df = pd.read_csv(src)
    X = data_df['string']
    Y = data_df['label']
    return X, Y


def train_valid_split(X, Y):
    """
        Split the value to train and valid
        :param X: X[[],[]]
        :param Y: Y[,]
        :return:
        """
    train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.1, random_state=2022, shuffle=True)
    return [train_X, np.array(train_Y, dtype=int)], [valid_X, np.array(valid_Y, dtype=int)]


def load_embedding(src):
    """
    load embeddings_index from src
    :param src: path that include the file.
    :return: embeddings_index
    """
    embeddings_index = dict()
    f = open(src, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
