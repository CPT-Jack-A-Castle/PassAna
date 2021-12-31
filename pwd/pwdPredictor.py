import logging
import os
from abc import abstractmethod

import nltk
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from datasets import load_dataset
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tqdm import tqdm

MAX_NB_WORDS = 10000


class Predictor(object):
    def __init__(self, padding_len, class_num, debug=False):
        self.padding_len: int = padding_len
        self.class_num = class_num

        self.model: Sequential = None
        self.tokenizer: Tokenizer = None
        self.max_word: int = None

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - '
                                       '%(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - '
                                       '%(levelname)s: %(message)s',
                                level=logging.INFO)

    @abstractmethod
    def create_model(self):
        pass

    def run(self, train, valid, epochs, batch_size):
        if self.model is None:
            logging.error('Model is None')
            raise ValueError('Create model at first!')
        print(f"X: {train[0].shape} Y:{train[1].shape}")
        self.model.fit(train[0], train[1],
                       epochs=epochs, batch_size=batch_size,
                       validation_data=(valid[0], valid[1]),
                       shuffle=True)

    def pre_processing_str(self, texts):
        """
        Divide the texts to words
        :param texts:
        :return:
        """
        logging.info("pre-processing train data...")
        processed_docs_train = []
        tokenizer = nltk.SyllableTokenizer()
        for doc in tqdm(texts):
            # Remove Space
            doc = doc.replace(' ', '')
            tokens = tokenizer.tokenize(doc)
            processed_docs_train.append(" ".join(tokens))
        return processed_docs_train

    def fit_words_dict(self, processed_docs_train):
        logging.info("Tokenizing input data...")
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, char_level=False)
        tokenizer.fit_on_texts(processed_docs_train)
        self.tokenizer = tokenizer

    def tokenizer_words(self, processed_docs_train):
        """
        Leverage tokenizer to map texts
        :param processed_docs_train: texts
        :return:
        """
        if self.tokenizer is None:
            logging.error('Load or fit tokenizer at first!')
            raise ValueError('Load or fit tokenizer at first!')

        # Process
        word_seq_train = self.tokenizer.texts_to_sequences(processed_docs_train)
        word_index = self.tokenizer.word_index
        self.max_word = len(word_index)
        logging.info("Dictionary size: ", len(word_index))

        return word_seq_train

    def words2vec(self, texts, labels, fit=True):
        """
        Map raw texts to int vector
        :param texts: [ [], [],..., [] ]
        :param labels:[]
        :param fit: Whether refit the tokenizer
        :return: texts, labels
        """

        texts = self.pre_processing_str(texts)
        if fit:
            self.fit_words_dict(texts)
            self.save_tokenizer('a')
        else:
            self.load_tokenizer('b')
        texts = self.tokenizer_words(texts)

        # padding the cols to padding_len
        texts = sequence.pad_sequences(texts, maxlen=self.padding_len)
        # trans label to label type
        labels = to_categorical(labels)
        return texts, labels

    def load_tokenizer(self, src):
        # TODO
        pass

    def save_tokenizer(self, src):
        # TODO
        pass

    @staticmethod
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

    @staticmethod
    def train_valid_split(X, Y):
        """
        Split the value to train and valid
        :param X: X[[],[]]
        :param Y: Y[,]
        :return:
        """
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=0)
        return [train_X, np.array(train_Y, dtype=int)], [valid_X, np.array(valid_Y, dtype=int)]


class FastTextPredictor(Predictor):
    def __init__(self, padding_len, class_num, embedding_dim=50, debug=False):
        super(FastTextPredictor, self).__init__(padding_len, class_num, debug)
        self.embedding_dim = embedding_dim
        
    def create_model(self):
        """
        create keras model
        :return:
        """
        logging.info("Create Model...")
        model = Sequential()
        model.add(Embedding(self.max_word, self.embedding_dim, input_length=self.padding_len))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(self.class_num, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model


if __name__ == '__main__':
    dataset = load_dataset('sst', 'default')

    X = np.array(dataset.data['train'][0])
    Y = np.array(dataset.data['train'][1]).round()

    fastTextPredictor = FastTextPredictor(128, 2)
    X, Y = fastTextPredictor.words2vec(X, Y, True)
    train_data, valid_data = FastTextPredictor.train_valid_split(X, Y)

    fastTextPredictor.create_model()
    fastTextPredictor.run(train_data, valid_data, epochs=25, batch_size=64)
