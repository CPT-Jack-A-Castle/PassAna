import logging
from abc import abstractmethod

import nltk
import numpy as np
from datasets import load_dataset
from keras.layers import Dense, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import to_categorical
from tqdm import tqdm

from tool import MyTokenizer, train_valid_split

MAX_NB_WORDS = 10000


class Predictor(object):
    def __init__(self, padding_len, class_num, debug=False):
        self.padding_len: int = padding_len
        self.class_num = class_num

        self.model: Sequential = None
        self.tokenizer: MyTokenizer = MyTokenizer()

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

    def processing_texts(self, texts):
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
        tokenizer = MyTokenizer(num_words=MAX_NB_WORDS, lower=False, char_level=False)
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

        return word_seq_train

    def words2vec(self, texts, labels, fit=True):
        """
        Map raw texts to int vector
        :param texts: [ [], [],..., [] ]
        :param labels:[]
        :param fit: Whether refit the tokenizer
        :return: texts, labels
        """

        # process to word
        texts = self.processing_texts(texts)

        # Load or fit dict()
        if fit:
            self.fit_words_dict(texts)
            self.tokenizer.save_tokenizer(f"D:\\program\\PassAna\\tokenizer\\pwd.pkl")
        else:
            self.tokenizer.load_tokenizer(f"D:\\program\\PassAna\\tokenizer\\pwd.pkl")
        logging.info(f"Dictionary size: {self.tokenizer.vocab_size()}")

        # words to vector
        texts = self.tokenizer_words(texts)

        # padding the cols to padding_len
        texts = sequence.pad_sequences(texts, maxlen=self.padding_len)
        # trans label to label type
        labels = to_categorical(labels)

        return texts, labels


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
        model.add(Embedding(self.tokenizer.vocab_size(), self.embedding_dim, input_length=self.padding_len))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(self.class_num, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model


if __name__ == '__main__':
    dataset = load_dataset('sst', 'default')

    X = np.array(dataset.data['train'][0])
    Y = np.array(dataset.data['train'][1]).round()

    fastTextPredictor = FastTextPredictor(padding_len=128, class_num=2)
    X, Y = fastTextPredictor.words2vec(X, Y, False)
    train_data, valid_data = train_valid_split(X, Y)

    fastTextPredictor.create_model()
    fastTextPredictor.run(train_data, valid_data, epochs=25, batch_size=64)
