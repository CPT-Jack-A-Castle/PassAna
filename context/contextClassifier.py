import logging
from abc import abstractmethod, ABC

import numpy as np
from datasets import load_dataset
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout
from keras.layers import Embedding
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from tokenizer.tool import MyTokenizer, train_valid_split, load_embedding

MAX_NB_WORDS = 10000


class ContextClassifier:
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
        logging.info(f"X: {train[0].shape} Y:{train[1].shape}")
        self.model.fit(train[0], train[1],
                       epochs=epochs, batch_size=batch_size,
                       validation_data=(valid[0], valid[1]),
                       shuffle=True)

    def words2vec(self, texts, labels, fit=True):
        """
        Map raw texts to int vector
        :param texts: [ [], [],..., [] ]
        :param labels:[]
        :param fit: Whether refit the tokenizer
        :return: texts, labels
        """
        logging.info("pre-processing train data...")

        if fit:
            self.tokenizer.fit_on_texts(texts)
            self.tokenizer.save_tokenizer(f"D:\\program\\PassAna\\tokenizer\\context.pkl")
        else:
            self.tokenizer.load_tokenizer(f"D:\\program\\PassAna\\tokenizer\\context.pkl")
        logging.info(f"Dictionary size: {self.tokenizer.vocab_size()}")

        # integer encode the documents
        encoded_docs = self.tokenizer.texts_to_sequences(texts)
        # pad documents to a max length of 4 words
        texts = pad_sequences(encoded_docs, maxlen=self.padding_len, padding='post')
        # trans label to label type
        labels = to_categorical(labels)

        return texts, labels


class CNNClassifierGlove(ContextClassifier, ABC):
    def __init__(self, padding_len, class_num, glove_dim=50, debug=False):
        super(CNNClassifierGlove, self).__init__(padding_len, class_num, debug)
        if glove_dim not in [50, 100, 200, 300]:
            logging.error(f'Not support this glove_dim -- {glove_dim}, which must in [50, 100, 200, 300]')
            raise ValueError(f'Not support this glove_dim -- {glove_dim}, which must in [50, 100, 200, 300]')

        self.glove_dim = glove_dim
        self.embedding_matrix = None

    def create_model(self):
        """
        create keras model
        :return:
        """

        if self.embedding_matrix is None:
            logging.warning("Get glove 6B matrix at first")
            raise ValueError("Get glove 6B matrix at first")

        logging.info("Create Model...")
        model = Sequential()
        model.add(Embedding(self.tokenizer.vocab_size(), self.glove_dim,
                            weights=[self.embedding_matrix], input_length=self.padding_len, trainable=False))
        model.add(Conv1D(16, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(16, 7, activation='relu', padding='same'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(self.class_num, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    def get_matrix_6b(self, src):
        """
        get embedding_matrix from src
        :param src:
        :param length:
        :return:
        """
        # Certain the glove_dim
        path = f"{src}/glove.6B.{self.glove_dim}d.txt"

        # Load glve 6B
        embeddings_index = load_embedding(path)
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((self.tokenizer.vocab_size(), self.glove_dim))

        for word, i in self.tokenizer.word_index().items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix


if __name__ == '__main__':
    dataset = load_dataset('sst', 'default')

    X = np.array(dataset.data['train'][0])
    Y = np.array(dataset.data['train'][1]).round()

    cnnClassifier = CNNClassifierGlove(padding_len=128, class_num=2, glove_dim=50)

    X, Y = cnnClassifier.words2vec(X, Y, fit=False)
    cnnClassifier.get_matrix_6b(f"D:\\program\\glove.6B")

    train_data, valid_data = train_valid_split(X, Y)

    cnnClassifier.create_model()
    cnnClassifier.run(train_data, valid_data, epochs=32, batch_size=64)
